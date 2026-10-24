"""Feature engineering + the bridge from the AI Rep Counter's workout log."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_CHECKIN_RATE = 0.6  # typical reply rate, used when a client never got a check-in form


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features a coach would reason with. Pure function: returns a copy."""
    out = df.copy()
    sessions_per_month = out["sessions_planned_per_week"].clip(lower=1) * 4.33
    out["fee_per_session_inr"] = (out["monthly_fee_inr"] / sessions_per_month).round(1)
    out["is_new_client"] = (out["tenure_weeks"] <= 4).astype(int)
    out["uses_app"] = out["avg_form_score"].notna().astype(int)
    # One number for "is this client still engaged?": shows up, replies, keeps a streak.
    # Fixed fill value (not the batch median) so training and single-client serving
    # compute exactly the same feature - no training/serving skew.
    checkin = out["checkin_response_rate"].fillna(DEFAULT_CHECKIN_RATE)
    recency = np.exp(-out["days_since_last_workout"] / 7)  # 1.0 = trained today, ~0.37 after a week
    out["engagement_index"] = (0.5 * out["attendance_rate_4w"] + 0.3 * checkin + 0.2 * recency).round(3)
    return out


def aggregate_workout_log(log: pd.DataFrame, as_of: Optional[str] = None, window_days: int = 30) -> pd.DataFrame:
    """Turn the AI Rep Counter's ``workout_log.csv`` into per-client features.

    Returns one row per client with ``avg_form_score`` (last ``window_days``),
    ``days_since_last_workout`` and ``logged_sessions_30d``. Merge these onto the
    client table on ``client_id`` to replace manually tracked values.
    """
    log = log.copy()
    log["date"] = pd.to_datetime(log["date"])
    as_of_ts = pd.Timestamp(as_of) if as_of else log["date"].max()
    recent = log[(log["date"] > as_of_ts - pd.Timedelta(days=window_days)) & (log["date"] <= as_of_ts)]

    last = log[log["date"] <= as_of_ts].groupby("client_id")["date"].max()
    agg = pd.DataFrame({"days_since_last_workout": (as_of_ts - last).dt.days})
    agg["avg_form_score"] = recent.groupby("client_id")["form_score"].mean().round(1)
    agg["logged_sessions_30d"] = recent.groupby("client_id")["date"].nunique()
    agg["logged_sessions_30d"] = agg["logged_sessions_30d"].fillna(0).astype(int)
    return agg.reset_index()
