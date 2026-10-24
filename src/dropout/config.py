"""Project-wide constants: paths, feature lists, and the business assumptions behind the threshold."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "clients.csv"
MODEL_PATH = ROOT / "models" / "dropout_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "dropped_out"
ID_COL = "client_id"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age",
    "tenure_weeks",
    "monthly_fee_inr",
    "sessions_planned_per_week",
    "attendance_rate_4w",
    "attendance_trend",
    "days_since_last_workout",
    "checkin_response_rate",
    "avg_form_score",
    "weight_progress_pct",
    "paused_weeks_used",
    "support_msgs_30d",
    # engineered (see features.add_engineered_features)
    "fee_per_session_inr",
    "engagement_index",
]
CATEGORICAL_FEATURES = ["gender", "goal", "plan_type", "plan_months"]
BINARY_FEATURES = ["referred", "took_trial", "is_new_client", "uses_app"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# --- Business assumptions used to pick the decision threshold -----------------
# A flagged client gets a retention action (personal call + one free session).
# These are my estimates from coaching; change them and retrain to see the effect.
REVENUE_LOST_PER_DROPOUT_INR = 9_000  # ~2 more months of fees on average
RETENTION_ACTION_COST_INR = 500  # coach time + a free session
RETENTION_SUCCESS_RATE = 0.35  # share of at-risk clients a good call actually keeps
