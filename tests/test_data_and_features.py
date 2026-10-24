import numpy as np
import pandas as pd

from dropout.config import FEATURES, TARGET
from dropout.data import generate_clients
from dropout.features import add_engineered_features, aggregate_workout_log


def test_schema_and_ranges(clients):
    assert clients["client_id"].is_unique
    assert set(clients[TARGET].unique()) == {0, 1}
    assert clients["attendance_rate_4w"].between(0, 1).all()
    assert clients["avg_form_score"].dropna().between(0, 100).all()
    assert set(clients["plan_months"].unique()) <= {1, 3, 6}


def test_base_rate_is_realistic(clients):
    assert 0.15 < clients[TARGET].mean() < 0.45


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_clients(200, seed=1), generate_clients(200, seed=1))


def test_has_realistic_missing_values(clients):
    assert clients["avg_form_score"].isna().mean() > 0.05  # not every client uses the app
    assert clients["checkin_response_rate"].isna().any()


def test_low_attendance_drops_out_more(clients):
    low = clients[clients["attendance_rate_4w"] < 0.5][TARGET].mean()
    high = clients[clients["attendance_rate_4w"] > 0.8][TARGET].mean()
    assert low > high + 0.15


def test_engineered_features(clients):
    df = add_engineered_features(clients)
    assert set(FEATURES) <= set(df.columns)
    assert df["engagement_index"].between(0, 1).all()
    assert (df["uses_app"] == clients["avg_form_score"].notna()).all()
    assert "engagement_index" not in clients.columns  # pure function, input untouched


def test_engineered_features_same_for_one_row_or_many(clients):
    """No training/serving skew: a single client gets the same features as in a batch."""
    batch = add_engineered_features(clients.head(50))
    single = pd.concat([add_engineered_features(clients.iloc[[i]]) for i in range(50)])
    pd.testing.assert_frame_equal(batch, single)


def test_aggregate_workout_log_from_rep_counter():
    log = pd.DataFrame(
        {
            "date": ["2026-10-01", "2026-10-10", "2026-10-20", "2026-08-01", "2026-10-15"],
            "client_id": ["C1", "C1", "C1", "C2", "C3"],
            "exercise": ["squat"] * 5,
            "form_score": [70, 80, 90, 50, np.nan],
        }
    )
    agg = aggregate_workout_log(log, as_of="2026-10-24").set_index("client_id")
    assert agg.loc["C1", "avg_form_score"] == 80.0
    assert agg.loc["C1", "days_since_last_workout"] == 4
    assert agg.loc["C1", "logged_sessions_30d"] == 3
    assert agg.loc["C2", "logged_sessions_30d"] == 0  # only an old session
    assert np.isnan(agg.loc["C2", "avg_form_score"])
