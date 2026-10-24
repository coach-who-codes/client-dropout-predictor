import pytest
from fastapi.testclient import TestClient

AT_RISK = {
    "client_id": "C00042",
    "age": 29,
    "gender": "M",
    "goal": "fat_loss",
    "plan_type": "online",
    "plan_months": 1,
    "monthly_fee_inr": 4000,
    "tenure_weeks": 3,
    "sessions_planned_per_week": 4,
    "attendance_rate_4w": 0.3,
    "attendance_trend": -0.3,
    "days_since_last_workout": 14,
    "checkin_response_rate": 0.2,
    "avg_form_score": None,
    "weight_progress_pct": 5,
}
COMMITTED = {
    **AT_RISK,
    "client_id": "C00007",
    "plan_type": "in_person",
    "plan_months": 6,
    "monthly_fee_inr": 6000,
    "tenure_weeks": 30,
    "referred": True,
    "attendance_rate_4w": 0.95,
    "attendance_trend": 0.1,
    "days_since_last_workout": 1,
    "checkin_response_rate": 0.95,
    "avg_form_score": 85,
    "weight_progress_pct": 80,
}


@pytest.fixture(scope="module")
def client(model_file, monkeypatch_module):
    monkeypatch_module.setenv("MODEL_PATH", str(model_file))
    from api import main

    main.load_bundle.cache_clear()
    with TestClient(main.app) as c:
        yield c
    main.load_bundle.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and 0 < body["threshold"] < 1


def test_at_risk_client_is_flagged_with_reasons(client):
    body = client.post("/predict", json=AT_RISK).json()
    assert body["dropout_probability"] > 0.5
    assert body["contact_this_week"] is True
    assert body["risk_level"] in {"medium", "high"}
    assert 1 <= len(body["top_reasons"]) <= 3
    assert all(r["impact"] > 0 for r in body["top_reasons"])
    assert body["suggested_actions"]


def test_committed_client_is_low_risk(client):
    body = client.post("/predict", json=COMMITTED).json()
    assert body["dropout_probability"] < 0.2
    assert body["risk_level"] == "low"


def test_batch_is_sorted_by_risk(client):
    body = client.post("/predict/batch", json={"clients": [COMMITTED, AT_RISK]}).json()
    assert [r["client_id"] for r in body] == ["C00042", "C00007"]


@pytest.mark.parametrize(
    "field, value",
    [("attendance_rate_4w", 1.5), ("plan_months", 2), ("gender", "X"), ("age", 5)],
)
def test_invalid_input_is_rejected(client, field, value):
    assert client.post("/predict", json={**AT_RISK, field: value}).status_code == 422


def test_missing_model_gives_clear_error(monkeypatch, tmp_path):
    from api import main

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    main.load_bundle.cache_clear()
    try:
        response = TestClient(main.app).post("/predict", json=AT_RISK)
        assert response.status_code == 503 and "Train it first" in response.json()["detail"]
    finally:
        main.load_bundle.cache_clear()
