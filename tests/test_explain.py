import numpy as np

from dropout.explain import ACTIONS, Explainer, _original_feature, suggested_actions
from dropout.features import add_engineered_features


def test_feature_name_mapping():
    assert _original_feature("num__attendance_rate_4w") == "attendance_rate_4w"
    assert _original_feature("cat__plan_type_in_person") == "plan_type"
    assert _original_feature("cat__plan_months_6") == "plan_months"
    assert _original_feature("cat__goal_fat_loss") == "goal"
    assert _original_feature("bin__is_new_client") == "is_new_client"


def test_shap_values_add_up_to_model_output(bundle, clients):
    """SHAP's core property: base value + contributions = the model's raw output (log-odds)."""
    X = add_engineered_features(clients.head(20))[bundle["features"]]
    explainer = Explainer(bundle["pipeline"], bundle["background"])
    contrib = explainer.shap_by_feature(X)
    assert list(contrib.columns) == list(dict.fromkeys(explainer.groups))
    pipe = bundle["pipeline"]
    if hasattr(pipe.named_steps["clf"], "decision_function"):
        raw_output = pipe.decision_function(X)  # log-odds
    else:
        raw_output = pipe.predict_proba(X)[:, 1]  # tree ensembles like random forest explain probability
    base = np.ravel(explainer._explainer.expected_value)[-1]
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), raw_output, atol=1e-3)


def test_top_reasons_are_positive_and_actionable(bundle, clients):
    X = add_engineered_features(clients.head(30))[bundle["features"]]
    reasons = Explainer(bundle["pipeline"], bundle["background"]).top_reasons(X, k=3)
    assert len(reasons) == 30
    for r in reasons:
        assert len(r) <= 3
        assert all(item["impact"] > 0 for item in r)


def test_suggested_actions_dedupes_and_has_fallback():
    reasons = [{"action": ACTIONS["tenure_weeks"]}, {"action": ACTIONS["is_new_client"]}, {"action": None}]
    assert suggested_actions(reasons) == [ACTIONS["tenure_weeks"]]
    assert suggested_actions([]) == ["Quick personal check-in this week."]
