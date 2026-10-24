import numpy as np

from dropout.model import best_threshold, evaluate, expected_savings


def test_model_beats_baseline_on_test_set(bundle):
    test = bundle["metrics"]["test"]
    assert test["roc_auc"] > 0.75
    assert test["pr_auc"] > bundle["metrics"]["dropout_rate"] + 0.15  # far above the random baseline
    assert test["savings_inr"] > 0


def test_model_comparison_was_run(bundle):
    cv = bundle["metrics"]["cross_validation"]
    assert set(cv) == {"logistic_regression", "random_forest", "gradient_boosting"}
    assert bundle["model_name"] in cv


def test_threshold_is_sane(bundle):
    assert 0.05 <= bundle["threshold"] <= 0.95


def test_pipeline_handles_missing_values(bundle, clients):
    from dropout.features import add_engineered_features

    X = add_engineered_features(clients.head(5))
    X.loc[:, ["checkin_response_rate", "avg_form_score"]] = np.nan
    proba = bundle["pipeline"].predict_proba(add_engineered_features(X)[bundle["features"]])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_expected_savings_math():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    # threshold 0.5 flags rows 0 (TP) and 2 (FP): 0.35*9000 - 500 = 2650 ; minus 500 for the FP
    assert expected_savings(y, p, 0.5) == 2650 - 500
    assert expected_savings(y, p, 0.99) == 0  # flag nobody -> no cost, no savings


def test_best_threshold_maximises_savings():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.5 + rng.random(500) * 0.5, 0, 1)
    t = best_threshold(y, p)
    assert expected_savings(y, p, t) >= max(expected_savings(y, p, x) for x in (0.1, 0.3, 0.5, 0.7, 0.9))


def test_evaluate_returns_all_metrics():
    m = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6]), 0.5).as_dict()
    assert m["roc_auc"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
