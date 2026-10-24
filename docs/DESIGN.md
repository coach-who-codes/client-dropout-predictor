# Design notes

## Problem framing

- **Prediction target:** did the client stop training / not renew within the next 30 days (binary).
- **Unit:** one row per client, a snapshot of the last 4 weeks.
- **Why it matters:** a dropout loses ~2 months of fees on average, and a quick personal call
  keeps a meaningful share of at-risk clients. The model's job is to say *who* to call and *what* to say.

## Data

Real client data is private, so `data.py` generates a synthetic book of clients with the patterns
I see in coaching: a hidden motivation level drives attendance and replies; month one is the riskiest;
attendance trend and gaps since the last workout are the strongest alarms; referred and in-person
clients stay longer; slow progress on an expensive plan pushes people out. Part of motivation is
deliberately left *unobservable*, so the model can't be perfect - like real life.

Swap in a real export with the same columns and everything else runs unchanged.

## Leakage and validation

- All preprocessing (imputation, scaling, one-hot) lives inside the scikit-learn `Pipeline`, so it is
  fitted on training folds only.
- 80/20 stratified split. Model selection **and** threshold selection use 5-fold out-of-fold
  predictions on the training set. The test set is scored once.
- Feature engineering is stateless (fixed fill values, no batch statistics), and a test checks that a
  single client gets identical features to the same client inside a batch - no training/serving skew.

## Model choice

Logistic regression, random forest and histogram gradient boosting are compared. The signal is mostly
additive, so the linear baseline is as good as the ensembles - and it's faster and easier to explain.
The selection rule is "most money saved in cross-validation" (ties -> PR-AUC), not raw accuracy.

## Imbalance and calibration

~25% of clients drop out. Instead of `class_weight="balanced"` (which distorts probabilities), the model
is trained on the natural distribution and the **decision threshold** is moved. Probabilities stay
calibrated, so the API's "30% risk" means roughly 30%.

## Threshold = business decision

```
value of catching a real dropout = success_rate x revenue_lost - action_cost = 0.35 x 9000 - 500 = Rs 2,650
cost of a false alarm            = action_cost                                    = Rs 500
```

The threshold that maximises expected savings on out-of-fold predictions is used in production.
All three numbers live in `config.py`; change them and retrain.

## Explanations

SHAP values (exact for linear and tree models) are summed back from one-hot columns to the original
feature, so "plan_type_online" and "plan_type_in_person" become one reason: *Online vs in-person*.
The top positive contributors become `top_reasons`, each mapped to a concrete coaching action.

## Serving

- FastAPI with Pydantic validation (bad ranges -> 422 with a clear message).
- `/predict` for one client, `/predict/batch` returns clients sorted by risk.
- Docker image trains the model at build time so the pickle always matches the installed scikit-learn.
- CI: lint, tests (which train a fresh model), a full training run, and a Docker build + health check.

## Limitations / next steps

- Synthetic data: the real test is my own client history.
- Snapshot model: a survival model (time-to-dropout) would also say *when*.
- Fixed threshold: with limited coach time, rank by expected savings and take the top-K instead.
- Monitoring: track feature drift and the realised dropout rate, retrain monthly.
