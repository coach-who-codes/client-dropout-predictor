"""End-to-end training: data -> model comparison -> threshold -> test evaluation -> artefacts.

python -m dropout.train            # uses data/clients.csv (generates it if missing)
python -m dropout.train --regenerate --n 6000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from . import plots
from .config import (
    DATA_PATH,
    FEATURES,
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from .data import generate_clients
from .explain import READABLE, Explainer
from .features import add_engineered_features
from .model import best_threshold, candidate_models, confusion, evaluate, expected_savings, oof_probabilities


def load_data(regenerate: bool = False, n: int = 4000) -> pd.DataFrame:
    if regenerate or not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_clients(n=n).to_csv(DATA_PATH, index=False)
        print(f"Generated {n} synthetic clients -> {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def train(df: pd.DataFrame, make_figures: bool = True, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    df = add_engineered_features(df)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    log(f"Train {len(X_train)} / test {len(X_test)} clients, dropout rate {y.mean():.1%}")

    # 1) Compare models with 5-fold cross-validation on the training set only.
    models = candidate_models()
    oof = {name: oof_probabilities(pipe, X_train, y_train) for name, pipe in models.items()}
    cv_table = {}
    for name, proba in oof.items():
        thr = best_threshold(y_train, proba)
        cv_table[name] = {"threshold": thr, **evaluate(y_train, proba, thr).as_dict()}
    log("\nCross-validation (train set):")
    log(pd.DataFrame(cv_table).T[["roc_auc", "pr_auc", "brier", "savings_per_100_clients_inr"]].round(3))

    # 2) Pick the model that saves the most money (ties broken by PR-AUC).
    best_name = max(cv_table, key=lambda m: (cv_table[m]["savings_per_100_clients_inr"], cv_table[m]["pr_auc"]))
    threshold = cv_table[best_name]["threshold"]
    log(f"\nSelected: {best_name}  (threshold {threshold:.2f}, chosen on out-of-fold predictions)")

    # 3) Refit on the full training set, evaluate ONCE on the untouched test set.
    pipeline = models[best_name].fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold).as_dict()
    tn, fp, fn, tp = confusion(y_test, test_proba, threshold).ravel()
    log("\nTest set:", json.dumps(test_metrics, indent=2))

    # 4) Explanations: which features drive risk on the test set.
    background = X_train.sample(min(100, len(X_train)), random_state=RANDOM_STATE)
    explainer = Explainer(pipeline, background)
    shap_test = explainer.shap_by_feature(X_test)
    importance = shap_test.abs().mean().sort_values(ascending=False)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_clients": int(len(df)),
        "dropout_rate": round(float(y.mean()), 4),
        "selected_model": best_name,
        "threshold": threshold,
        "cross_validation": cv_table,
        "test": {**test_metrics, "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}},
        "top_features": {k: round(float(v), 4) for k, v in importance.head(10).items()},
    }
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "features": FEATURES,
        "background": background,
        "model_name": best_name,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }

    if make_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plots.model_comparison(y_train, oof, FIGURES_DIR / "model_comparison.png")
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        per100 = np.array([expected_savings(y_train, oof[best_name], t) / len(y_train) * 100 for t in grid])
        plots.savings_curve(grid, per100, threshold, FIGURES_DIR / "savings_vs_threshold.png")
        plots.feature_importance(importance, READABLE, FIGURES_DIR / "feature_importance.png")
        plots.rate_by_bucket(
            df,
            "attendance_rate_4w",
            [0, 0.4, 0.55, 0.7, 0.85, 1.0],
            ["<40%", "40-55%", "55-70%", "70-85%", "85%+"],
            TARGET,
            "Dropout rate by attendance (last 4 weeks)",
            "Share of planned sessions attended",
            FIGURES_DIR / "dropout_by_attendance.png",
        )
    return bundle


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the client dropout model")
    parser.add_argument("--regenerate", action="store_true", help="re-create the synthetic dataset")
    parser.add_argument("--n", type=int, default=4000, help="number of synthetic clients")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    bundle = train(load_data(args.regenerate, args.n), make_figures=not args.no_figures)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=2))
    print(f"\nSaved model -> {MODEL_PATH}\nSaved metrics -> {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
