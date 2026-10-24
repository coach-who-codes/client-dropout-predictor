"""Model building, evaluation and cost-based threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RETENTION_ACTION_COST_INR,
    RETENTION_SUCCESS_RATE,
    REVENUE_LOST_PER_DROPOUT_INR,
)


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numbers, one-hot encode categories.

    plan_months (1 / 3 / 6) is a number but behaves like a category - a 6-month
    plan isn't "twice" a 3-month plan - so it's one-hot encoded.
    Living inside the Pipeline means imputation medians and scaling are learned
    on training folds only (no data leakage) and are reused as-is at serving time.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        verbose_feature_names_out=True,
    )


def candidate_models() -> Dict[str, Pipeline]:
    """Three models of increasing flexibility. Baseline first - always.

    No class_weight="balanced": it inflates probabilities. The 25% minority class is
    handled by moving the decision threshold instead, so a "0.30" from the API
    really means ~30% risk.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE
        ),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("clf", clf)]) for name, clf in models.items()}


def oof_probabilities(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> np.ndarray:
    """Out-of-fold predictions: every row is scored by a model that never saw it."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(clone(pipeline), X, y, cv=cv, method="predict_proba")[:, 1]


# ----------------------------------------------------------------- business cost
def expected_savings(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    """Net rupees saved by contacting every client with risk >= threshold.

    * true positive:  we keep RETENTION_SUCCESS_RATE of them -> revenue saved, minus action cost
    * false positive: we spend the action cost on someone who wasn't leaving
    * false/true negatives: no action, no cost (vs. doing nothing)
    """
    flagged = proba >= threshold
    tp = np.sum(flagged & (y_true == 1))
    fp = np.sum(flagged & (y_true == 0))
    value_per_tp = RETENTION_SUCCESS_RATE * REVENUE_LOST_PER_DROPOUT_INR - RETENTION_ACTION_COST_INR
    return float(tp * value_per_tp - fp * RETENTION_ACTION_COST_INR)


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    savings = [expected_savings(y_true, proba, t) for t in grid]
    return float(grid[int(np.argmax(savings))])


# -------------------------------------------------------------------- metrics
@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    brier: float
    precision: float
    recall: float
    flagged_share: float
    savings_inr: float
    savings_per_100_clients_inr: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Metrics:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    savings = expected_savings(y_true, proba, threshold)
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        pr_auc=float(average_precision_score(y_true, proba)),
        brier=float(brier_score_loss(y_true, proba)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        flagged_share=float(pred.mean()),
        savings_inr=savings,
        savings_per_100_clients_inr=savings / len(y_true) * 100,
    )


def confusion(y_true, proba, threshold) -> np.ndarray:
    return confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
