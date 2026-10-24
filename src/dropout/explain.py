"""Per-client explanations (SHAP) and the coaching action each reason maps to."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

ORIGINAL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

READABLE = {
    "attendance_rate_4w": "Attendance (last 4 weeks)",
    "attendance_trend": "Attendance trend",
    "days_since_last_workout": "Days since last workout",
    "checkin_response_rate": "Replies to weekly check-ins",
    "weight_progress_pct": "Progress towards goal",
    "avg_form_score": "Form score (AI Rep Counter)",
    "uses_app": "App usage",
    "tenure_weeks": "Weeks as a client",
    "is_new_client": "In first month",
    "plan_months": "Plan length",
    "plan_type": "Online vs in-person",
    "monthly_fee_inr": "Monthly fee",
    "fee_per_session_inr": "Fee per session",
    "referred": "Referral",
    "engagement_index": "Overall engagement",
    "paused_weeks_used": "Pause weeks used",
    "support_msgs_30d": "Messages to coach (30 days)",
    "sessions_planned_per_week": "Sessions planned per week",
    "took_trial": "Took the free trial",
    "goal": "Goal",
    "gender": "Gender",
    "age": "Age",
}

ACTIONS = {
    "attendance_rate_4w": "Lock in 2-3 fixed session slots for the next two weeks.",
    "attendance_trend": "Attendance is slipping - check in personally this week.",
    "days_since_last_workout": "Call today (not a text) and book the next session on the call.",
    "checkin_response_rate": "Switch check-ins to a 2-minute call or voice note.",
    "engagement_index": "Re-engage: short call, a fresh weekly target, celebrate a small win.",
    "weight_progress_pct": "Review the plan together and set a small 2-week goal they can hit.",
    "avg_form_score": "Book a form-check session to rebuild confidence.",
    "uses_app": "Help them set up the app so progress becomes visible.",
    "tenure_weeks": "Do a 'first month' review call - month one is when most people quit.",
    "is_new_client": "Do a 'first month' review call - month one is when most people quit.",
    "plan_months": "Offer a 3-month plan with the pause benefit.",
    "monthly_fee_inr": "Show results so far to justify the investment.",
    "fee_per_session_inr": "Show results so far to justify the investment.",
    "paused_weeks_used": "Plan the comeback week together after the pause.",
    "support_msgs_30d": "Reach out first - they've gone quiet.",
}


def _original_feature(transformed_name: str) -> str:
    """'num__attendance_rate_4w' -> 'attendance_rate_4w', 'cat__plan_type_online' -> 'plan_type'."""
    name = transformed_name.split("__", 1)[-1]
    matches = [f for f in ORIGINAL_FEATURES if name == f or name.startswith(f + "_")]
    return max(matches, key=len) if matches else name


class Explainer:
    """Wraps SHAP for a fitted pipeline and aggregates one-hot columns back to original features."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame) -> None:
        import shap  # imported lazily - only needed when explaining

        self.prep = pipeline.named_steps["prep"]
        self.clf = pipeline.named_steps["clf"]
        self.names = list(self.prep.get_feature_names_out())
        self.groups = [_original_feature(n) for n in self.names]
        bg = self.prep.transform(background)
        if hasattr(self.clf, "coef_"):
            self._explainer = shap.LinearExplainer(self.clf, bg)
        else:
            self._explainer = shap.TreeExplainer(self.clf, bg, model_output="raw")

    def shap_by_feature(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds units) per original feature. Positive = pushes risk up."""
        values = self._explainer.shap_values(self.prep.transform(X))
        values = np.asarray(values)
        if values.ndim == 3:  # (rows, features, classes) for some tree models
            values = values[:, :, 1]
        frame = pd.DataFrame(values, columns=self.names, index=X.index)
        return frame.T.groupby(self.groups, sort=False).sum().T

    def top_reasons(self, X: pd.DataFrame, k: int = 3) -> List[List[Dict]]:
        contrib = self.shap_by_feature(X)
        results = []
        for idx, row in contrib.iterrows():
            ranked = row[row > 0].sort_values(ascending=False).head(k)
            reasons = []
            for feature, impact in ranked.items():
                value = X.loc[idx, feature] if feature in X.columns else None
                if isinstance(value, (np.generic,)):
                    value = value.item()
                if isinstance(value, float) and np.isnan(value):
                    value = None
                reasons.append(
                    {
                        "feature": feature,
                        "label": READABLE.get(feature, feature),
                        "value": value,
                        "impact": round(float(impact), 3),
                        "action": ACTIONS.get(feature),
                    }
                )
            results.append(reasons)
        return results


def suggested_actions(reasons: List[Dict], max_actions: int = 2) -> List[str]:
    actions: List[str] = []
    for reason in reasons:
        action = reason.get("action")
        if action and action not in actions:
            actions.append(action)
    return actions[:max_actions] or ["Quick personal check-in this week."]
