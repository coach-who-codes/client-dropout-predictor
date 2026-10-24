"""REST API: score a client's dropout risk and get the reasons + what to do.

uvicorn api.main:app --reload
open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dropout import __version__  # noqa: E402
from dropout.config import MODEL_PATH  # noqa: E402
from dropout.explain import Explainer, suggested_actions  # noqa: E402
from dropout.features import add_engineered_features  # noqa: E402


class Client(BaseModel):
    """One client's current snapshot. Ranges are validated - bad input gets a clear 422."""

    client_id: str = Field("anonymous", examples=["C00042"])
    age: int = Field(..., ge=14, le=90, examples=[29])
    gender: Literal["M", "F"] = Field(..., examples=["M"])
    goal: Literal["fat_loss", "muscle_gain", "general_fitness", "strength"] = Field(..., examples=["fat_loss"])
    plan_type: Literal["online", "in_person"] = Field(..., examples=["online"])
    plan_months: Literal[1, 3, 6] = Field(..., examples=[1])
    monthly_fee_inr: float = Field(..., gt=0, examples=[4000])
    tenure_weeks: int = Field(..., ge=0, le=520, examples=[3])
    referred: bool = Field(False, examples=[False])
    took_trial: bool = Field(False, examples=[True])
    sessions_planned_per_week: int = Field(..., ge=1, le=7, examples=[4])
    attendance_rate_4w: float = Field(..., ge=0, le=1, examples=[0.45])
    attendance_trend: float = Field(0.0, ge=-1, le=1, examples=[-0.2])
    days_since_last_workout: int = Field(..., ge=0, le=365, examples=[9])
    checkin_response_rate: Optional[float] = Field(None, ge=0, le=1, examples=[0.3])
    avg_form_score: Optional[float] = Field(
        None, ge=0, le=100, description="From the AI Rep Counter; null if the client doesn't use it", examples=[61]
    )
    weight_progress_pct: float = Field(..., ge=-100, le=300, examples=[12])
    paused_weeks_used: int = Field(0, ge=0, le=52, examples=[0])
    support_msgs_30d: int = Field(0, ge=0, le=500, examples=[1])


class Reason(BaseModel):
    feature: str
    label: str
    value: Optional[Union[float, int, str]] = None
    impact: float
    action: Optional[str] = None


class Prediction(BaseModel):
    client_id: str
    dropout_probability: float
    risk_level: Literal["low", "medium", "high"]
    contact_this_week: bool
    top_reasons: List[Reason]
    suggested_actions: List[str]


class Batch(BaseModel):
    clients: List[Client] = Field(..., min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def load_bundle():
    path = os.environ.get("MODEL_PATH", str(MODEL_PATH))
    if not Path(path).exists():
        raise RuntimeError(f"Model not found at {path}. Train it first:  python -m dropout.train")
    bundle = joblib.load(path)
    bundle["explainer"] = Explainer(bundle["pipeline"], bundle["background"])
    return bundle


def risk_level(p: float, threshold: float) -> str:
    if p >= max(0.6, threshold):
        return "high"
    return "medium" if p >= threshold else "low"


def score(clients: List[Client]) -> List[Prediction]:
    bundle = load_bundle()
    raw = pd.DataFrame([c.model_dump() for c in clients])
    for col in ("referred", "took_trial"):
        raw[col] = raw[col].astype(int)
    raw[["checkin_response_rate", "avg_form_score"]] = raw[["checkin_response_rate", "avg_form_score"]].astype(float)
    X = add_engineered_features(raw)[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    reasons = bundle["explainer"].top_reasons(X, k=3)
    thr = bundle["threshold"]
    return [
        Prediction(
            client_id=c.client_id,
            dropout_probability=round(float(p), 4),
            risk_level=risk_level(float(p), thr),
            contact_this_week=bool(p >= thr),
            top_reasons=r,
            suggested_actions=suggested_actions(r),
        )
        for c, p, r in zip(clients, proba, reasons)
    ]


app = FastAPI(
    title="Client Dropout Predictor",
    version=__version__,
    description="Predicts which personal-training clients are likely to quit in the next 30 days, "
    "explains why (SHAP) and suggests a retention action.",
)


@app.get("/health")
def health():
    try:
        bundle = load_bundle()
    except RuntimeError as exc:
        return {"status": "no_model", "detail": str(exc)}
    return {
        "status": "ok",
        "model": bundle["model_name"],
        "threshold": bundle["threshold"],
        "test_roc_auc": bundle["metrics"]["test"]["roc_auc"],
    }


@app.post("/predict", response_model=Prediction)
def predict(client: Client):
    try:
        return score([client])[0]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(batch: Batch):
    """Score many clients; results are sorted by risk, highest first."""
    try:
        results = score(batch.clients)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return sorted(results, key=lambda r: r.dropout_probability, reverse=True)
