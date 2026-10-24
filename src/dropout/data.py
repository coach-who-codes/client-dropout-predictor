"""Synthetic personal-training client dataset.

Real client data is private, so this generator simulates a realistic book of
clients. The *structure* comes from what I see as a coach:

* a hidden "motivation" level drives attendance, check-in replies and app use;
* early weeks are the riskiest (most quitting happens in month one);
* falling attendance and long gaps since the last workout are the loudest alarms;
* referred clients and in-person clients stick around longer;
* slow results on an expensive plan pushes people out.

The target ``dropped_out`` = the client stopped training / didn't renew within
the next 30 days. Swap in a real export with the same columns and every other
part of the project works unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

FEES_INR = {  # monthly-equivalent fee by (plan_type, plan_months)
    ("in_person", 1): 8000,
    ("in_person", 3): 6667,
    ("in_person", 6): 6000,
    ("online", 1): 4000,
    ("online", 3): 3333,
    ("online", 6): 3000,
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_clients(n: int = 4000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    motivation = rng.normal(0, 1, n)  # latent - never shown to the model

    age = np.clip(rng.normal(31, 8, n), 18, 60).round().astype(int)
    gender = rng.choice(["M", "F"], n, p=[0.6, 0.4])
    goal = rng.choice(["fat_loss", "muscle_gain", "general_fitness", "strength"], n, p=[0.45, 0.3, 0.15, 0.1])
    plan_type = rng.choice(["online", "in_person"], n, p=[0.55, 0.45])
    plan_months = rng.choice([1, 3, 6], n, p=[0.4, 0.4, 0.2])
    monthly_fee = np.array([FEES_INR[(t, m)] for t, m in zip(plan_type, plan_months)])
    tenure_weeks = np.clip(rng.exponential(14, n), 1, 52).round().astype(int)
    referred = rng.random(n) < 0.3
    took_trial = rng.random(n) < 0.5
    sessions_planned = rng.choice([3, 4, 5, 6], n, p=[0.3, 0.35, 0.25, 0.1])

    in_person = plan_type == "in_person"
    attendance = _sigmoid(0.9 + 0.9 * motivation + 0.3 * in_person + 0.25 * referred + rng.normal(0, 0.6, n))
    attendance_trend = np.clip(0.12 * motivation + rng.normal(-0.02, 0.12, n), -0.6, 0.4)
    days_since_last = np.clip(rng.exponential(1.5 + 9 * (1 - attendance) ** 2, n), 0, 45).round().astype(int)
    checkin = np.clip(_sigmoid(0.5 + 0.9 * motivation + rng.normal(0, 0.7, n)), 0, 1)
    uses_app = rng.random(n) < _sigmoid(0.9 + 0.5 * motivation)
    form_score = np.clip(rng.normal(68 + 6 * motivation + 4 * in_person, 10, n), 20, 100)
    progress = np.clip(
        rng.normal(15 + 45 * attendance + 0.8 * np.minimum(tenure_weeks, 24) + 5 * motivation, 15, n), -20, 120
    )
    paused = np.minimum(rng.poisson(0.25 + 0.03 * tenure_weeks, n), np.maximum(1, tenure_weeks // 4))
    support_msgs = rng.poisson(np.clip(3 + 2 * motivation, 0.2, None), n)

    # --- dropout probability (the "truth" the model has to learn) -------------
    logit = (
        -1.8
        - 3.0 * (attendance - 0.7)
        - 2.2 * attendance_trend
        + 0.07 * np.minimum(days_since_last, 21)
        - 1.0 * (checkin - 0.6)
        - 0.015 * (progress - 40)
        - 0.6 * referred
        - 0.35 * in_person
        + 0.5 * (plan_months == 1)
        - 0.3 * (plan_months == 6)
        + 0.7 * (tenure_weeks <= 4)
        - 0.015 * np.where(uses_app, form_score - 68, 0)
        + 0.3 * ~uses_app
        + 0.5 * ((monthly_fee >= 6000) & (progress < 30))
        + 0.25 * (paused >= 2)
        - 0.4 * motivation  # part of motivation the features can't fully see
        + rng.normal(0, 0.6, n)
    )
    dropped_out = rng.random(n) < _sigmoid(logit)

    df = pd.DataFrame(
        {
            "client_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "age": age,
            "gender": gender,
            "goal": goal,
            "plan_type": plan_type,
            "plan_months": plan_months,
            "monthly_fee_inr": monthly_fee,
            "tenure_weeks": tenure_weeks,
            "referred": referred.astype(int),
            "took_trial": took_trial.astype(int),
            "sessions_planned_per_week": sessions_planned,
            "attendance_rate_4w": attendance.round(3),
            "attendance_trend": attendance_trend.round(3),
            "days_since_last_workout": days_since_last,
            "checkin_response_rate": checkin.round(3),
            "avg_form_score": np.where(uses_app, form_score.round(1), np.nan),
            "weight_progress_pct": progress.round(1),
            "paused_weeks_used": paused,
            "support_msgs_30d": support_msgs,
            "dropped_out": dropped_out.astype(int),
        }
    )
    # Real exports have gaps: some clients never answered a check-in form.
    missing = rng.random(n) < 0.07
    df.loc[missing, "checkin_response_rate"] = np.nan
    return df


if __name__ == "__main__":
    data = generate_clients()
    print(data.head())
    print(f"\n{len(data)} clients, dropout rate {data['dropped_out'].mean():.1%}")
