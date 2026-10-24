"""Coach dashboard: upload your client list, see who to call this week and why.

streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout.config import DATA_PATH, MODEL_PATH, TARGET
from dropout.explain import Explainer, suggested_actions
from dropout.features import add_engineered_features

st.set_page_config(page_title="Client Dropout Radar", page_icon=":bar_chart:", layout="wide")


@st.cache_resource
def load():
    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m dropout.train` first.")
        st.stop()
    bundle = joblib.load(MODEL_PATH)
    return bundle, Explainer(bundle["pipeline"], bundle["background"])


bundle, explainer = load()
threshold = bundle["threshold"]

st.title("Client Dropout Radar")
st.caption(
    f"Model: {bundle['model_name'].replace('_', ' ')} - test ROC-AUC {bundle['metrics']['test']['roc_auc']:.2f}. "
    f"Clients with risk >= {threshold:.0%} are flagged for a call this week."
)

uploaded = st.file_uploader("Upload your clients CSV (same columns as data/clients.csv)", type="csv")
if uploaded:
    clients = pd.read_csv(uploaded)
else:
    st.info("Showing a sample of the synthetic dataset. Upload your own CSV to score real clients.")
    clients = pd.read_csv(DATA_PATH).sample(60, random_state=1)

X = add_engineered_features(clients.drop(columns=[TARGET], errors="ignore"))[bundle["features"]]
clients = clients.assign(risk=bundle["pipeline"].predict_proba(X)[:, 1])
reasons = explainer.top_reasons(X, k=3)
clients["why"] = [", ".join(r["label"] for r in rs) or "-" for rs in reasons]
clients["next_step"] = [suggested_actions(rs, 1)[0] for rs in reasons]
clients = clients.sort_values("risk", ascending=False)
flagged = clients[clients["risk"] >= threshold]

c1, c2, c3 = st.columns(3)
c1.metric("Clients scored", len(clients))
c2.metric("Call this week", len(flagged))
c3.metric("Average risk", f"{clients['risk'].mean():.0%}")

st.subheader("Who to call this week")
view = flagged.assign(risk=flagged["risk"] * 100, attendance_rate_4w=flagged["attendance_rate_4w"] * 100)
columns = ["client_id", "risk", "plan_type", "plan_months", "attendance_rate_4w", "days_since_last_workout"]
st.dataframe(
    view[columns + ["why", "next_step"]],
    hide_index=True,
    width="stretch",
    column_config={
        "risk": st.column_config.ProgressColumn("Risk", format="%.0f%%", min_value=0, max_value=100),
        "attendance_rate_4w": st.column_config.NumberColumn("Attendance (4w)", format="%.0f%%"),
        "days_since_last_workout": st.column_config.NumberColumn("Days since workout"),
        "why": "Top reasons",
        "next_step": "Suggested next step",
    },
)

with st.expander("All clients"):
    st.dataframe(clients, hide_index=True, width="stretch")
