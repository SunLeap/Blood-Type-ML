"""Streamlit dashboard. CMD: python -m streamlit run app.py

No API server, database, user login, or live hospital connection is required.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from src.feature_engineering import load_processed, generate_processed
from src.prediction import (
    CLASSIFIER_PATH, REGRESSOR_PATH, CLASS_METRICS_PATH, REG_METRICS_PATH,
    predict_next_week,
)

st.set_page_config(page_title="Blood Shortage ML", page_icon="🩸", layout="wide")
st.title("🩸 Blood Shortage Risk Prediction")
st.caption("University research prototype • Zhejiang, China • Not real-time clinical decision support")

@st.cache_data(show_spinner=False)
def get_data():
    return load_processed()

@st.cache_data(show_spinner=False)
def get_forecasts():
    return predict_next_week()

try:
    data = get_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Unable to read the dataset: {exc}")
    st.stop()

if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
    st.error("Models are missing. Open a terminal in the project directory and run:")
    st.code("python -m src.prediction --train-all", language="bash")
    st.stop()

try:
    prediction = get_forecasts()
except Exception as exc:
    st.error(f"Prediction failed: {exc}. Re-run the training command if dependencies changed.")
    st.stop()

st.info("Historical observations are separate from model forecasts. The prediction estimates the week after the final observation, not the current calendar week.")

st.sidebar.header("Dashboard controls")
chosen = st.sidebar.selectbox("Blood type", ["All", "A", "B", "O", "AB"])
window = st.sidebar.slider("Recent history (weeks)", min_value=12, max_value=104, value=52, step=4)
st.sidebar.markdown("**Shortage-risk thresholds**")
st.sidebar.caption("High: coverage < 80% | Medium: 80–<100% | Low: ≥100%")
st.sidebar.caption("Coverage = supplied units / demand units. Collection efficiency = supplied / collected.")

st.subheader("Next-observation forecast")
subset_predictions = prediction if chosen == "All" else prediction[prediction.blood_type == chosen]
st.caption(f"Last recorded week: {prediction.last_observed_week.max().date()} • Forecast week: {prediction.forecast_week.max().date()}")
cols = st.columns(len(subset_predictions))
for col, (_, row) in zip(cols, subset_predictions.iterrows()):
    with col:
        with st.container(border=True):
            st.markdown(f"#### Blood type {row.blood_type}")
            st.metric("Predicted risk", row.predicted_risk)
            st.metric("Forecast demand (units)", f"{row.predicted_demand:,.0f}")
            st.caption(f"Model confidence: {row.risk_confidence:.1%} • Current observed risk: {row.observed_risk}")
            if row.predicted_risk == "High":
                st.error("⚠️ Forecast HIGH risk: review stock and demand")
            elif row.predicted_risk == "Medium":
                st.warning("Forecast MEDIUM risk: monitor inventory")
            else:
                st.success("Forecast LOW risk")

st.subheader("Observed historical supply and demand")
chart_data = data if chosen == "All" else data[data.blood_type == chosen]
if chosen == "All":
    weekly = chart_data.groupby("date", as_index=False)[["demand", "supply", "collected"]].sum(min_count=1)
else:
    weekly = chart_data[["date", "demand", "supply", "collected"]].copy()
weekly = weekly.sort_values("date").tail(window)
plot_df = weekly.melt(id_vars="date", value_vars=["demand", "supply", "collected"], var_name="Indicator", value_name="Units")
st.plotly_chart(px.line(plot_df, x="date", y="Units", color="Indicator", title="Observed data (no predictions in this chart)"), use_container_width=True)

st.subheader("Observed indicators")
latest = data.sort_values("date").groupby("blood_type", as_index=False).tail(1)
if chosen != "All":
    latest = latest[latest.blood_type == chosen]
a, b, c = st.columns(3)
with a:
    st.metric("Last observed coverage", f"{latest.coverage.mean():.1%}")
with b:
    st.metric("Last observed collection efficiency", f"{latest.collection_efficiency.mean():.1%}")
with c:
    st.metric("Historical observations shown", f"{len(weekly):,}")

st.subheader("Historical shortage-risk distribution")
risk_counts = chart_data.groupby(["blood_type", "observed_risk"]).size().reset_index(name="Weeks")
st.plotly_chart(px.bar(risk_counts, x="blood_type", y="Weeks", color="observed_risk", barmode="group",
                        category_orders={"observed_risk": ["High", "Medium", "Low"]},
                        title="Counts are historical observations, not forecasts"), use_container_width=True)

st.subheader("Model evaluation on held-out chronological test data")
if CLASS_METRICS_PATH.exists() and REG_METRICS_PATH.exists():
    class_scores = pd.read_csv(CLASS_METRICS_PATH)
    reg_scores = pd.read_csv(REG_METRICS_PATH)
    left, right = st.columns(2)
    with left:
        st.markdown("**Classification: macro F1 and HIGH-risk recall**")
        st.plotly_chart(px.bar(class_scores, x="model", y=["test_macro_f1", "test_high_recall"],
                               barmode="group", labels={"value": "Score", "model": "Algorithm"}), use_container_width=True)
        with st.expander("Classification score table"):
            st.dataframe(class_scores.round(3), hide_index=True)
    with right:
        st.markdown("**Regression: MAE (lower is better)**")
        st.plotly_chart(px.bar(reg_scores, x="model", y="test_mae",
                               labels={"test_mae": "Mean absolute error (units)", "model": "Algorithm"}), use_container_width=True)
        with st.expander("Regression score table"):
            st.dataframe(reg_scores.round(3), hide_index=True)
else:
    st.info("Run training to generate the model comparison files.")

with st.expander("Methodology and limitations"):
    st.markdown("""
    - **Target:** next observed week, only if it is 5–9 days after the current date.
    - **Features:** latest known demand, supply, collection, coverage, weekly lags, and rolling averages.
    - **Splitting:** chronological 70% train, 15% validation, 15% test, based on target dates.
    - **Choice of models:** best validation macro-F1 (classification) and validation MAE (regression).
    - **Data cleaning:** same-date conflicting rows are averaged; this assumption should be checked with the data provider.
    - **Scope:** a historical retrospective demonstration, not a real-time blood-bank alert service. Confidence values are model probabilities, not calibrated medical probabilities.
    - **Alerts:** displayed only for the next predicted week; sustained consecutive-forecast alerts need future repeated forecasts and were intentionally excluded from this prototype.
    """)
