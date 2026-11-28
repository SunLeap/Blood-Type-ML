from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    BLOOD_TYPES, DEFAULT_PATHS, HIGH_RISK_COVERAGE, LOW_RISK_COVERAGE, RUNS_DIR, ProjectPaths, safe_name,
)
from src.data_pipeline import load_processed, prepare_dataset
from src.prediction import evaluate_saved_models, predict_next_week

st.set_page_config(page_title="Blood Shortage ML", page_icon="🩸", layout="wide")

RISK_STYLE = {
    "High": ("#c0392b", "#fdecea", "⚠️ HIGH RISK - WARNING", "Forecast coverage below 80%: review stock and demand."),
    "Medium": ("#b7791f", "#fff6e0", "● MEDIUM RISK - NOTICE", "Forecast coverage 80-100%: monitor inventory."),
    "Low": ("#2e7d32", "#e8f5e9", "✓ LOW RISK", "Forecast coverage at or above 100%."),
}
PRIMARY = "Zhejiang (primary research dataset)"
UPLOAD = "Upload a compatible CSV"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_primary(_paths: ProjectPaths, key: str) -> pd.DataFrame:
    return load_processed(paths=_paths)


@st.cache_data(show_spinner=False)
def prepare_upload(content: bytes):
    prepared = prepare_dataset(io.BytesIO(content))
    return prepared.report, prepared.processed


@st.cache_data(show_spinner=False)
def forecast_for(processed: pd.DataFrame, _model_paths: ProjectPaths, key: str) -> pd.DataFrame:
    return predict_next_week(processed, _model_paths)


@st.cache_data(show_spinner=False)
def external_evaluation(processed: pd.DataFrame, _model_paths: ProjectPaths, key: str) -> dict:
    return evaluate_saved_models(processed, _model_paths)


def saved_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if (p / "models" / "best_classifier.joblib").exists())


def models_ready(paths: ProjectPaths) -> bool:
    return paths.classifier.exists() and paths.regressor.exists()


# ---------------------------------------------------------------------------
# Sidebar: dataset and model selection
# ---------------------------------------------------------------------------
st.title("🩸 Blood Shortage Risk Prediction")
st.caption("ITM-390 university research prototype • Zhejiang, China • Not real-time clinical decision support")

st.sidebar.header("1. Dataset")
runs = saved_runs()
source_options = [PRIMARY, UPLOAD] + [f"Saved experiment: {r}" for r in runs]
source = st.sidebar.radio("Data source", source_options, index=0)

data_paths = DEFAULT_PATHS      # where the displayed data/results come from
model_paths = DEFAULT_PATHS     # which trained models make the forecast
processed, report, dataset_label = None, None, source

if source == PRIMARY:
    if not DEFAULT_PATHS.raw_data.exists():
        st.error(f"Primary dataset not found: {DEFAULT_PATHS.raw_data}")
        st.stop()
    try:
        processed = load_primary(DEFAULT_PATHS, f"primary-{_mtime(DEFAULT_PATHS.raw_data)}-{_mtime(DEFAULT_PATHS.processed_data)}")
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Unable to read the dataset: {exc}")
        st.stop()
elif source == UPLOAD:
    upload = st.sidebar.file_uploader("CSV with the Zhejiang column layout", type=["csv"])
    st.sidebar.caption("Required: date, collected_/demand_/supply_ for A, B, O, AB. Other columns are optional.")
    if upload is None:
        st.info("Upload a CSV in the sidebar. Required columns: `date`, and `collected_X`, `demand_X`, `supply_X` "
                "for X in A, B, O, AB. Totals, platelets, team and individual donations are optional.")
        st.stop()
    content = upload.getvalue()
    upload_key = hashlib.sha256(content).hexdigest()[:12]
    report, processed = prepare_upload(content)
    dataset_label = f"Uploaded file: {upload.name}"
    with st.expander("Data validation report", expanded=not report.ok):
        st.dataframe(report.as_frame(), hide_index=True)
        for msg in report.errors:
            st.error(msg)
        for msg in report.warnings:
            st.warning(msg)
    if not report.ok:
        st.error("This file is not compatible. Fix the errors above and upload again.")
        st.stop()

    st.sidebar.header("2. What to do with the upload")
    mode = st.sidebar.radio("Mode", ["Predict with the existing trained models",
                                     "Train NEW models on this file (separate experiment)"])
    if mode.startswith("Train"):
        run_name = st.sidebar.text_input("Experiment name", value=safe_name(Path(upload.name).stem) or "upload")
        st.sidebar.caption("Saved to runs/<name>/. The original dataset, results and models are not touched. "
                           "Uses the quick search (fewer candidates).")
        if not report.can_train:
            st.sidebar.error("Not enough history to train on this file.")
        elif st.sidebar.button("Train models now", type="primary"):
            from run_pipeline import run
            with tempfile.TemporaryDirectory() as tmp:
                tmp_csv = Path(tmp) / "upload.csv"
                tmp_csv.write_bytes(content)
                with st.spinner("Training (time-series CV tuning). This can take a few minutes..."):
                    try:
                        run(tmp_csv, run_name, quick=True, figures=True, log=lambda *_: None)
                    except Exception as exc:
                        st.sidebar.error(f"Training failed: {exc}")
                        st.stop()
            st.sidebar.success(f"Done. Select 'Saved experiment: {safe_name(run_name)}' above to view it.")
            st.cache_data.clear()
        st.info("Choose an experiment name and press **Train models now**. Until then, the forecast below uses "
                "the existing Zhejiang-trained models.")
    else:
        st.sidebar.caption("No retraining: the Zhejiang-trained models are applied to this file.")
else:
    run_name = source.split(": ", 1)[1]
    data_paths = model_paths = ProjectPaths.for_run(run_name)
    processed = load_primary(data_paths, f"{run_name}-{_mtime(data_paths.processed_data)}")
    dataset_label = f"Saved experiment: {run_name} (models trained on that dataset)"

if not models_ready(model_paths):
    st.error("Models are missing. Open a terminal in the project directory and run:")
    st.code("python run_pipeline.py", language="bash")
    st.stop()

st.sidebar.header("3. Display")
chosen = st.sidebar.selectbox("Blood type", ["All", *BLOOD_TYPES])
window = st.sidebar.slider("Recent history (weeks)", min_value=12, max_value=156, value=52, step=4)
alert_mode = st.sidebar.radio("Alert rule", ["Standard (most probable class)", "High-recall alert mode"],
                              help="High-recall mode raises a HIGH alert whenever P(High) reaches a threshold "
                                   "chosen by cross-validation. It catches more shortages but gives more false alarms.")
st.sidebar.markdown("**Shortage-risk thresholds**")
st.sidebar.caption(f"High: coverage < {HIGH_RISK_COVERAGE:.0%} | Medium: {HIGH_RISK_COVERAGE:.0%}–<{LOW_RISK_COVERAGE:.0%} "
                   f"| Low: ≥{LOW_RISK_COVERAGE:.0%}")
st.sidebar.caption("Coverage = supplied units / demand units. Collection efficiency = supplied / collected.")

model_key = f"{model_paths.name}-{_mtime(model_paths.classifier)}-{_mtime(model_paths.regressor)}"
data_key = f"{dataset_label}-{len(processed)}-{processed.date.max()}"
try:
    prediction = forecast_for(processed, model_paths, model_key + data_key)
except Exception as exc:
    st.error(f"Prediction failed: {exc}. Re-run `python run_pipeline.py` if dependencies changed.")
    st.stop()

high_recall = alert_mode.startswith("High")
risk_col = "predicted_risk_high_recall_mode" if high_recall else "predicted_risk"
st.markdown(f"**Dataset:** {dataset_label}  \n**Models:** classifier = {prediction.classifier.iloc[0]}, "
            f"regressor = {prediction.regressor.iloc[0]} (from `{model_paths.models_dir.relative_to(DEFAULT_PATHS.models_dir.parent)}`)")

# ---------------------------------------------------------------------------
# Section 1: model predictions
# ---------------------------------------------------------------------------
st.header("🔮 Machine-learning forecast (prediction, not an observation)")
st.caption(f"Last recorded week: {prediction.last_observed_week.max().date()} → forecast week: "
           f"{prediction.forecast_week.max().date()}. The forecast is for the week after the final observation, "
           "not the current calendar week.")
if high_recall and prediction.high_recall_threshold.notna().any():
    st.caption(f"High-recall alert mode: HIGH whenever P(High) ≥ {prediction.high_recall_threshold.iloc[0]:.3f}.")

subset = prediction if chosen == "All" else prediction[prediction.blood_type == chosen]
counts = subset[risk_col].value_counts()
if counts.get("High", 0):
    st.error(f"⚠️ {counts['High']} blood type(s) forecast at HIGH risk: "
             + ", ".join(subset.loc[subset[risk_col] == "High", "blood_type"]))
elif counts.get("Medium", 0):
    st.warning(f"{counts['Medium']} blood type(s) forecast at MEDIUM risk; none at HIGH risk.")
else:
    st.success("No blood type is forecast at HIGH or MEDIUM risk.")

cols = st.columns(max(1, len(subset)))
for col, (_, row) in zip(cols, subset.iterrows()):
    risk = row[risk_col]
    color, background, badge, note = RISK_STYLE[risk]
    with col:
        st.markdown(
            f"<div style='border:2px solid {color};background:{background};border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:1.15rem;font-weight:700'>Blood type {row.blood_type}</div>"
            f"<div style='color:{color};font-weight:800;font-size:1.05rem;margin:4px 0'>{badge}</div>"
            f"<div style='font-size:0.85rem'>{note}</div></div>", unsafe_allow_html=True)
        st.metric("Forecast demand (units)", f"{row.predicted_demand:,.0f}",
                  delta=f"{row.predicted_demand - row.observed_demand:+,.0f} vs last observed",
                  delta_color="off")
        st.progress(float(row.p_high), text=f"P(High) = {row.p_high:.1%}")
        st.caption(f"P(Medium) {row.p_medium:.1%} • P(Low) {row.p_low:.1%} • "
                   f"last observed risk: {row.observed_risk}")

with st.expander("Forecast table"):
    st.dataframe(subset[["blood_type", "last_observed_week", "forecast_week", "observed_risk", "predicted_risk",
                         "predicted_risk_high_recall_mode", "p_high", "p_medium", "p_low", "risk_confidence",
                         "predicted_demand", "risk_alert"]].round({"p_high": 3, "p_medium": 3, "p_low": 3, "risk_confidence": 3,
                                                        "predicted_demand": 0}), hide_index=True)

# ---------------------------------------------------------------------------
# Section 2: observed history
# ---------------------------------------------------------------------------
st.header("📊 Observed historical data (actual records, no predictions)")
history = processed if chosen == "All" else processed[processed.blood_type == chosen]
if chosen == "All":
    weekly = history.groupby("date", as_index=False)[["demand", "supply", "collected"]].sum(min_count=1)
    weekly["coverage"] = weekly.supply / weekly.demand
    weekly["collection_efficiency"] = weekly.supply / weekly.collected
else:
    weekly = history[["date", "demand", "supply", "collected", "coverage", "collection_efficiency"]].copy()
weekly = weekly.sort_values("date").tail(window)

left, right = st.columns([3, 2])
with left:
    plot_df = weekly.melt(id_vars="date", value_vars=["demand", "supply", "collected"], var_name="Indicator", value_name="Units")
    st.plotly_chart(px.line(plot_df, x="date", y="Units", color="Indicator",
                            title="Observed supply, demand and collection"))
with right:
    fig = px.line(weekly, x="date", y="coverage", title="Observed coverage (supply / demand)")
    fig.add_hline(y=HIGH_RISK_COVERAGE, line_dash="dash", line_color="#c0392b", annotation_text="High-risk cutoff")
    fig.add_hline(y=LOW_RISK_COVERAGE, line_dash="dot", line_color="#2e7d32", annotation_text="Low-risk cutoff")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig)

st.subheader("Observed indicators (latest recorded week)")
latest = processed.sort_values("date").groupby("blood_type", as_index=False).tail(1)
latest = latest.set_index("blood_type").reindex(list(BLOOD_TYPES)).reset_index()
if chosen != "All":
    latest = latest[latest.blood_type == chosen]
metric_cols = st.columns(len(latest) + 1)
for c, (_, row) in zip(metric_cols, latest.iterrows()):
    with c:
        st.metric(f"{row.blood_type}: coverage", f"{row.coverage:.1%}")
        st.metric(f"{row.blood_type}: collection efficiency", f"{row.collection_efficiency:.1%}")
with metric_cols[-1]:
    st.metric("Weeks shown", f"{len(weekly):,}")
    st.metric("Mean efficiency (window)", f"{weekly.collection_efficiency.mean():.1%}")

eff = history[history.date >= weekly.date.min()]
st.plotly_chart(px.line(eff, x="date", y="collection_efficiency", color="blood_type",
                        title="Observed collection efficiency by blood type (supplied / collected)")
                .update_yaxes(tickformat=".0%"))

risk_counts = history.groupby(["blood_type", "observed_risk"]).size().reset_index(name="Weeks")
st.plotly_chart(px.bar(risk_counts, x="blood_type", y="Weeks", color="observed_risk", barmode="group",
                       category_orders={"observed_risk": ["High", "Medium", "Low"], "blood_type": list(BLOOD_TYPES)},
                       color_discrete_map={k: v[0] for k, v in RISK_STYLE.items()},
                       title="Historical shortage-risk distribution (observed weeks, not forecasts)"))

# ---------------------------------------------------------------------------
# Section 3: model evaluation
# ---------------------------------------------------------------------------
st.header("🧪 Model evaluation on held-out chronological test data")
results = model_paths.results_dir
if source == UPLOAD:
    evaluation = external_evaluation(processed, model_paths, model_key + data_key)
    if evaluation:
        st.markdown("**Existing models scored on the uploaded file** (no retraining; if the file overlaps the "
                    "Zhejiang training period these scores are optimistic).")
        for task, table in evaluation.items():
            st.dataframe(table.round(3), hide_index=True)

class_path, reg_path = model_paths.class_metrics, model_paths.reg_metrics
if class_path.exists() and reg_path.exists():
    class_scores, reg_scores = pd.read_csv(class_path), pd.read_csv(reg_path)
    best_c = class_scores.loc[class_scores.selected].iloc[0]
    best_r = reg_scores.loc[reg_scores.selected].iloc[0]
    protocol = str(best_c.get("test_protocol", "static")).replace("_", "-")
    st.caption(f"Test protocol: **{protocol}**"
               + (" (models retrained every few weeks on every outcome known at that time, as in practice)."
                  if protocol == "walk-forward" else " (one model fitted before the test period)."))
    v1_dir = results / "previous_version_v1"
    v1_c = v1_r = None
    if (v1_dir / "classification_metrics.csv").exists() and (v1_dir / "regression_metrics.csv").exists():
        v1c, v1r = pd.read_csv(v1_dir / "classification_metrics.csv"), pd.read_csv(v1_dir / "regression_metrics.csv")
        v1_c, v1_r = v1c.loc[v1c.selected].iloc[0], v1r.loc[v1r.selected].iloc[0]

    def _delta(new, old_row, col, pct=True):
        if old_row is None or col not in old_row:
            return None
        diff = new - old_row[col]
        return f"{diff * 100:+.1f} pts vs v1" if pct else f"{diff:+.1f} vs v1"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Test accuracy", f"{best_c.test_accuracy:.1%}", delta=_delta(best_c.test_accuracy, v1_c, "test_accuracy"),
              help="Proposal target: 80%")
    k2.metric("Test macro-F1", f"{best_c.test_macro_f1:.3f}",
              delta=None if v1_c is None else f"{best_c.test_macro_f1 - v1_c.test_macro_f1:+.3f} vs v1")
    k3.metric("Test HIGH recall", f"{best_c.test_high_recall:.1%}",
              delta=_delta(best_c.test_high_recall, v1_c, "test_high_recall"), help="Proposal target: 75%")
    if "test_high_recall_mode_high_recall" in best_c:
        k4.metric("HIGH recall (alert mode)", f"{best_c.test_high_recall_mode_high_recall:.1%}")
    k5.metric("Test demand MAE", f"{best_r.test_mae:,.1f} units",
              delta=_delta(best_r.test_mae, v1_r, "test_mae", pct=False), delta_color="inverse")

    tab1, tab2, tab3, tab4 = st.tabs(["Classification", "Regression", "Before vs after", "Figures"])
    with tab1:
        cols_c = [c for c in ["model", "selected", "feature_set", "weighting", "train_macro_f1", "cv_macro_f1",
                              "validation_macro_f1", "validation_wf_macro_f1", "test_accuracy", "test_macro_precision",
                              "test_macro_recall", "test_macro_f1", "test_high_recall", "test_static_accuracy",
                              "test_static_high_recall", "test_high_recall_mode_high_recall",
                              "test_high_recall_mode_accuracy"] if c in class_scores]
        st.dataframe(class_scores[cols_c].round(3), hide_index=True)
        st.plotly_chart(px.bar(class_scores, x="model", y=["test_macro_f1", "test_high_recall"], barmode="group",
                               labels={"value": "Score", "model": "Algorithm"},
                               title="Classification: test macro-F1 and HIGH-risk recall"))
        cm_path = results / "classification_confusion_matrices.csv"
        if cm_path.exists():
            cm = pd.read_csv(cm_path)
            cm = cm[(cm.model == best_c.model)]
            c1, c2 = st.columns(2)
            for col, mode in zip((c1, c2), ("standard", "high_recall_mode")):
                part = cm[cm["mode"] == mode].pivot(index="actual", columns="predicted", values="count")
                part = part.reindex(index=["High", "Medium", "Low"], columns=["High", "Medium", "Low"])
                col.plotly_chart(px.imshow(part, text_auto=True, color_continuous_scale="Blues",
                                           title=f"{best_c.model} confusion matrix: {mode.replace('_', ' ')}",
                                           labels={"x": "Predicted", "y": "Actual"}))
        per_class = results / "classification_per_class_report.csv"
        if per_class.exists():
            with st.expander("Per-class precision / recall / F1"):
                st.dataframe(pd.read_csv(per_class).round(3), hide_index=True)
    with tab2:
        cols_r = [c for c in ["model", "selected", "feature_set", "train_mae", "cv_mae", "validation_mae", "validation_wf_mae",
                              "test_mae", "test_rmse", "test_mape_percent", "test_static_mae"] if c in reg_scores]
        st.dataframe(reg_scores[cols_r].round(2), hide_index=True)
        pred_path = results / "regression_test_predictions.csv"
        if pred_path.exists():
            preds = pd.read_csv(pred_path, parse_dates=["target_date"])
            col_name = f"pred_{safe_name(best_r.model)}"
            if col_name in preds:
                show = preds if chosen == "All" else preds[preds.blood_type == chosen]
                st.plotly_chart(px.scatter(show, x="actual_demand", y=col_name, color="blood_type",
                                           labels={"actual_demand": "Observed next-week demand", col_name: "Predicted"},
                                           title=f"Test period: observed vs predicted demand ({best_r.model})"))
        per_type = results / "regression_per_blood_type.csv"
        if per_type.exists():
            with st.expander("Error by blood type"):
                st.dataframe(pd.read_csv(per_type).round(2), hide_index=True)
    with tab3:
        for name in ("classification_before_after.csv", "regression_before_after.csv"):
            if (results / name).exists():
                table = pd.read_csv(results / name)
                keep = [c for c in ["model", "stage", "cv_macro_f1", "test_accuracy", "test_macro_f1", "test_high_recall",
                                    "cv_mae", "test_mae", "test_rmse", "test_mape_percent"] if c in table]
                st.markdown(f"**{name.split('_')[0].title()}**")
                st.dataframe(table[keep].round(3), hide_index=True)
    with tab4:
        figures = sorted(model_paths.figures_dir.glob("*.png")) if model_paths.figures_dir.exists() else []
        if not figures:
            st.info("No saved figures yet. Run `python run_pipeline.py`.")
        for fig_path in figures:
            st.image(str(fig_path), caption=fig_path.stem.replace("_", " "))
else:
    st.info("Run `python run_pipeline.py` to generate the model comparison files.")

with st.expander("Methodology and limitations"):
    st.markdown(f"""
    - **Target:** next observed week, only if it is 5–9 days after the current date.
    - **Risk labels:** High < {HIGH_RISK_COVERAGE:.0%} coverage, Medium {HIGH_RISK_COVERAGE:.0%}–<{LOW_RISK_COVERAGE:.0%}, Low ≥ {LOW_RISK_COVERAGE:.0%} (unchanged from the proposal).
    - **Features:** latest known demand, supply, collection, coverage, weekly lags, rolling averages, calendar of the
      observation week and of the target week (incl. Spring Festival proximity). No `next_*` value is ever a feature.
    - **Splitting:** chronological 70% train, 15% validation, 15% test, based on target dates.
    - **Tuning:** random search over hyper-parameters, class weighting and feature set, scored by expanding-window
      time-series cross-validation inside train+validation. The test period is used once, after all choices.
    - **Choice of models:** best CV macro-F1 (classification) and CV MAE (regression), among Decision Tree,
      Random Forest, XGBoost and an average of the tuned Random Forest and XGBoost (version 2).
    - **Version 2 test protocol (walk-forward):** through the validation and test periods each model is retrained
      every few weeks on every outcome known at that time, as a blood bank would. Static scores (one model fitted
      before the test period, the version-1 protocol) are shown alongside.
    - **Version 2 features:** a stock proxy (net collected minus supplied over 4/13/26 weeks, in weeks of demand),
      lagged collection, and same-week coverage of all four blood types. No inventory column exists in the data.
    - **Forecast model:** refit on every labeled week before forecasting the week after the last observation.
    - **High-recall alert mode:** P(High) threshold chosen on cross-validation predictions to reach 75% HIGH recall
      there; on later data it may reach less because of temporal drift.
    - **Data cleaning:** same-date conflicting rows are averaged; this assumption should be checked with the data provider.
    - **Scope:** a historical retrospective demonstration, not a real-time blood-bank alert service. Probabilities are
      model scores, not calibrated medical probabilities. Zhejiang results do not validate use in other countries.
    """)
