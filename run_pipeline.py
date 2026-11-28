"""Single entry point: prepare data -> tune and train models -> evaluate -> save -> report.

CMD (primary Zhejiang experiment):   python run_pipeline.py
CMD (faster, fewer candidates):       python run_pipeline.py --quick
CMD (train on an EXTERNAL dataset):   python run_pipeline.py --data path\\to\\other.csv --run-name my_test
CMD (existing models, new data):      python run_pipeline.py --predict-only path\\to\\other.csv
CMD (corrected Zhejiang copy):        python run_pipeline.py --corrected   (repairs date/entry errors, trains in runs/corrected/)

Training on an external dataset writes ONLY to runs/<run-name>/ and never touches
data/, models/ or results/ of the primary experiment.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import (
    EVALUATION_PROTOCOL, QUICK_SEARCH_ITERATIONS, RAW_DATA, SEARCH_ITERATIONS, TARGET_ACCURACY, TARGET_HIGH_RECALL,
    WALK_FORWARD_STEP, ProjectPaths,
)
from src.data_pipeline import generate_processed, prepare_dataset
from src.prediction import predict_external, predict_next_week, train_classification, train_regression


def run(data: Path | None = None, run_name: str | None = None, quick: bool = False,
        figures: bool = True, log=print) -> dict:
    """Run the full pipeline and return the summary tables."""
    started = time.time()
    if data is None or Path(data).resolve() == RAW_DATA.resolve():
        if run_name:
            paths = ProjectPaths.for_run(run_name, RAW_DATA)
        else:
            paths = ProjectPaths.default()
    else:
        if not run_name:
            raise ValueError("Training on an external dataset needs --run-name so results go to runs/<name>/.")
        prepared = prepare_dataset(Path(data))
        log(prepared.report.summary())
        if not prepared.report.can_train:
            raise ValueError("This dataset cannot be used for training (see errors/warnings above).")
        paths = ProjectPaths.for_run(run_name)
        paths.ensure()
        shutil.copy2(data, paths.raw_data)  # work on a copy; the user's file is never modified
    paths.ensure()
    n_iter = QUICK_SEARCH_ITERATIONS if quick else SEARCH_ITERATIONS

    log(f"Step 1/4  Preparing data from {paths.raw_data}")
    processed = generate_processed(paths=paths)
    log(f"          {processed.date.nunique()} dates, {len(processed)} blood-type rows, "
        f"{processed.next_risk.notna().sum()} next-week labels")
    log("Step 2/4  Classification (tuning with time-series CV, then one test evaluation)")
    classification = train_classification(processed, paths, n_iter=n_iter, make_figures=figures, log=log)
    log("Step 3/4  Regression")
    regression = train_regression(processed, paths, n_iter=n_iter, make_figures=figures, log=log)
    log("Step 4/4  Forecast and report")
    forecast = predict_next_week(processed, paths)
    forecast.to_csv(paths.result("next_week_forecast.csv"), index=False)
    report = write_report(paths, classification, regression, forecast, n_iter, time.time() - started)
    log(report)
    return {"paths": paths, "classification": classification, "regression": regression, "forecast": forecast}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def write_report(paths: ProjectPaths, classification: pd.DataFrame, regression: pd.DataFrame,
                 forecast: pd.DataFrame, n_iter: int, seconds: float) -> str:
    """Write results/summary_report.md and return it as text."""
    c = classification.loc[classification.selected].iloc[0]
    r = regression.loc[regression.selected].iloc[0]
    before_c = pd.read_csv(paths.result("classification_before_after.csv"))
    before_r = pd.read_csv(paths.result("regression_before_after.csv"))
    ci_c = pd.read_csv(paths.result("classification_test_confidence_intervals.csv")).set_index("metric")
    ci_r = pd.read_csv(paths.result("regression_test_confidence_intervals.csv")).set_index("metric")
    lines = [
        "# Model performance summary", "",
        f"Run: `{paths.name}` | search candidates per family: {n_iter} | runtime: "
        f"{f'{seconds / 60:.1f} min' if seconds > 0 else 'not measured (notebook run)'}", "",
        f"Reported test protocol: **{EVALUATION_PROTOCOL.replace('_', '-')}** (models retrained every {WALK_FORWARD_STEP} weeks "
        "on every outcome known at that time). Static scores are shown for comparison with version 1.", "",
        "## Classification (next-week risk)", "",
        f"Selected by time-series CV macro-F1: **{c.model}** (feature set: {c.feature_set}, "
        f"{int(c.n_features)} numeric features, weighting: {c.weighting})", "",
        "| Partition | Accuracy | Macro-F1 | High recall |", "|---|---:|---:|---:|",
        f"| Train | {_pct(c.train_accuracy)} | {c.train_macro_f1:.3f} | {_pct(c.train_high_recall)} |",
        f"| Time-series CV | {_pct(c.cv_accuracy)} | {c.cv_macro_f1:.3f} | {_pct(c.cv_high_recall)} |",
        f"| Validation (static) | {_pct(c.validation_accuracy)} | {c.validation_macro_f1:.3f} | {_pct(c.validation_high_recall)} |",
        f"| Validation (walk-forward) | {_pct(c.validation_wf_accuracy)} | {c.validation_wf_macro_f1:.3f} | {_pct(c.validation_wf_high_recall)} |",
        f"| Test (static: one model fitted before the test period, version-1 protocol) | {_pct(c.test_static_accuracy)} | "
        f"{c.test_static_macro_f1:.3f} | {_pct(c.test_static_high_recall)} |",
        f"| **Test ({c.test_protocol.replace('_', '-')}, standard rule)** | **{_pct(c.test_accuracy)}** | **{c.test_macro_f1:.3f}** | **{_pct(c.test_high_recall)}** |",
        f"| Test (high-recall alert mode, P(High) >= {c.high_recall_threshold:.3f}) | "
        f"{_pct(c.test_high_recall_mode_accuracy)} | {c.test_high_recall_mode_macro_f1:.3f} | "
        f"{_pct(c.test_high_recall_mode_high_recall)} |", "",
        f"Test 95% bootstrap intervals (standard mode): accuracy {_pct(ci_c.loc['accuracy','lower_95'])}-"
        f"{_pct(ci_c.loc['accuracy','upper_95'])}, macro-F1 {ci_c.loc['macro_f1','lower_95']:.3f}-"
        f"{ci_c.loc['macro_f1','upper_95']:.3f}, High recall {_pct(ci_c.loc['high_recall','lower_95'])}-"
        f"{_pct(ci_c.loc['high_recall','upper_95'])}", "",
        f"Proposal targets: accuracy >= {_pct(TARGET_ACCURACY)} -> "
        f"{'met' if c.test_accuracy >= TARGET_ACCURACY else 'NOT met'}; High recall >= {_pct(TARGET_HIGH_RECALL)} -> "
        f"{'met' if c.test_high_recall >= TARGET_HIGH_RECALL else 'NOT met'} (standard), "
        f"{'met' if c.test_high_recall_mode_high_recall >= TARGET_HIGH_RECALL else 'NOT met'} (high-recall mode).", "",
        "### Before vs after optimization (test period)", "",
        "| Model | Stage | CV macro-F1 | Test accuracy | Test macro-F1 | Test High recall |", "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in before_c.iterrows():
        lines.append(f"| {row.model} | {row.stage} | {row.cv_macro_f1:.3f} | {_pct(row.test_accuracy)} | "
                     f"{row.test_macro_f1:.3f} | {_pct(row.test_high_recall)} |")
    lines += ["", "## Regression (next-week demand)", "",
              f"Selected by time-series CV MAE: **{r.model}** (feature set: {r.feature_set}, {int(r.n_features)} numeric features)", "",
              f"Test MAE {r.test_mae:.1f} units (95% CI {ci_r.loc['mae','lower_95']:.1f}-{ci_r.loc['mae','upper_95']:.1f}), "
              f"RMSE {r.test_rmse:.1f}, MAPE {r.test_mape_percent:.2f}% "
              f"(static protocol: MAE {r.test_static_mae:.1f}; walk-forward validation MAE {r.validation_wf_mae:.1f})", "",
              "| Model | Stage | CV MAE | Validation MAE | Test MAE | Test RMSE | Test MAPE % |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in before_r.iterrows():
        lines.append(f"| {row.model} | {row.stage} | {row.cv_mae:.1f} | {row.validation_mae:.1f} | {row.test_mae:.1f} | "
                     f"{row.test_rmse:.1f} | {row.test_mape_percent:.2f} |")
    lines += version1_comparison(paths, c, r)
    lines += ["", "## Next-week forecast (from the last observed week)", "",
              "| Blood type | Last observed | Forecast week | Observed risk | Predicted risk | P(High) | Predicted demand |",
              "|---|---|---|---|---|---:|---:|"]
    for _, row in forecast.iterrows():
        lines.append(f"| {row.blood_type} | {row.last_observed_week.date()} | {row.forecast_week.date()} | "
                     f"{row.observed_risk} | {row.predicted_risk} | {row.p_high:.2f} | {row.predicted_demand:,.0f} |")
    lines += ["", "Forecasts are model estimates, not observations. Results are retrospective (Zhejiang data) "
              "and do not validate operational or clinical use."]
    text = "\n".join(lines)
    paths.result("summary_report.md").write_text(text, encoding="utf-8")
    return text


def version1_comparison(paths: ProjectPaths, c: pd.Series, r: pd.Series) -> list[str]:
    """Compare with the selected models saved by version 1 (results/previous_version_v1/)."""
    folder = paths.results_dir / "previous_version_v1"
    if not (folder / "classification_metrics.csv").exists():
        return []
    vc = pd.read_csv(folder / "classification_metrics.csv")
    vr = pd.read_csv(folder / "regression_metrics.csv")
    vc, vr = vc.loc[vc.selected].iloc[0], vr.loc[vr.selected].iloc[0]
    return ["", "## Version 1 vs version 2 (selected models, test period)", "",
            "| Version | Classifier | Test accuracy | Test macro-F1 | Test High recall | High recall (alert mode) | Regressor | Test MAE |",
            "|---|---|---:|---:|---:|---:|---|---:|",
            f"| Version 1 (static) | {vc.model} | {_pct(vc.test_accuracy)} | {vc.test_macro_f1:.3f} | {_pct(vc.test_high_recall)} | "
            f"{_pct(vc.get('test_high_recall_mode_high_recall', float('nan')))} | {vr.model} | {vr.test_mae:.1f} |",
            f"| Version 2 ({c.test_protocol.replace('_', '-')}) | {c.model} | {_pct(c.test_accuracy)} | {c.test_macro_f1:.3f} | "
            f"{_pct(c.test_high_recall)} | {_pct(c.test_high_recall_mode_high_recall)} | {r.model} | {r.test_mae:.1f} |"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Blood Shortage ML - full pipeline")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_SEARCH_ITERATIONS} search candidates per model instead of {SEARCH_ITERATIONS}")
    parser.add_argument("--data", type=Path, help="external CSV to TRAIN on (requires --run-name)")
    parser.add_argument("--run-name", help="folder name under runs/ for an external experiment")
    parser.add_argument("--predict-only", type=Path, metavar="CSV",
                        help="use the EXISTING trained models on a new CSV (no training, nothing overwritten)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--corrected", action="store_true",
                        help="repair data-entry errors into a corrected copy and train on it in runs/corrected/")
    args = parser.parse_args()
    try:
        if args.predict_only:
            out = predict_external(args.predict_only, name=args.predict_only.stem)
            print(out["report"].summary())
            if out["forecast"] is None:
                sys.exit(1)
            print(out["forecast"][["blood_type", "last_observed_week", "forecast_week", "predicted_risk",
                                   "p_high", "predicted_demand", "risk_alert"]].to_string(index=False))
            for task, table in out["evaluation"].items():
                print(f"\nExisting {task} model on this file:\n{table.round(4).to_string(index=False)}")
            print(f"\nSaved to {out['output_dir']}")
            return
        if args.corrected:
            from src.data_repair import CORRECTED_DATA, repair
            log = repair()
            print(f"Corrected copy written to {CORRECTED_DATA} ({len(log)} changes, see results/data_repair_log.csv)")
            run(CORRECTED_DATA, args.run_name or "corrected", args.quick, not args.no_figures)
            return
        run(args.data, args.run_name, args.quick, not args.no_figures)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
