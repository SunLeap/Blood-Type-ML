from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from src.config import BLOOD_TYPES, RISK_LABELS, TARGET_ACCURACY, TARGET_HIGH_RECALL, ProjectPaths

RISK_COLORS = {"High": "#c0392b", "Medium": "#e1a325", "Low": "#2e8b57"}
SERIES = ["#2f6db3", "#e07b39", "#6a9f58", "#8e6bb8", "#7f7f7f"]


def _save(fig: Figure, paths: ProjectPaths, name: str) -> None:
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / name, dpi=130)


def _confusion(ax, y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(3), RISK_LABELS)
    ax.set_yticks(range(3), RISK_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)


def _grouped_bars(ax, table: pd.DataFrame, columns: list[str], labels: list[str]) -> None:
    x = np.arange(len(table))
    width = 0.8 / len(columns)
    for k, (col, lab) in enumerate(zip(columns, labels)):
        ax.bar(x + (k - (len(columns) - 1) / 2) * width, table[col], width, label=lab, color=SERIES[k % len(SERIES)])
    ax.set_xticks(x, table["model"], rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def make_task_figures(task, paths, test, evaluated, best_name, comparison, result) -> None:
    if task == "classification":
        classification_figures(paths, test, evaluated, best_name, comparison, result)
    else:
        regression_figures(paths, test, evaluated, best_name, comparison, result)


def classification_figures(paths, test, evaluated, best_name, comparison, result) -> None:
    from src.prediction import LABEL_TO_ID
    y_true = test.next_risk.map(LABEL_TO_ID).astype(int).to_numpy()

    n = len(evaluated)
    fig = Figure(figsize=(4.4 * n, 4))
    for k, (family, (_, ev)) in enumerate(evaluated.items()):
        _confusion(fig.add_subplot(1, n, k + 1), y_true, ev["test_pred"],
                   f"{family}{' (selected)' if family == best_name else ''}")
    fig.suptitle("Held-out chronological test weeks - optimized classifiers")
    _save(fig, paths, "confusion_matrices_test.png")

    ev = evaluated[best_name][1]
    fig = Figure(figsize=(9, 4))
    _confusion(fig.add_subplot(1, 2, 1), y_true, ev["test_pred"], f"{best_name}: standard (most probable class)")
    _confusion(fig.add_subplot(1, 2, 2), y_true, ev["hr_pred"],
               f"{best_name}: high-recall alert mode (P(High) >= {ev['threshold']:.3f})")
    _save(fig, paths, "confusion_matrix_selected_modes.png")

    fig = Figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    _grouped_bars(ax, comparison, ["test_accuracy", "test_macro_f1", "test_high_recall"],
                  ["Test accuracy", "Test macro-F1", "Test High recall"])
    ax.axhline(TARGET_ACCURACY, ls="--", color="grey", lw=1)
    ax.axhline(TARGET_HIGH_RECALL, ls=":", color="black", lw=1)
    ax.set_ylim(0, 1)
    ax.set_title("Classification before vs after optimization (dashed = 80% accuracy target, dotted = 75% High-recall target)", fontsize=9)
    _save(fig, paths, "classification_before_after.png")

    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    _grouped_bars(ax, result, ["train_macro_f1", "cv_macro_f1", "validation_macro_f1", "test_macro_f1"],
                  ["Train", "Time-series CV", "Validation", "Test"])
    ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 by partition (large train/test gap = over-fitting)", fontsize=10)
    _save(fig, paths, "classification_train_validation_test.png")

    curve = ev["curve"]
    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    for col, lab, c in (("cv_high_recall", "High recall", SERIES[0]), ("cv_high_precision", "High precision", SERIES[1]),
                        ("cv_accuracy", "Accuracy", SERIES[2]), ("cv_macro_f1", "Macro-F1", SERIES[3])):
        ax.plot(curve.threshold, curve[col], label=lab, color=c)
    ax.axvline(ev["threshold"], ls="--", color="black", lw=1, label=f"chosen = {ev['threshold']:.3f}")
    ax.axhline(TARGET_HIGH_RECALL, ls=":", color="grey", lw=1)
    ax.set_xlabel("P(High) alert threshold")
    ax.set_title(f"{best_name}: cross-validation trade-off (development period only)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, paths, "high_risk_threshold_curve.png")
    _protocol_figure(paths, result, ["test_static_accuracy", "test_walk_forward_accuracy", "test_static_high_recall",
                                     "test_walk_forward_high_recall"],
                     ["Accuracy, static", "Accuracy, walk-forward", "High recall, static", "High recall, walk-forward"],
                     "classification", "Score")
    _importance_figure(paths, "classification", best_name)


def _protocol_figure(paths, result, columns, labels, task, ylabel) -> None:
    if not set(columns) <= set(result.columns):
        return
    fig = Figure(figsize=(9, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    _grouped_bars(ax, result, columns, labels)
    ax.set_ylabel(ylabel)
    if task == "classification":
        ax.set_ylim(0, 1)
        ax.axhline(TARGET_ACCURACY, ls="--", color="grey", lw=1)
    ax.set_title("Test period: one model fitted before the test (static) vs retrained every few weeks (walk-forward)",
                 fontsize=9)
    _save(fig, paths, f"{task}_static_vs_walk_forward.png")


def regression_figures(paths, test, evaluated, best_name, comparison, result) -> None:
    ev = evaluated[best_name][1]
    actual, predicted = test.next_demand.to_numpy(), ev["test_pred"]
    fig = Figure(figsize=(6, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    for k, bt in enumerate(BLOOD_TYPES):
        m = (test.blood_type == bt).to_numpy()
        ax.scatter(actual[m], predicted[m], s=14, alpha=0.6, label=bt, color=SERIES[k])
    lim = float(max(actual.max(), predicted.max())) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="grey")
    ax.set_xlabel("Observed next-week demand")
    ax.set_ylabel("Predicted next-week demand")
    ax.set_title(f"Test forecasts: {best_name}")
    ax.legend(title="Blood type")
    _save(fig, paths, "regression_actual_vs_predicted.png")

    fig = Figure(figsize=(12, 7))
    for k, bt in enumerate(BLOOD_TYPES):
        ax = fig.add_subplot(2, 2, k + 1)
        m = (test.blood_type == bt).to_numpy()
        dates = test.target_date.to_numpy()[m]
        ax.plot(dates, actual[m], label="Observed", color="black", lw=1.4)
        ax.plot(dates, predicted[m], label="Predicted", color=SERIES[1], lw=1.2)
        ax.set_title(f"Blood type {bt}", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Next-week demand over the test period ({best_name})")
    _save(fig, paths, "regression_actual_vs_predicted_timeseries.png")

    fig = Figure(figsize=(10, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    _grouped_bars(ax, comparison, ["validation_mae", "cv_mae", "test_mae"], ["Validation MAE", "CV MAE", "Test MAE"])
    ax.set_ylabel("Units (lower is better)")
    ax.set_title("Regression before vs after optimization", fontsize=10)
    _save(fig, paths, "regression_before_after.png")

    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    _grouped_bars(ax, result, ["train_mae", "cv_mae", "validation_mae", "test_mae"],
                  ["Train", "Time-series CV", "Validation", "Test"])
    ax.set_ylabel("MAE (units)")
    ax.set_title("MAE by partition (large train/test gap = over-fitting)", fontsize=10)
    _save(fig, paths, "regression_train_validation_test.png")
    _protocol_figure(paths, result, ["test_static_mae", "test_walk_forward_mae"], ["MAE, static", "MAE, walk-forward"],
                     "regression", "MAE (units, lower is better)")
    _importance_figure(paths, "regression", best_name)


def _importance_figure(paths: ProjectPaths, task: str, family: str, top: int = 20) -> None:
    table = pd.read_csv(paths.result(f"{task}_feature_importance.csv"))
    if not (table.family == family).any():   # the ensemble has no own importance: show its XGBoost member
        family = "XGBoost" if (table.family == "XGBoost").any() else str(table.family.iloc[0])
    table = table[table.family == family].head(top).iloc[::-1]
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    colors = [SERIES[0] if v > 0 else SERIES[4] for v in table.importance_mean]
    ax.barh(table.feature, table.importance_mean, xerr=table.importance_std, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.tick_params(axis="y", labelsize=8)
    metric = "macro-F1" if task == "classification" else "MAE"
    ax.set_title(f"{family} permutation importance (drop in {metric}, early holdout)", fontsize=10)
    _save(fig, paths, f"feature_importance_{task}.png")
