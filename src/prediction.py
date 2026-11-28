from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    BLOOD_TYPES, BOOTSTRAP_SAMPLES, CV_MIN_TRAIN_FRACTION, CV_SPLITS, DEFAULT_PATHS, ENSEMBLE_NAME,
    EVALUATION_PROTOCOL, WALK_FORWARD_STEP,
    HIGH_RISK_COVERAGE, LOW_RISK_COVERAGE, N_JOBS, RANDOM_STATE, RESULTS_DIR, RISK_LABELS,
    SEARCH_ITERATIONS, TARGET_ACCURACY, TARGET_GAP_DAYS, TARGET_HIGH_RECALL, TRAIN_FRACTION,
    VALIDATION_FRACTION, ProjectPaths, safe_name,
    CLASSIFIER_PATH, REGRESSOR_PATH, CLASS_METRICS_PATH, REG_METRICS_PATH,  # noqa: F401 (re-exported)
)
from src.data_pipeline import (
    ALL_NUMERICAL_FEATURES, FEATURE_COLUMNS, FEATURE_SETS, NUMERICAL_FEATURES,  # noqa: F401
    V2_NUMERICAL_FEATURES, EXTERNAL_NUMERICAL_FEATURES, has_external_features,  # noqa: F401
    ORIGINAL_NUMERICAL_FEATURES, CsvSource, check_no_leakage, feature_columns, load_processed,
    generate_processed, prepare_dataset, risk_from_coverage,  # noqa: F401
)

LABEL_TO_ID = {label: i for i, label in enumerate(RISK_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
HIGH_ID = LABEL_TO_ID["High"]
TASKS = ("classification", "regression")
FAMILIES = ("Decision Tree", "Random Forest", "XGBoost")
ALERTS = {"High": "WARNING", "Medium": "NOTICE", "Low": "NONE"}

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
# The original project's settings are kept as the "baseline" for before/after comparison.
BASELINE_PARAMS = {
    "classification": {
        "Decision Tree": {"max_depth": 6, "min_samples_leaf": 8},
        "Random Forest": {"n_estimators": 120, "max_depth": 9, "min_samples_leaf": 4},
        "XGBoost": {"n_estimators": 130, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.9,
                    "colsample_bytree": 0.9},
    },
    "regression": {
        "Decision Tree": {"max_depth": 7, "min_samples_leaf": 8},
        "Random Forest": {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 3},
        "XGBoost": {"n_estimators": 130, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.9,
                    "colsample_bytree": 0.9},
    },
}
BASELINE_WEIGHTING = {"classification": "balanced", "regression": "none"}

# Class-weighting strategies investigated for the imbalanced risk labels.
WEIGHTING_STRATEGIES = ("none", "balanced", "sqrt_balanced", "balanced_high_boost")

SEARCH_SPACES = {
    "classification": {
        "Decision Tree": {"max_depth": [3, 4, 5, 6, 8, 10, None], "min_samples_leaf": [4, 8, 16, 32, 64],
                          "criterion": ["gini", "entropy"], "max_features": [None, "sqrt", 0.5],
                          "weighting": list(WEIGHTING_STRATEGIES)},
        "Random Forest": {"n_estimators": [150, 300, 500], "max_depth": [4, 6, 8, 10, 14, None],
                          "min_samples_leaf": [2, 4, 8, 16], "max_features": ["sqrt", 0.3, 0.5],
                          "weighting": list(WEIGHTING_STRATEGIES)},
        "XGBoost": {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5],
                    "learning_rate": [0.02, 0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 0.9],
                    "colsample_bytree": [0.5, 0.7, 0.9], "min_child_weight": [1, 3, 5, 10],
                    "reg_lambda": [1.0, 5.0], "weighting": list(WEIGHTING_STRATEGIES)},
    },
    "regression": {
        "Decision Tree": {"max_depth": [3, 4, 5, 6, 8, 10, None], "min_samples_leaf": [4, 8, 16, 32],
                          "max_features": [None, "sqrt", 0.5]},
        "Random Forest": {"n_estimators": [150, 300, 500], "max_depth": [6, 8, 10, 14, None],
                          "min_samples_leaf": [1, 3, 5, 10], "max_features": [1.0, 0.5, "sqrt"]},
        "XGBoost": {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5],
                    "learning_rate": [0.02, 0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 0.9],
                    "colsample_bytree": [0.5, 0.7, 0.9], "min_child_weight": [1, 3, 5, 10],
                    "objective": ["reg:squarederror", "reg:absoluteerror"]},
    },
}


def build_estimator(task: str, family: str, params: dict):
    """Create one unfitted estimator with fixed seeds and thread counts."""
    params = {k: v for k, v in params.items() if k != "weighting"}
    if task == "classification":
        if family == "Decision Tree":
            return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
        if family == "Random Forest":
            return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, **params)
        if family == "XGBoost":
            return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                                 tree_method="hist", random_state=RANDOM_STATE, n_jobs=N_JOBS, **params)
    else:
        if family == "Decision Tree":
            return DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
        if family == "Random Forest":
            return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **params)
        if family == "XGBoost":
            params.setdefault("objective", "reg:squarederror")
            return XGBRegressor(tree_method="hist", random_state=RANDOM_STATE, n_jobs=N_JOBS, **params)
    raise ValueError(f"Unknown task/family: {task}/{family}")


# Backwards-compatible model dictionaries (original settings).
def classification_models() -> dict:
    return {f: build_estimator("classification", f, BASELINE_PARAMS["classification"][f]) for f in FAMILIES}


def regression_models() -> dict:
    return {f: build_estimator("regression", f, BASELINE_PARAMS["regression"][f]) for f in FAMILIES}


def make_pipeline(model, numerical: list[str] | None = None) -> Pipeline:
    """Fit imputers and one-hot encoder on training data ONLY."""
    numerical = list(numerical or NUMERICAL_FEATURES)
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", add_indicator=False), numerical),
            ("blood_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["blood_type"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def chronological_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by prediction TARGET dates, keeping all blood types in the same period.

    70% earliest target weeks = train; next 15% = validation; final 15% = test.
    We never shuffle or use a random train/test split.
    """
    dates = np.array(sorted(df.target_date.dropna().unique()))
    if len(dates) < 30:
        raise ValueError("Need at least 30 unique target dates to make meaningful temporal splits.")
    a = max(1, int(len(dates) * TRAIN_FRACTION))
    b = max(a + 1, int(len(dates) * (TRAIN_FRACTION + VALIDATION_FRACTION)))
    first_end, second_end = dates[a - 1], dates[b - 1]
    parts = {
        "train": df.loc[df.target_date <= first_end].copy(),
        "validation": df.loc[(df.target_date > first_end) & (df.target_date <= second_end)].copy(),
        "test": df.loc[df.target_date > second_end].copy(),
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("Empty temporal partition. Examine your dataset.")
    return parts


def time_series_folds(development: pd.DataFrame, n_splits: int = CV_SPLITS,
                      min_train_fraction: float = CV_MIN_TRAIN_FRACTION) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds over development target dates (never touches the test period).

    Fold k trains on every target date up to a cut-off and validates on the next block
    of dates, so each model is always evaluated on weeks later than those it learned from.
    """
    dates = np.array(sorted(development.target_date.dropna().unique()))
    start = int(len(dates) * min_train_fraction)
    block = (len(dates) - start) // n_splits
    if start < 10 or block < 3:
        raise ValueError("Not enough development dates for time-series cross-validation.")
    folds = []
    for k in range(n_splits):
        train_end = dates[start + k * block - 1]
        val_end = dates[-1] if k == n_splits - 1 else dates[start + (k + 1) * block - 1]
        train = development.loc[development.target_date <= train_end]
        val = development.loc[(development.target_date > train_end) & (development.target_date <= val_end)]
        folds.append((train, val))
    return folds


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------
def target_values(task: str, df: pd.DataFrame) -> np.ndarray:
    if task == "classification":
        return df.next_risk.map(LABEL_TO_ID).astype(int).to_numpy()
    return df.next_demand.astype(float).to_numpy()


def sample_weights(y: np.ndarray, strategy: str) -> np.ndarray | None:
    """Return per-row weights for an imbalance strategy (None = unweighted)."""
    if strategy in (None, "none"):
        return None
    weights = compute_sample_weight(class_weight="balanced", y=y)
    if strategy == "balanced":
        return weights
    if strategy == "sqrt_balanced":
        return np.sqrt(weights)
    if strategy == "balanced_high_boost":
        return weights * np.where(y == HIGH_ID, 1.5, 1.0)
    raise ValueError(f"Unknown weighting strategy: {strategy}")


class AveragingEnsemble:
    """Average of already-fitted member pipelines (probabilities for classification).

    Each member keeps its own tuned hyper-parameters, weighting and feature set; the
    ensemble receives the union of their input columns.
    """

    def __init__(self, task: str, members: list[Pipeline]):
        self.task, self.members = task, members

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return self.predict_proba(X).argmax(axis=1)
        return np.mean([m.predict(X) for m in self.members], axis=0)


def ensemble_config(member_configs: list[dict]) -> dict:
    features = [f for f in ALL_NUMERICAL_FEATURES + EXTERNAL_NUMERICAL_FEATURES
                if any(f in c["features"] for c in member_configs)]
    return {"family": ENSEMBLE_NAME, "members": member_configs, "params": {"members": [c["family"] for c in member_configs]},
            "weighting": "+".join(c.get("weighting", "none") for c in member_configs),
            "features": features, "feature_set": "union of members"}


def fit_config(task: str, config: dict, train: pd.DataFrame):
    """Fit a pipeline for {family, params, weighting, features} on one training frame."""
    if config["family"] == ENSEMBLE_NAME:
        return AveragingEnsemble(task, [fit_config(task, member, train) for member in config["members"]])
    pipeline = make_pipeline(build_estimator(task, config["family"], config["params"]), config["features"])
    y = target_values(task, train)
    weights = sample_weights(y, config.get("weighting", "none")) if task == "classification" else None
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    pipeline.fit(train[feature_columns(config["features"])], y, **fit_kwargs)
    return pipeline


def predict_risk_ids(pipeline: Pipeline, X: pd.DataFrame, high_threshold: float | None = None) -> np.ndarray:
    """Standard rule = most probable class.  With ``high_threshold`` (high-recall alert mode),
    a week is High whenever P(High) >= threshold, otherwise the more probable of Medium/Low."""
    proba = pipeline.predict_proba(X)
    return ids_from_proba(proba, high_threshold)


def ids_from_proba(proba: np.ndarray, high_threshold: float | None = None) -> np.ndarray:
    if high_threshold is None:
        return proba.argmax(axis=1).astype(int)
    other = np.where(proba[:, LABEL_TO_ID["Medium"]] >= proba[:, LABEL_TO_ID["Low"]],
                     LABEL_TO_ID["Medium"], LABEL_TO_ID["Low"])
    return np.where(proba[:, HIGH_ID] >= high_threshold, HIGH_ID, other).astype(int)


def predict_values(task: str, pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if task == "classification":
        return pipeline.predict(X).astype(int)
    return np.clip(pipeline.predict(X), 0, None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_scores(y_true, y_pred, prefix: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(3)), zero_division=0)
    scores = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_precision": float(np.mean(precision)),
        f"{prefix}_macro_recall": float(np.mean(recall)),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, labels=list(range(3)), average="macro", zero_division=0),
    }
    for label, i in LABEL_TO_ID.items():
        scores[f"{prefix}_{label.lower()}_recall"] = float(recall[i])
    for label, i in LABEL_TO_ID.items():
        scores[f"{prefix}_{label.lower()}_precision"] = float(precision[i])
    return scores


def regression_scores(y_true, y_pred, prefix: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mape_percent": mape,
    }


def score(task: str, y_true, y_pred, prefix: str) -> dict:
    return (classification_scores if task == "classification" else regression_scores)(y_true, y_pred, prefix)


def selection_metric(task: str) -> tuple[str, bool]:
    """(metric suffix, higher_is_better) used for every model choice."""
    return ("macro_f1", True) if task == "classification" else ("mae", False)


def _better(task: str, a: float, b: float) -> bool:
    _, higher = selection_metric(task)
    return a > b + 1e-12 if higher else a < b - 1e-12


# ---------------------------------------------------------------------------
# Cross-validation, search and feature selection
# ---------------------------------------------------------------------------
def cross_validate(task: str, config: dict, folds) -> dict:
    """Mean/std of fold scores plus out-of-fold predictions (and probabilities)."""
    fold_scores, oof = [], []
    for k, (train, val) in enumerate(folds):
        pipeline = fit_config(task, config, train)
        X_val = val[feature_columns(config["features"])]
        y_val = target_values(task, val)
        if task == "classification":
            proba = pipeline.predict_proba(X_val)
            pred = proba.argmax(axis=1)
            frame = pd.DataFrame(proba, columns=[f"p_{l.lower()}" for l in RISK_LABELS])
        else:
            pred = np.clip(pipeline.predict(X_val), 0, None)
            frame = pd.DataFrame({"prediction": pred})
        frame["y_true"], frame["fold"] = y_val, k
        frame["blood_type"] = val.blood_type.to_numpy()
        frame["target_date"] = val.target_date.to_numpy()
        oof.append(frame)
        fold_scores.append(score(task, y_val, pred, "cv"))
    table = pd.DataFrame(fold_scores)
    result = table.mean().to_dict()
    metric, _ = selection_metric(task)
    result[f"cv_{metric}_std"] = float(table[f"cv_{metric}"].std(ddof=0))
    result["oof"] = pd.concat(oof, ignore_index=True)
    return result


def random_search(task: str, family: str, features: list[str], folds, n_iter: int,
                  log=print) -> pd.DataFrame:
    """Random search over SEARCH_SPACES; the original configuration is always candidate 0."""
    metric, higher = selection_metric(task)
    baseline = dict(BASELINE_PARAMS[task][family])
    if task == "classification":
        baseline["weighting"] = BASELINE_WEIGHTING[task]
    candidates = [baseline] + list(ParameterSampler(SEARCH_SPACES[task][family], n_iter=n_iter,
                                                    random_state=RANDOM_STATE))
    rows, seen = [], set()
    for i, params in enumerate(candidates):
        key = json.dumps(params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        config = {"family": family, "params": {k: v for k, v in params.items() if k != "weighting"},
                  "weighting": params.get("weighting", "none"), "features": features}
        cv = cross_validate(task, config, folds)
        rows.append({"family": family, "candidate": i, "is_original_settings": i == 0,
                     "params": json.dumps(config["params"], default=str), "weighting": config["weighting"],
                     **{k: v for k, v in cv.items() if k != "oof"}})
    table = pd.DataFrame(rows).sort_values(f"cv_{metric}", ascending=not higher).reset_index(drop=True)
    best = table.iloc[0]
    log(f"  {family:<13} best CV {metric} = {best[f'cv_{metric}']:.4f} "
        f"(original settings: {table.loc[table.is_original_settings, f'cv_{metric}'].iloc[0]:.4f}, "
        f"{len(table)} candidates)")
    return table


def importance_split(folds, holdout_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split of the data BEFORE the first CV validation block.

    Feature importance is measured here so that the later CV blocks, which judge whether
    a reduced feature set really helps, were not used to pick those features.
    """
    early = folds[0][0]
    dates = np.array(sorted(early.target_date.unique()))
    cut = dates[int(len(dates) * (1 - holdout_fraction)) - 1]
    return early.loc[early.target_date <= cut], early.loc[early.target_date > cut]


def permutation_importance_table(task: str, config: dict, train: pd.DataFrame, holdout: pd.DataFrame,
                                 n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance: drop in held-out score when one feature is shuffled."""
    scoring = "f1_macro" if task == "classification" else "neg_mean_absolute_error"
    columns = feature_columns(config["features"])
    pipeline = fit_config(task, config, train)
    imp = permutation_importance(pipeline, holdout[columns], target_values(task, holdout), scoring=scoring,
                                 n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": columns, "importance_mean": imp.importances_mean,
                          "importance_std": imp.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def tune_family(task: str, family: str, folds, n_iter: int, log=print) -> dict:
    """Tune hyper-parameters, then choose the feature set (original / extended / all / selected, and
    all_external when the optional external columns are present) by time-series CV."""
    metric, _ = selection_metric(task)
    search = random_search(task, family, ALL_NUMERICAL_FEATURES, folds, n_iter, log)
    best = search.iloc[0]
    params = json.loads(best["params"])
    base = {"family": family, "params": params, "weighting": best["weighting"]}

    imp_train, imp_holdout = importance_split(folds)
    importance = permutation_importance_table(task, {**base, "features": ALL_NUMERICAL_FEATURES},
                                              imp_train, imp_holdout)
    numeric_imp = importance[importance.feature != "blood_type"]
    keep = numeric_imp.loc[numeric_imp.importance_mean > 0, "feature"].tolist()
    if len(keep) < 10:  # never shrink to a handful of features
        keep = numeric_imp.feature.head(10).tolist()
    selected = [f for f in ALL_NUMERICAL_FEATURES if f in keep]

    feature_trials = {"all": (ALL_NUMERICAL_FEATURES, {k: v for k, v in best.items()})}
    candidates = [("original", FEATURE_SETS["original"]), ("extended", FEATURE_SETS["extended"]), ("selected", selected)]
    if has_external_features(folds[-1][0]):   # optional holiday / school-break / COVID columns are in the data
        candidates.append(("all_external", FEATURE_SETS["all_external"]))
    for name, feats in candidates:
        cv = cross_validate(task, {**base, "features": feats}, folds)
        feature_trials[name] = (feats, cv)
    chosen_name = "all"
    for name, _ in candidates:
        if _better(task, feature_trials[name][1][f"cv_{metric}"], feature_trials[chosen_name][1][f"cv_{metric}"]):
            chosen_name = name
    feats = feature_trials[chosen_name][0]
    trials = pd.DataFrame([{"family": family, "feature_set": n, "n_features": len(f),
                            f"cv_{metric}": c[f"cv_{metric}"]} for n, (f, c) in feature_trials.items()])
    log(f"  {family:<13} feature set chosen: {chosen_name} ({len(feats)} numeric features)")
    importance.insert(0, "family", family)
    return {"config": {**base, "features": feats, "feature_set": chosen_name},
            "search": search, "importance": importance, "feature_trials": trials}


# ---------------------------------------------------------------------------
# Walk-forward (periodic retraining) evaluation
# ---------------------------------------------------------------------------
def walk_forward_predict(task: str, config: dict, history: pd.DataFrame, period: pd.DataFrame,
                         step: int = WALK_FORWARD_STEP) -> dict:
    """Predict ``period`` while retraining every ``step`` observation dates.

    To predict from observation date t, the model may only learn from rows whose OUTCOME
    (target_date) is already known at t, i.e. target_date <= t.  ``history`` must contain
    every labeled row (earlier periods and ``period`` itself); rows after t are filtered out.
    """
    columns = feature_columns(config["features"])
    origins = np.array(sorted(period.date.unique()))
    pred = np.empty(len(period), dtype=float if task == "regression" else int)
    proba = np.zeros((len(period), len(RISK_LABELS))) if task == "classification" else None
    refits, position = 0, np.arange(len(period))
    for start in range(0, len(origins), max(1, step)):
        block = origins[start:start + step]
        cutoff = block[0]
        train = history.loc[history.target_date <= cutoff]
        model = fit_config(task, config, train)
        refits += 1
        mask = period.date.isin(block).to_numpy()
        X = period.loc[mask, columns]
        if task == "classification":
            p = model.predict_proba(X)
            proba[position[mask]] = p
            pred[position[mask]] = p.argmax(axis=1)
        else:
            pred[position[mask]] = np.clip(model.predict(X), 0, None)
    return {"pred": pred, "proba": proba, "refits": refits}


# ---------------------------------------------------------------------------
# Evaluation of one configuration across train / CV / validation / test
# ---------------------------------------------------------------------------
def evaluate_config(task: str, config: dict, parts: dict, folds, protocol: str = EVALUATION_PROTOCOL) -> dict:
    """Train/validation/CV/test scores for a configuration.

    The test period is scored under both protocols: ``static`` (one model fitted on train +
    validation) and ``walk_forward`` (retrained every WALK_FORWARD_STEP weeks).  ``protocol``
    decides which one is reported as ``test_*``; the other is ``test_static_*`` / ``test_walk_forward_*``.
    Validation is also scored with walk-forward (``validation_wf_*``) so the effect of periodic
    retraining can be judged WITHOUT looking at the test period.
    """
    columns = feature_columns(config["features"])
    fit_train = fit_config(task, config, parts["train"])
    scores = {}
    scores.update(score(task, target_values(task, parts["train"]),
                        predict_values(task, fit_train, parts["train"][columns]), "train"))
    scores.update(score(task, target_values(task, parts["validation"]),
                        predict_values(task, fit_train, parts["validation"][columns]), "validation"))
    cv = cross_validate(task, config, folds)
    scores.update({k: v for k, v in cv.items() if k != "oof"})
    development = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    everything = pd.concat([development, parts["test"]], ignore_index=True)
    y_val, y_test = target_values(task, parts["validation"]), target_values(task, parts["test"])

    wf_val = walk_forward_predict(task, config, development, parts["validation"])
    scores.update(score(task, y_val, wf_val["pred"], "validation_wf"))

    final = fit_config(task, config, development)
    static = {"pred": predict_values(task, final, parts["test"][columns]),
              "proba": final.predict_proba(parts["test"][columns]) if task == "classification" else None}
    walk = walk_forward_predict(task, config, everything, parts["test"])
    scores.update(score(task, y_test, static["pred"], "test_static"))
    scores.update(score(task, y_test, walk["pred"], "test_walk_forward"))
    primary = walk if protocol == "walk_forward" else static
    scores.update(score(task, y_test, primary["pred"], "test"))
    scores["test_protocol"] = protocol
    scores["walk_forward_refits"] = walk["refits"]
    return {"scores": scores, "pipeline": final, "test_pred": primary["pred"], "test_proba": primary["proba"],
            "static_test_pred": static["pred"], "static_test_proba": static["proba"], "oof": cv["oof"],
            "fit_through_date": str(pd.Timestamp(development.target_date.max()).date())}


def naive_scores(task: str, parts: dict, folds) -> dict:
    """Persistence baseline: next week equals this week (risk or demand)."""
    def predict(df):
        if task == "classification":
            return df.observed_risk.map(LABEL_TO_ID).fillna(LABEL_TO_ID["Medium"]).astype(int).to_numpy()
        return df.demand.fillna(df.demand_rolling4).fillna(0).to_numpy()
    scores = {}
    for name in ("train", "validation", "test"):
        scores.update(score(task, target_values(task, parts[name]), predict(parts[name]), name))
    for alias, name in (("validation_wf", "validation"), ("test_static", "test"), ("test_walk_forward", "test")):
        scores.update(score(task, target_values(task, parts[name]), predict(parts[name]), alias))
    fold_scores = [score(task, target_values(task, v), predict(v), "cv") for _, v in folds]
    scores.update(pd.DataFrame(fold_scores).mean().to_dict())
    return scores


def choose_high_recall_threshold(oof: pd.DataFrame, target: float = TARGET_HIGH_RECALL) -> tuple[float, pd.DataFrame]:
    """Pick the HIGHEST P(High) threshold whose out-of-fold High recall reaches ``target``.

    Chosen only from cross-validation predictions in the development period, never the test set.
    """
    proba = oof[["p_high", "p_medium", "p_low"]].to_numpy()
    y = oof.y_true.to_numpy()
    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.025), 3):
        pred = ids_from_proba(proba, t)
        s = classification_scores(y, pred, "cv")
        rows.append({"threshold": t, "cv_accuracy": s["cv_accuracy"], "cv_macro_f1": s["cv_macro_f1"],
                     "cv_high_recall": s["cv_high_recall"], "cv_high_precision": s["cv_high_precision"]})
    curve = pd.DataFrame(rows)
    reaching = curve[curve.cv_high_recall >= target]
    threshold = float(reaching.threshold.max()) if not reaching.empty else float(curve.threshold.min())
    return threshold, curve


def bootstrap_ci(task: str, frame: pd.DataFrame, n: int = BOOTSTRAP_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """95% intervals by resampling whole target dates (keeps the 4 blood types together)."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby("target_date")]
    stats = []
    for _ in range(n):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if task == "classification":
            s = classification_scores(sample.y_true, sample.y_pred, "test")
            stats.append({k: s[f"test_{k}"] for k in ("accuracy", "macro_f1", "high_recall")})
        else:
            s = regression_scores(sample.y_true, sample.y_pred, "test")
            stats.append({k: s[f"test_{k}"] for k in ("mae", "rmse", "mape_percent")})
    table = pd.DataFrame(stats)
    return pd.DataFrame({"metric": table.columns, "lower_95": table.quantile(0.025).to_numpy(),
                         "median": table.median().to_numpy(), "upper_95": table.quantile(0.975).to_numpy()})


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------
def _versions() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _write_json(path: Path, payload: dict) -> None:
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def _add_optimized(task: str, config: dict, name: str, parts: dict, folds, evaluated: dict,
                   optimized: list, comparison: list) -> None:
    """Evaluate one tuned configuration and append its rows to the result tables."""
    ev = evaluate_config(task, config, parts, folds)
    evaluated[name] = (config, ev)
    row = {"model": name, "family": name, "stage": "after optimization", **ev["scores"],
           "feature_set": config["feature_set"], "n_features": len(config["features"]),
           "weighting": config["weighting"], "params": json.dumps(config["params"], default=str),
           "train_rows": len(parts["train"]), "validation_rows": len(parts["validation"]),
           "test_rows": len(parts["test"])}
    if task == "classification":
        threshold, curve = choose_high_recall_threshold(ev["oof"])
        y_test = target_values(task, parts["test"])
        hr_pred = ids_from_proba(ev["test_proba"], threshold)
        row.update({"high_recall_threshold": threshold,
                    **classification_scores(y_test, hr_pred, "test_high_recall_mode")})
        ev["threshold"], ev["curve"], ev["hr_pred"] = threshold, curve, hr_pred
    optimized.append(row)
    comparison.append({**row, "model": f"{name} (optimized)"})


def train_task(task: str, df: pd.DataFrame | None = None, paths: ProjectPaths | None = None,
               n_iter: int = SEARCH_ITERATIONS, make_figures: bool = True, log=print) -> pd.DataFrame:
    """Tune, compare and save models for one task; return the optimized-model metrics table."""
    paths = (paths or DEFAULT_PATHS).ensure()
    df = load_processed(paths=paths) if df is None else df
    check_no_leakage(df, ALL_NUMERICAL_FEATURES)
    target_col = "next_risk" if task == "classification" else "next_demand"
    labeled = df.dropna(subset=[target_col, "target_date"]).copy()
    parts = chronological_splits(labeled)
    development = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    folds = time_series_folds(development)
    metric, higher = selection_metric(task)
    started = time.time()
    log(f"[{task}] train={len(parts['train'])} validation={len(parts['validation'])} "
        f"test={len(parts['test'])} rows; {len(folds)} time-series CV folds; {n_iter} search candidates/family")

    comparison, optimized, searches, importances, trials, evaluated = [], [], [], [], [], {}
    naive = naive_scores(task, parts, folds)
    comparison.append({"model": "Naive persistence", "family": "Naive", "stage": "naive baseline", **naive})

    for family in FAMILIES:
        base_config = {"family": family, "params": dict(BASELINE_PARAMS[task][family]),
                       "weighting": BASELINE_WEIGHTING[task], "features": ORIGINAL_NUMERICAL_FEATURES,
                       "feature_set": "original"}
        base_eval = evaluate_config(task, base_config, parts, folds)
        comparison.append({"model": f"{family} (original settings)", "family": family,
                           "stage": "before optimization", **base_eval["scores"]})

        tuned = tune_family(task, family, folds, n_iter, log)
        searches.append(tuned["search"]); importances.append(tuned["importance"]); trials.append(tuned["feature_trials"])
        _add_optimized(task, tuned["config"], family, parts, folds, evaluated, optimized, comparison)

    # Averaging ensemble of the two tuned tree ensembles (members fixed in advance, judged by CV like the rest).
    members = [evaluated[f][0] for f in ("Random Forest", "XGBoost") if f in evaluated]
    if len(members) == 2:
        _add_optimized(task, ensemble_config(members), ENSEMBLE_NAME, parts, folds, evaluated, optimized, comparison)
        log(f"  {ENSEMBLE_NAME:<13} CV {metric} = {optimized[-1][f'cv_{metric}']:.4f}")

    result = pd.DataFrame(optimized).sort_values(f"cv_{metric}", ascending=not higher).reset_index(drop=True)
    best_name = str(result.loc[0, "model"])
    result["selected"] = result.model.eq(best_name)
    comparison = pd.DataFrame(comparison)
    comparison["selected"] = comparison.model.eq(f"{best_name} (optimized)")
    best_config, best_eval = evaluated[best_name]

    # ---- Save tables --------------------------------------------------------
    prefix = "classification" if task == "classification" else "regression"
    result.to_csv(paths.class_metrics if task == "classification" else paths.reg_metrics, index=False)
    comparison.to_csv(paths.result(f"{prefix}_before_after.csv"), index=False)
    pd.concat(searches, ignore_index=True).to_csv(paths.result(f"{prefix}_hyperparameter_search.csv"), index=False)
    pd.concat(importances, ignore_index=True).to_csv(paths.result(f"{prefix}_feature_importance.csv"), index=False)
    pd.concat(trials, ignore_index=True).to_csv(paths.result(f"{prefix}_feature_set_trials.csv"), index=False)

    test = parts["test"]
    predictions = test[["date", "target_date", "blood_type"]].copy()
    if task == "classification":
        predictions["actual_risk"] = test.next_risk.to_numpy()
        predictions["next_coverage"] = test.next_coverage.to_numpy()
        for family, (_, ev) in evaluated.items():
            predictions[f"pred_{safe_name(family)}"] = [ID_TO_LABEL[i] for i in ev["test_pred"]]
        for label, i in LABEL_TO_ID.items():
            predictions[f"p_{label.lower()}_selected"] = best_eval["test_proba"][:, i]
        predictions["pred_selected_high_recall_mode"] = [ID_TO_LABEL[i] for i in best_eval["hr_pred"]]
        predictions["pred_selected_static"] = [ID_TO_LABEL[int(i)] for i in best_eval["static_test_pred"]]
        _save_classification_details(paths, test, evaluated, best_name)
        best_eval["curve"].to_csv(paths.result("high_risk_threshold_curve_cv.csv"), index=False)
        boot = pd.DataFrame({"target_date": test.target_date.to_numpy(), "y_true": target_values(task, test),
                             "y_pred": best_eval["test_pred"]})
    else:
        predictions["actual_demand"] = test.next_demand.to_numpy()
        for family, (_, ev) in evaluated.items():
            predictions[f"pred_{safe_name(family)}"] = ev["test_pred"]
        predictions["pred_selected_static"] = best_eval["static_test_pred"]
        per_type = []
        for family, (_, ev) in evaluated.items():
            for bt in BLOOD_TYPES:
                mask = (test.blood_type == bt).to_numpy()
                per_type.append({"model": family, "blood_type": bt, "test_rows": int(mask.sum()),
                                 **regression_scores(test.next_demand.to_numpy()[mask], ev["test_pred"][mask], "test")})
        pd.DataFrame(per_type).to_csv(paths.result("regression_per_blood_type.csv"), index=False)
        boot = pd.DataFrame({"target_date": test.target_date.to_numpy(), "y_true": target_values(task, test),
                             "y_pred": best_eval["test_pred"]})
    predictions.to_csv(paths.result(f"{prefix}_test_predictions.csv"), index=False)
    ci = bootstrap_ci(task, boot)
    ci.insert(0, "model", best_name)
    ci.to_csv(paths.result(f"{prefix}_test_confidence_intervals.csv"), index=False)

    # ---- Save model + metadata ---------------------------------------------
    # The saved model is refit on EVERY labeled week (train + validation + test), because the
    # next-week forecast should learn from all known outcomes.  Its scores are the walk-forward
    # test scores above, which mimic exactly this "retrain on everything known so far" practice.
    production = fit_config(task, best_config, labeled)
    member_params = ({m["family"]: {"params": m["params"], "weighting": m.get("weighting"),
                                    "features": m["features"]} for m in best_config["members"]}
                     if "members" in best_config else None)
    meta = {
        "task": task, "model_name": best_name, "params": best_config["params"],
        "weighting": best_config["weighting"], "feature_set": best_config["feature_set"],
        "numerical_features": best_config["features"], "features": feature_columns(best_config["features"]),
        "selection_metric": f"cv_{metric}", "cv_score": float(result.loc[0, f"cv_{metric}"]),
        "fit_through_date": str(pd.Timestamp(labeled.target_date.max()).date()),
        "evaluation_fit_through_date_static": best_eval["fit_through_date"],
        "test_protocol": EVALUATION_PROTOCOL, "walk_forward_step_weeks": WALK_FORWARD_STEP,
        "member_params": member_params,
        "data_range": [str(df.date.min().date()), str(df.date.max().date())],
        "source_data": str(paths.raw_data), "source_sha256": _file_sha256(paths.raw_data),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(), "versions": _versions(),
        "search_iterations": n_iter, "run": paths.name,
    }
    bundle = {"pipeline": production, **meta}
    if task == "classification":
        meta["labels"] = list(RISK_LABELS)
        meta["high_recall_threshold"] = best_eval["threshold"]
        bundle.update(labels=RISK_LABELS, high_recall_threshold=best_eval["threshold"])
    joblib.dump(bundle, paths.classifier if task == "classification" else paths.regressor)
    _write_json(paths.model_metadata, {task: meta})
    _write_json(paths.result("training_config.json"), {"general": training_config(), task: {
        "search_space": SEARCH_SPACES[task], "baseline_params": BASELINE_PARAMS[task],
        "baseline_weighting": BASELINE_WEIGHTING[task], "search_iterations": n_iter,
        "runtime_seconds": round(time.time() - started, 1)}})

    if make_figures:
        try:
            from src import plots
            plots.make_task_figures(task, paths, test, evaluated, best_name, comparison, result)
        except Exception as exc:  # figures must never break training
            log(f"  (figures skipped: {exc})")
    log(f"[{task}] selected by CV {metric}: {best_name}  ({time.time() - started:.0f}s)")
    return result


def _save_classification_details(paths: ProjectPaths, test: pd.DataFrame, evaluated: dict, best_name: str) -> None:
    y_true = target_values("classification", test)
    cm_rows, report_rows, type_rows = [], [], []
    for family, (_, ev) in evaluated.items():
        modes = [("standard", ev["test_pred"]), ("high_recall_mode", ev["hr_pred"])]
        for mode, pred in modes:
            cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
            for i, actual in enumerate(RISK_LABELS):
                for j, predicted in enumerate(RISK_LABELS):
                    cm_rows.append({"model": family, "mode": mode, "actual": actual, "predicted": predicted,
                                    "count": int(cm[i, j])})
            rep = classification_report(y_true, pred, labels=[0, 1, 2], target_names=list(RISK_LABELS),
                                        output_dict=True, zero_division=0)
            for cls, values in rep.items():
                if isinstance(values, dict):
                    report_rows.append({"model": family, "mode": mode, "class": cls, **values})
            for bt in BLOOD_TYPES:
                mask = (test.blood_type == bt).to_numpy()
                s = classification_scores(y_true[mask], pred[mask], "test")
                type_rows.append({"model": family, "mode": mode, "blood_type": bt, "test_rows": int(mask.sum()),
                                  "actual_high_weeks": int((y_true[mask] == HIGH_ID).sum()), **s})
    pd.DataFrame(cm_rows).to_csv(paths.result("classification_confusion_matrices.csv"), index=False)
    pd.DataFrame(report_rows).to_csv(paths.result("classification_per_class_report.csv"), index=False)
    pd.DataFrame(type_rows).to_csv(paths.result("classification_per_blood_type.csv"), index=False)


def training_config() -> dict:
    return {"random_state": RANDOM_STATE, "n_jobs": N_JOBS, "cv_splits": CV_SPLITS,
            "cv_min_train_fraction": CV_MIN_TRAIN_FRACTION, "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION, "target_gap_days": TARGET_GAP_DAYS,
            "risk_thresholds": {"High": f"coverage < {HIGH_RISK_COVERAGE}",
                                "Medium": f"{HIGH_RISK_COVERAGE} <= coverage < {LOW_RISK_COVERAGE}",
                                "Low": f"coverage >= {LOW_RISK_COVERAGE}"},
            "proposal_targets": {"accuracy": TARGET_ACCURACY, "high_recall": TARGET_HIGH_RECALL},
            "weighting_strategies": list(WEIGHTING_STRATEGIES), "feature_sets": list(FEATURE_SETS) + ["selected"],
            "evaluation_protocol": EVALUATION_PROTOCOL, "walk_forward_step_weeks": WALK_FORWARD_STEP,
            "ensemble": ENSEMBLE_NAME, "v2_features": V2_NUMERICAL_FEATURES}


def train_classification(df: pd.DataFrame | None = None, paths: ProjectPaths | None = None,
                         n_iter: int = SEARCH_ITERATIONS, make_figures: bool = True, log=print) -> pd.DataFrame:
    """Choose best classifier by time-series CV macro-F1; never choose using test."""
    return train_task("classification", df, paths, n_iter, make_figures, log)


def train_regression(df: pd.DataFrame | None = None, paths: ProjectPaths | None = None,
                     n_iter: int = SEARCH_ITERATIONS, make_figures: bool = True, log=print) -> pd.DataFrame:
    """Choose best regressor by time-series CV MAE; do not use test for selection."""
    return train_task("regression", df, paths, n_iter, make_figures, log)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_models(paths: ProjectPaths | None = None) -> tuple[dict, dict]:
    paths = paths or DEFAULT_PATHS
    if not paths.classifier.exists() or not paths.regressor.exists():
        raise FileNotFoundError("Saved models missing. Run: python run_pipeline.py")
    return joblib.load(paths.classifier), joblib.load(paths.regressor)


def _model_inputs(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(bundle.get("features", FEATURE_COLUMNS))
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Data is missing model features {missing}. Rebuild features with src.data_pipeline.")
    return frame[columns]


def predict_next_week(df: pd.DataFrame | None = None, paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Predict from LAST observed week; do not treat forecasts as observed outcomes.

    ``paths`` selects which trained models to use; ``df`` may be any processed dataset.
    """
    paths = paths or DEFAULT_PATHS
    df = load_processed(paths=paths) if df is None else df
    classifier, regressor = load_models(paths)
    latest = df.sort_values("date").groupby("blood_type", as_index=False).tail(1).copy()
    latest = latest.set_index("blood_type").reindex(list(BLOOD_TYPES)).dropna(subset=["date"]).reset_index()
    X_class = _model_inputs(classifier, latest)
    probability = classifier["pipeline"].predict_proba(X_class)
    risk_ids = probability.argmax(axis=1)
    threshold = classifier.get("high_recall_threshold")
    hr_ids = ids_from_proba(probability, threshold) if threshold is not None else risk_ids
    demand = regressor["pipeline"].predict(_model_inputs(regressor, latest))
    outcome = latest[["blood_type", "date", "coverage", "collection_efficiency", "observed_risk", "demand", "supply"]].copy()
    outcome = outcome.rename(columns={"date": "last_observed_week", "coverage": "observed_coverage",
                                      "collection_efficiency": "observed_efficiency", "demand": "observed_demand",
                                      "supply": "observed_supply"})
    outcome["forecast_week"] = outcome.last_observed_week + pd.Timedelta(days=7)
    outcome["predicted_risk"] = [ID_TO_LABEL[int(i)] for i in risk_ids]
    outcome["risk_confidence"] = probability.max(axis=1)
    for label, i in LABEL_TO_ID.items():
        outcome[f"p_{label.lower()}"] = probability[:, i]
    outcome["predicted_risk_high_recall_mode"] = [ID_TO_LABEL[int(i)] for i in hr_ids]
    outcome["high_recall_threshold"] = threshold
    outcome["predicted_demand"] = np.maximum(demand, 0)
    outcome["risk_alert"] = outcome.predicted_risk.map(ALERTS)
    outcome["risk_alert_high_recall_mode"] = outcome.predicted_risk_high_recall_mode.map(ALERTS)
    outcome["classifier"] = classifier.get("model_name")
    outcome["regressor"] = regressor.get("model_name")
    return outcome


def evaluate_saved_models(processed: pd.DataFrame, paths: ProjectPaths | None = None) -> dict[str, pd.DataFrame]:
    """Score already-trained models on every labeled row of a (new) dataset.  Nothing is refit."""
    classifier, regressor = load_models(paths)
    out = {}
    labeled = processed.dropna(subset=["next_risk", "target_date"])
    if not labeled.empty:
        y = target_values("classification", labeled)
        proba = classifier["pipeline"].predict_proba(_model_inputs(classifier, labeled))
        rows = []
        for mode, thr in (("standard", None), ("high_recall_mode", classifier.get("high_recall_threshold"))):
            if mode == "high_recall_mode" and thr is None:
                continue
            pred = ids_from_proba(proba, thr)
            rows.append({"mode": mode, "rows": len(labeled), **classification_scores(y, pred, "external")})
            for bt in BLOOD_TYPES:
                m = (labeled.blood_type == bt).to_numpy()
                if m.any():
                    rows.append({"mode": mode, "blood_type": bt, "rows": int(m.sum()),
                                 **classification_scores(y[m], pred[m], "external")})
        out["classification"] = pd.DataFrame(rows)
    labeled = processed.dropna(subset=["next_demand", "target_date"])
    if not labeled.empty:
        pred = np.clip(regressor["pipeline"].predict(_model_inputs(regressor, labeled)), 0, None)
        y = labeled.next_demand.to_numpy()
        rows = [{"blood_type": "All", "rows": len(labeled), **regression_scores(y, pred, "external")}]
        for bt in BLOOD_TYPES:
            m = (labeled.blood_type == bt).to_numpy()
            if m.any():
                rows.append({"blood_type": bt, "rows": int(m.sum()), **regression_scores(y[m], pred[m], "external")})
        out["regression"] = pd.DataFrame(rows)
    return out


def predict_external(source: CsvSource, model_paths: ProjectPaths | None = None, save: bool = True,
                     name: str = "external") -> dict:
    """Use EXISTING trained models on a new compatible CSV (no retraining, no overwriting).

    Outputs go to results/external/<name>_<timestamp>/ when ``save`` is True.
    """
    prepared = prepare_dataset(source)
    output = {"report": prepared.report, "processed": prepared.processed, "forecast": None,
              "evaluation": {}, "output_dir": None}
    if not prepared.report.ok:
        return output
    output["forecast"] = predict_next_week(prepared.processed, model_paths)
    output["evaluation"] = evaluate_saved_models(prepared.processed, model_paths)
    if save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = RESULTS_DIR / "external" / f"{safe_name(name) or 'external'}_{stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        prepared.report.as_frame().to_csv(folder / "validation_report.csv", index=False)
        output["forecast"].to_csv(folder / "next_week_forecast.csv", index=False)
        for task, table in output["evaluation"].items():
            table.to_csv(folder / f"{task}_evaluation_existing_model.csv", index=False)
        output["output_dir"] = folder
    return output


def main():
    parser = argparse.ArgumentParser(description="Blood Shortage ML training/prediction")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--train-all", action="store_true")
    commands.add_argument("--train-classification", action="store_true")
    commands.add_argument("--train-regression", action="store_true")
    commands.add_argument("--predict", action="store_true")
    commands.add_argument("--predict-external", metavar="CSV", help="use existing models on a new CSV")
    parser.add_argument("--quick", action="store_true", help="fewer search candidates (faster, less thorough)")
    args = parser.parse_args()
    if args.predict:
        print(predict_next_week().to_string(index=False))
        return
    if args.predict_external:
        out = predict_external(Path(args.predict_external), name=Path(args.predict_external).stem)
        print(out["report"].summary())
        if out["forecast"] is not None:
            print(out["forecast"].to_string(index=False))
            for task, table in out["evaluation"].items():
                print(f"\nExisting {task} model on this file:\n{table.round(4).to_string(index=False)}")
            print(f"\nSaved to {out['output_dir']}")
        return
    from src.config import QUICK_SEARCH_ITERATIONS
    n_iter = QUICK_SEARCH_ITERATIONS if args.quick else SEARCH_ITERATIONS
    df = generate_processed()  # Always rebuild features from the current raw CSV on training.
    if args.train_all or args.train_classification:
        print(train_classification(df, n_iter=n_iter).round(4).to_string(index=False))
    if args.train_all or args.train_regression:
        print(train_regression(df, n_iter=n_iter).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
