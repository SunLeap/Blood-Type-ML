"""Compare ML models, save best models, then forecast risk and demand.

CMD: python -m src.prediction --train-all
CMD: python -m src.prediction --train-classification
CMD: python -m src.prediction --train-regression
CMD: python -m src.prediction --predict
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score, recall_score,
    mean_absolute_error, mean_squared_error,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing import ROOT, BLOOD_TYPES
from src.feature_engineering import FEATURE_COLUMNS, NUMERICAL_FEATURES, RISK_LABELS, load_processed, generate_processed, risk_from_coverage

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "best_regressor.joblib"
CLASS_METRICS_PATH = RESULTS_DIR / "classification_metrics.csv"
REG_METRICS_PATH = RESULTS_DIR / "regression_metrics.csv"
LABEL_TO_ID = {label: i for i, label in enumerate(RISK_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def make_pipeline(model) -> Pipeline:
    """Fit imputers and one-hot encoder on training data ONLY."""
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", add_indicator=False), NUMERICAL_FEATURES),
            ("blood_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["blood_type"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def chronological_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by prediction TARGET dates, keeping all blood types in the same period.

    70% earliest target weeks = train; next 15% = validation; final 15% = test.
    We never shuffle or use a random train/test split.
    """
    dates = np.array(sorted(df.target_date.dropna().unique()))
    if len(dates) < 30:
        raise ValueError("Need at least 30 unique target dates to make meaningful temporal splits.")
    a = max(1, int(len(dates) * .70))
    b = max(a + 1, int(len(dates) * .85))
    first_end = dates[a - 1]
    second_end = dates[b - 1]
    parts = {
        "train": df.loc[df.target_date <= first_end].copy(),
        "validation": df.loc[(df.target_date > first_end) & (df.target_date <= second_end)].copy(),
        "test": df.loc[df.target_date > second_end].copy(),
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("Empty temporal partition. Examine your dataset.")
    return parts


def classification_models() -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=9, min_samples_leaf=4, random_state=42, n_jobs=2),
        "XGBoost": XGBClassifier(n_estimators=130, max_depth=4, learning_rate=0.06, subsample=0.9,
                                 colsample_bytree=0.9, objective="multi:softprob", num_class=3,
                                 eval_metric="mlogloss", random_state=42, n_jobs=2),
    }


def regression_models() -> dict:
    return {
        "Decision Tree": DecisionTreeRegressor(max_depth=7, min_samples_leaf=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=3,
                                               random_state=42, n_jobs=2),
        "XGBoost": XGBRegressor(n_estimators=130, max_depth=4, learning_rate=0.06, subsample=0.9,
                                colsample_bytree=0.9, objective="reg:squarederror", random_state=42, n_jobs=2),
    }


def classification_scores(y_true, y_pred, prefix: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(3)), zero_division=0
    )
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_precision": float(np.mean(precision)),
        f"{prefix}_macro_recall": float(np.mean(recall)),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, labels=list(range(3)), average="macro", zero_division=0),
        f"{prefix}_high_recall": recall[LABEL_TO_ID["High"]],
        f"{prefix}_medium_recall": recall[LABEL_TO_ID["Medium"]],
        f"{prefix}_low_recall": recall[LABEL_TO_ID["Low"]],
    }


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


def train_classification(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Choose best classifier with validation macro-F1; never choose using test."""
    df = load_processed() if df is None else df
    labeled = df.dropna(subset=["next_risk", "target_date"]).copy()
    parts = chronological_splits(labeled)
    fit_train = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    y_train = parts["train"].next_risk.map(LABEL_TO_ID).astype(int)
    y_val = parts["validation"].next_risk.map(LABEL_TO_ID).astype(int)
    y_test = parts["test"].next_risk.map(LABEL_TO_ID).astype(int)
    y_refit = fit_train.next_risk.map(LABEL_TO_ID).astype(int)
    rows = []
    fitted = {}
    for name, model in classification_models().items():
        pipeline = make_pipeline(model)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(parts["train"][FEATURE_COLUMNS], y_train, model__sample_weight=sample_weight)
        val_pred = pipeline.predict(parts["validation"][FEATURE_COLUMNS])
        scores = classification_scores(y_val, val_pred, "validation")
        # Refit on train + validation, then make a single final chronological test evaluation.
        refit_pipeline = make_pipeline(classification_models()[name])
        refit_pipeline.fit(fit_train[FEATURE_COLUMNS], y_refit,
                           model__sample_weight=compute_sample_weight(class_weight="balanced", y=y_refit))
        test_pred = refit_pipeline.predict(parts["test"][FEATURE_COLUMNS])
        scores.update(classification_scores(y_test, test_pred, "test"))
        rows.append({"model": name, **scores, "train_rows": len(parts["train"]),
                     "validation_rows": len(parts["validation"]), "test_rows": len(parts["test"])})
        fitted[name] = refit_pipeline
    result = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False).reset_index(drop=True)
    best_name = str(result.loc[0, "model"])
    result["selected"] = result.model.eq(best_name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(CLASS_METRICS_PATH, index=False)
    joblib.dump({"pipeline": fitted[best_name], "features": FEATURE_COLUMNS,
                 "model_name": best_name, "labels": RISK_LABELS, "fit_through_date": str(fit_train.target_date.max().date()),
                 "trained_at_utc": datetime.now(timezone.utc).isoformat()}, CLASSIFIER_PATH)
    print(f"Classifier selected by validation macro-F1: {best_name}")
    return result


def train_regression(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Choose best regressor with validation MAE; do not use test for selection."""
    df = load_processed() if df is None else df
    labeled = df.dropna(subset=["next_demand", "target_date"]).copy()
    parts = chronological_splits(labeled)
    fit_train = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    rows = []
    fitted = {}
    for name, model in regression_models().items():
        pipeline = make_pipeline(model)
        pipeline.fit(parts["train"][FEATURE_COLUMNS], parts["train"].next_demand)
        val_pred = np.clip(pipeline.predict(parts["validation"][FEATURE_COLUMNS]), 0, None)
        scores = regression_scores(parts["validation"].next_demand, val_pred, "validation")
        refit_pipeline = make_pipeline(regression_models()[name])
        refit_pipeline.fit(fit_train[FEATURE_COLUMNS], fit_train.next_demand)
        test_pred = np.clip(refit_pipeline.predict(parts["test"][FEATURE_COLUMNS]), 0, None)
        scores.update(regression_scores(parts["test"].next_demand, test_pred, "test"))
        rows.append({"model": name, **scores, "train_rows": len(parts["train"]),
                     "validation_rows": len(parts["validation"]), "test_rows": len(parts["test"])})
        fitted[name] = refit_pipeline
    result = pd.DataFrame(rows).sort_values("validation_mae", ascending=True).reset_index(drop=True)
    best_name = str(result.loc[0, "model"])
    result["selected"] = result.model.eq(best_name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REG_METRICS_PATH, index=False)
    joblib.dump({"pipeline": fitted[best_name], "features": FEATURE_COLUMNS, "model_name": best_name,
                 "fit_through_date": str(fit_train.target_date.max().date()),
                 "trained_at_utc": datetime.now(timezone.utc).isoformat()}, REGRESSOR_PATH)
    print(f"Regressor selected by validation MAE: {best_name}")
    return result


def predict_next_week(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predict from LAST observed week; do not treat forecasts as observed outcomes."""
    df = load_processed() if df is None else df
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        raise FileNotFoundError("Saved models missing. Run: python -m src.prediction --train-all")
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    latest = df.sort_values("date").groupby("blood_type", as_index=False).tail(1).copy()
    latest = latest.set_index("blood_type").loc[list(BLOOD_TYPES)].reset_index()
    features = latest[FEATURE_COLUMNS]
    risk_ids = classifier["pipeline"].predict(features).astype(int)
    probability = classifier["pipeline"].predict_proba(features)
    demand = regressor["pipeline"].predict(features)
    outcome = latest[["blood_type", "date", "coverage", "collection_efficiency", "observed_risk", "demand", "supply"]].copy()
    outcome = outcome.rename(columns={"date": "last_observed_week", "coverage": "observed_coverage",
                                      "collection_efficiency": "observed_efficiency", "demand": "observed_demand", "supply": "observed_supply"})
    outcome["forecast_week"] = outcome.last_observed_week + pd.Timedelta(days=7)
    outcome["predicted_risk"] = [ID_TO_LABEL[i] for i in risk_ids]
    outcome["risk_confidence"] = probability.max(axis=1)
    outcome["predicted_demand"] = np.maximum(demand, 0)
    outcome["risk_alert"] = outcome.predicted_risk.map({"High": "WARNING", "Medium": "NOTICE", "Low": "NONE"})
    return outcome


def main():
    parser = argparse.ArgumentParser(description="Blood Shortage ML training/prediction")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--train-all", action="store_true")
    commands.add_argument("--train-classification", action="store_true")
    commands.add_argument("--train-regression", action="store_true")
    commands.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    if args.predict:
        print(predict_next_week().to_string(index=False))
        return
    df = generate_processed()  # Always rebuild features from the current raw CSV on training.
    if args.train_all or args.train_classification:
        print(train_classification(df).round(4).to_string(index=False))
    if args.train_all or args.train_regression:
        print(train_regression(df).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
