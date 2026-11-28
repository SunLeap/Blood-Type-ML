import json

import numpy as np
import pandas as pd
import pytest

import src.prediction as prediction
from src.config import DEFAULT_PATHS, ProjectPaths
from src.data_pipeline import engineer_features, load_and_clean
from src.prediction import (
    HIGH_ID, LABEL_TO_ID, chronological_splits, ids_from_proba, predict_external, predict_next_week,
    sample_weights, time_series_folds,
)


@pytest.fixture(scope="module")
def labeled():
    df = engineer_features(load_and_clean())
    return df.dropna(subset=["next_risk", "target_date"])


def test_chronological_split_is_ordered_and_keeps_dates_together(labeled):
    parts = chronological_splits(labeled)
    assert parts["train"].target_date.max() < parts["validation"].target_date.min()
    assert parts["validation"].target_date.max() < parts["test"].target_date.min()
    assert sum(len(p) for p in parts.values()) == len(labeled)
    for p in parts.values():
        assert p.groupby("target_date").blood_type.nunique().eq(4).all()


def test_time_series_folds_never_use_test_or_future(labeled):
    parts = chronological_splits(labeled)
    dev = pd.concat([parts["train"], parts["validation"]])
    folds = time_series_folds(dev)
    assert len(folds) == 4
    for train, val in folds:
        assert train.target_date.max() < val.target_date.min()
        assert val.target_date.max() <= dev.target_date.max() < parts["test"].target_date.min()
    assert folds[0][0].shape[0] < folds[-1][0].shape[0]  # expanding window


def test_sample_weight_strategies():
    y = np.array([0, 1, 1, 1, 2, 2])
    assert sample_weights(y, "none") is None
    balanced = sample_weights(y, "balanced")
    assert balanced[0] > balanced[1]
    assert np.allclose(sample_weights(y, "sqrt_balanced"), np.sqrt(balanced))
    boosted = sample_weights(y, "balanced_high_boost")
    assert np.isclose(boosted[0], balanced[0] * 1.5) and np.isclose(boosted[1], balanced[1])
    with pytest.raises(ValueError):
        sample_weights(y, "unknown")


def test_high_recall_threshold_rule():
    proba = np.array([[0.30, 0.50, 0.20], [0.10, 0.20, 0.70], [0.60, 0.30, 0.10]])
    assert list(ids_from_proba(proba)) == [LABEL_TO_ID["Medium"], LABEL_TO_ID["Low"], HIGH_ID]
    assert list(ids_from_proba(proba, 0.25)) == [HIGH_ID, LABEL_TO_ID["Low"], HIGH_ID]


@pytest.fixture(scope="module")
def trained_paths(tmp_path_factory):
    """Train a tiny configuration into a temporary folder (primary models untouched)."""
    base = tmp_path_factory.mktemp("run")
    paths = ProjectPaths(raw_data=DEFAULT_PATHS.raw_data, data_dir=base / "data", models_dir=base / "models",
                         results_dir=base / "results", name="pytest").ensure()
    df = engineer_features(load_and_clean())
    before = {p: p.stat().st_mtime for p in (DEFAULT_PATHS.classifier, DEFAULT_PATHS.regressor) if p.exists()}
    mp = pytest.MonkeyPatch()
    mp.setattr(prediction, "FAMILIES", ("Decision Tree",))
    try:
        prediction.train_classification(df, paths, n_iter=1, make_figures=True, log=lambda *_: None)
        prediction.train_regression(df, paths, n_iter=1, make_figures=False, log=lambda *_: None)
    finally:
        mp.undo()
    after = {p: p.stat().st_mtime for p in before}
    assert before == after, "training into a separate folder must not touch the primary models"
    return paths


def test_training_saves_models_metrics_and_metadata(trained_paths):
    p = trained_paths
    for f in (p.classifier, p.regressor, p.class_metrics, p.reg_metrics, p.model_metadata,
              p.result("classification_before_after.csv"), p.result("classification_confusion_matrices.csv"),
              p.result("training_config.json"), p.figures_dir / "confusion_matrices_test.png"):
        assert f.exists(), f
    meta = json.loads(p.model_metadata.read_text())
    assert {"classification", "regression"} <= set(meta)
    assert 0 < meta["classification"]["high_recall_threshold"] < 1
    metrics = pd.read_csv(p.class_metrics)
    assert metrics.selected.sum() == 1
    for col in ("train_macro_f1", "cv_macro_f1", "validation_macro_f1", "validation_wf_macro_f1", "test_accuracy",
                "test_high_recall", "test_static_accuracy", "test_walk_forward_accuracy"):
        assert metrics[col].between(0, 1).all()
    stages = set(pd.read_csv(p.result("classification_before_after.csv")).stage)
    assert stages == {"naive baseline", "before optimization", "after optimization"}


def test_predict_next_week(trained_paths):
    forecast = predict_next_week(paths=trained_paths)
    assert list(forecast.blood_type) == ["A", "B", "O", "AB"]
    assert set(forecast.predicted_risk) <= {"High", "Medium", "Low"}
    assert (forecast.predicted_demand >= 0).all()
    assert np.allclose(forecast[["p_high", "p_medium", "p_low"]].sum(axis=1), 1, atol=1e-5)
    assert (forecast.forecast_week - forecast.last_observed_week).dt.days.eq(7).all()


def test_predict_external_uses_existing_models_without_writing(trained_paths, tmp_path):
    raw = pd.read_csv(DEFAULT_PATHS.raw_data, dtype="string").tail(120)
    csv = tmp_path / "new_data.csv"
    raw.to_csv(csv, index=False)
    model_mtime = trained_paths.classifier.stat().st_mtime
    out = predict_external(csv, model_paths=trained_paths, save=False)
    assert out["report"].ok and out["output_dir"] is None
    assert len(out["forecast"]) == 4
    assert "classification" in out["evaluation"] and "regression" in out["evaluation"]
    assert trained_paths.classifier.stat().st_mtime == model_mtime


def test_predict_external_rejects_incompatible_file(trained_paths, tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["1/1/2020"], "foo": [1]}).to_csv(csv, index=False)
    out = predict_external(csv, model_paths=trained_paths, save=False)
    assert not out["report"].ok and out["forecast"] is None


def test_missing_models_give_clear_error(tmp_path):
    empty = ProjectPaths(raw_data=DEFAULT_PATHS.raw_data, data_dir=tmp_path, models_dir=tmp_path / "m",
                         results_dir=tmp_path / "r")
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        prediction.load_models(empty)


def test_run_name_primary_is_rejected():
    with pytest.raises(ValueError):
        ProjectPaths.for_run("primary")


# ---------------------------------------------------------------------------
# Version 2: walk-forward evaluation and ensemble
# ---------------------------------------------------------------------------
def test_walk_forward_only_learns_from_known_outcomes(labeled, monkeypatch):
    parts = chronological_splits(labeled)
    seen = []
    real_fit = prediction.fit_config

    def spy(task, config, train):
        seen.append(train.target_date.max())
        return real_fit(task, config, train)

    monkeypatch.setattr(prediction, "fit_config", spy)
    config = {"family": "Decision Tree", "params": {"max_depth": 3}, "weighting": "none",
              "features": prediction.ORIGINAL_NUMERICAL_FEATURES}
    out = prediction.walk_forward_predict("classification", config, labeled, parts["test"], step=8)
    origins = np.array(sorted(parts["test"].date.unique()))
    cutoffs = origins[::8]
    assert out["refits"] == len(cutoffs) == len(seen)
    for latest_known, cutoff in zip(seen, cutoffs):
        assert latest_known <= cutoff  # never trained on an outcome after the forecast origin
    assert len(out["pred"]) == len(parts["test"]) and np.allclose(out["proba"].sum(axis=1), 1)


def test_ensemble_averages_member_probabilities(labeled):
    parts = chronological_splits(labeled)
    members = [{"family": f, "params": p, "weighting": "balanced", "features": prediction.ORIGINAL_NUMERICAL_FEATURES}
               for f, p in (("Random Forest", {"n_estimators": 20, "max_depth": 4}),
                            ("XGBoost", {"n_estimators": 20, "max_depth": 2}))]
    config = prediction.ensemble_config(members)
    model = prediction.fit_config("classification", config, parts["train"])
    X = parts["validation"][prediction.feature_columns(config["features"])]
    expected = np.mean([m.predict_proba(X) for m in model.members], axis=0)
    assert np.allclose(model.predict_proba(X), expected)
    assert np.array_equal(model.predict(X), expected.argmax(axis=1))


# ---------------------------------------------------------------------------
# Optional external weekly features (holidays, school breaks, COVID restrictions)
# ---------------------------------------------------------------------------
ENRICHED = DEFAULT_PATHS.data_dir / "Zhejiang_blood_donation_corrected_enriched.csv"


@pytest.mark.skipif(not ENRICHED.exists(), reason="enriched dataset not present")
def test_external_columns_reach_every_blood_type_row():
    from src.data_pipeline import EXTERNAL_NUMERICAL_FEATURES, has_external_features, prepare_dataset
    processed = prepare_dataset(ENRICHED).processed
    assert has_external_features(processed)
    per_date = processed.groupby("date")[EXTERNAL_NUMERICAL_FEATURES].nunique(dropna=False)
    assert (per_date <= 1).all().all()           # same weekly value for all four blood types
    assert not has_external_features(engineer_features(load_and_clean()))   # primary file: feature set not offered


@pytest.mark.skipif(not ENRICHED.exists(), reason="enriched dataset not present")
def test_external_feature_set_is_tried_only_when_present(tmp_path):
    from src.data_pipeline import prepare_dataset
    processed = prepare_dataset(ENRICHED).processed
    paths = ProjectPaths(raw_data=ENRICHED, data_dir=tmp_path / "data", models_dir=tmp_path / "models",
                         results_dir=tmp_path / "results", name="pytest_external").ensure()
    mp = pytest.MonkeyPatch()
    mp.setattr(prediction, "FAMILIES", ("Decision Tree",))
    try:
        prediction.train_classification(processed, paths, n_iter=1, make_figures=False, log=lambda *_: None)
    finally:
        mp.undo()
    trials = pd.read_csv(paths.result("classification_feature_set_trials.csv"))
    assert "all_external" in set(trials.feature_set)
