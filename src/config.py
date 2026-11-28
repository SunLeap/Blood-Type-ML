from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"  # separate experiments on external datasets live here

RAW_DATA = DATA_DIR / "Zhejiang_blood_donation.csv"  # primary research dataset (never modified)

# ---------------------------------------------------------------------------
# Research definitions (unchanged from the original project)
# ---------------------------------------------------------------------------
BLOOD_TYPES = ("A", "B", "O", "AB")
RISK_LABELS = ("High", "Medium", "Low")
HIGH_RISK_COVERAGE = 0.80   # coverage < 0.80          -> High
LOW_RISK_COVERAGE = 1.00    # 0.80 <= coverage < 1.00  -> Medium; >= 1.00 -> Low
TARGET_GAP_DAYS = (5, 9)    # next observation must be 5-9 days later to be a "next-week" label
TRAIN_FRACTION = 0.70       # earliest 70% of target dates
VALIDATION_FRACTION = 0.15  # next 15%; the final 15% is the untouched test period

# Proposal targets (research aims, not guarantees)
TARGET_ACCURACY = 0.80
TARGET_HIGH_RECALL = 0.75

# ---------------------------------------------------------------------------
# Reproducibility and tuning
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_JOBS = max(1, min(4, (os.cpu_count() or 2)))
CV_SPLITS = 4                 # expanding-window folds inside the development period
CV_MIN_TRAIN_FRACTION = 0.60  # first fold trains on the earliest 60% of development dates
SEARCH_ITERATIONS = 20        # random-search candidates per model family (plus the original config)
QUICK_SEARCH_ITERATIONS = 4   # used by --quick and the dashboard's "train on upload"
BOOTSTRAP_SAMPLES = 1000      # date-block bootstrap for test-score confidence intervals

# Version 2 evaluation protocol.  "walk_forward": during the validation/test period the model
# is retrained every WALK_FORWARD_STEP weeks on every outcome known at that point, as a blood
# bank would do in practice.  "static" (version 1): one model fitted before the period starts.
# Both are always computed; this setting only decides which one is reported as test_*.
EVALUATION_PROTOCOL = "walk_forward"
WALK_FORWARD_STEP = 4         # retrain every 4 weekly origins (about monthly); 1 = weekly
ENSEMBLE_NAME = "Ensemble (RF + XGBoost)"  # probability / prediction average of the two tuned models

# Spring Festival (Chinese New Year) dates: a calendar fact known in advance, used for
# a target-week holiday feature. Dates outside this table produce a missing value.
SPRING_FESTIVAL_DATES = (
    "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19",
    "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25", "2021-02-12",
    "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06",
    "2028-01-26", "2029-02-13", "2030-02-03",
)


@dataclass(frozen=True)
class ProjectPaths:
    """All output locations for one experiment.

    ``ProjectPaths.default()`` is the primary Zhejiang experiment.  ``ProjectPaths.for_run(name)``
    puts every generated file for an external dataset under ``runs/<name>/`` so the
    primary dataset, processed files and trained models are never overwritten.
    """

    raw_data: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path
    name: str = "primary"
    extra: dict = field(default_factory=dict, compare=False)

    # Derived file locations ------------------------------------------------
    @property
    def cleaned_data(self) -> Path:
        return self.data_dir / "cleaned_weekly.csv"

    @property
    def processed_data(self) -> Path:
        return self.data_dir / "processed_data.csv"

    @property
    def classifier(self) -> Path:
        return self.models_dir / "best_classifier.joblib"

    @property
    def regressor(self) -> Path:
        return self.models_dir / "best_regressor.joblib"

    @property
    def model_metadata(self) -> Path:
        return self.models_dir / "model_metadata.json"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def class_metrics(self) -> Path:
        return self.results_dir / "classification_metrics.csv"

    @property
    def reg_metrics(self) -> Path:
        return self.results_dir / "regression_metrics.csv"

    def result(self, filename: str) -> Path:
        return self.results_dir / filename

    def ensure(self) -> "ProjectPaths":
        for folder in (self.data_dir, self.models_dir, self.results_dir, self.figures_dir):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    # Factories ---------------------------------------------------------------
    @classmethod
    def default(cls) -> "ProjectPaths":
        return cls(raw_data=RAW_DATA, data_dir=DATA_DIR, models_dir=MODELS_DIR,
                   results_dir=RESULTS_DIR, name="primary")

    @classmethod
    def for_run(cls, run_name: str, raw_data: Path | None = None) -> "ProjectPaths":
        safe = safe_name(run_name)
        if safe in {"", "primary"}:
            raise ValueError("Choose a run name other than 'primary' for an external dataset.")
        base = RUNS_DIR / safe
        return cls(raw_data=Path(raw_data) if raw_data else base / "data" / "input.csv",
                   data_dir=base / "data", models_dir=base / "models",
                   results_dir=base / "results", name=safe)


def safe_name(text: str) -> str:
    """Turn any text into a Windows-safe folder name."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")[:60]


# Convenience aliases kept for older notebooks/scripts.
DEFAULT_PATHS = ProjectPaths.default()
CLASSIFIER_PATH = DEFAULT_PATHS.classifier
REGRESSOR_PATH = DEFAULT_PATHS.regressor
CLASS_METRICS_PATH = DEFAULT_PATHS.class_metrics
REG_METRICS_PATH = DEFAULT_PATHS.reg_metrics
PROCESSED_DATA = DEFAULT_PATHS.processed_data
CLEAN_DATA = DEFAULT_PATHS.cleaned_data
