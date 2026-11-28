from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

from src.config import (
    BLOOD_TYPES, DEFAULT_PATHS, HIGH_RISK_COVERAGE, LOW_RISK_COVERAGE, RAW_DATA, RISK_LABELS,
    SPRING_FESTIVAL_DATES, TARGET_GAP_DAYS, ProjectPaths,
)

CsvSource = Union[str, Path, IO[bytes], IO[str], pd.DataFrame]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["date"] + [f"{kind}_{bt}" for kind in ("collected", "demand", "supply") for bt in BLOOD_TYPES]
TOTAL_COLUMNS = ["collected_total", "demand_total", "supply_total"]
OPTIONAL_COLUMNS = TOTAL_COLUMNS + ["team_total", "Individual_donate", "platelet_collection",
                                    "platelet_collection_tx_units", "platelet_supply"] + [
    f"{kind}_{bt}" for kind in ("platelet_collection", "platelet_supply") for bt in BLOOD_TYPES
]
MIN_ROWS_FOR_PREDICTION = 5    # enough history for the 4-week lag
MIN_DATES_FOR_TRAINING = 60    # chronological 70/15/15 split + CV needs a reasonable history

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
ORIGINAL_NUMERICAL_FEATURES = [
    "collected", "supply", "demand", "platelet_collection", "platelet_supply",
    "coverage", "collection_efficiency", "collected_total", "demand_total",
    "supply_total", "team_total", "Individual_donate", "month", "quarter",
    "week_sin", "week_cos", "demand_growth", "supply_growth",
] + [
    f"{variable}_lag{lag}" for variable in ("demand", "supply", "collected", "coverage") for lag in (1, 2, 4)
] + [f"{variable}_rolling4" for variable in ("demand", "supply", "coverage", "collected")]

# New candidates added in the optimisation phase.  All use week t or earlier, or the
# calendar of the target week (date + 7 days), which is known when forecasting.
EXTENDED_NUMERICAL_FEATURES = [
    "coverage_rolling8", "coverage_rolling12", "coverage_std4", "coverage_ewm4",
    "collected_coverage", "collected_coverage_rolling4", "demand_vs_rolling12",
    "demand_share", "supply_share",
    "target_week_sin", "target_week_cos", "target_days_to_spring_festival", "target_spring_festival_window",
]
# Version 2 candidates.  Still week t or earlier:
# * stock proxy: running net flow (collected - supplied) over 4/13/26 weeks, expressed in
#   weeks of recent demand.  The data has no inventory column, so this is an approximation.
# * lagged collection relative to demand (blood collected now is often supplied next week).
# * cross-type context: every blood type's coverage this week, total coverage, the lowest
#   type coverage and how many types are below the High-risk cutoff (now and 4-week mean).
V2_NUMERICAL_FEATURES = [
    "net_flow_ratio", "stock_proxy4", "stock_proxy13", "stock_proxy26", "collected_coverage_lag1",
    "coverage_type_A", "coverage_type_B", "coverage_type_O", "coverage_type_AB",
    "total_coverage", "total_coverage_lag1", "min_type_coverage", "types_high_count",
    "types_high_count_rolling4",
]
ALL_NUMERICAL_FEATURES = ORIGINAL_NUMERICAL_FEATURES + EXTENDED_NUMERICAL_FEATURES + V2_NUMERICAL_FEATURES
FEATURE_SETS = {
    "original": ORIGINAL_NUMERICAL_FEATURES,
    "extended": ORIGINAL_NUMERICAL_FEATURES + EXTENDED_NUMERICAL_FEATURES,   # version 1 candidates
    "all": ALL_NUMERICAL_FEATURES,                                           # version 1 + version 2
}

# Optional EXTERNAL weekly columns (holidays, school breaks, COVID restrictions) produced by
# tools/build_weekly_features.py.  They are used only when the input CSV contains them, and only
# as an extra feature set that must win the time-series CV.  Calendar columns (incl. next_week_*)
# are known in advance; COVID stringency is the current week's observed value.  The quarterly
# GDP/health/population context is deliberately NOT included (too slow-moving, same-quarter GDP is
# published only after the quarter ends).
_HOLIDAYS = ("new_year", "spring_festival", "qingming", "labour_day", "dragon_boat", "mid_autumn", "national_day",
             "other_holiday")
_CALENDAR = ["holiday_days_off", "makeup_workdays", "total_days_off", "school_break_days_approx"] + \
            [f"{h}_days" for h in _HOLIDAYS]
EXTERNAL_NUMERICAL_FEATURES = _CALENDAR + [f"next_week_{c}" for c in _CALENDAR] + [
    "days_to_next_holiday", "days_since_last_holiday", "covid_stringency_mean", "covid_stringency_max"]
FEATURE_SETS["all_external"] = ALL_NUMERICAL_FEATURES + EXTERNAL_NUMERICAL_FEATURES


def has_external_features(df: pd.DataFrame, min_share: float = 0.5) -> bool:
    """True when the data carries the optional external columns for most rows."""
    present = [c for c in EXTERNAL_NUMERICAL_FEATURES if c in df.columns]
    return len(present) == len(EXTERNAL_NUMERICAL_FEATURES) and df[present].notna().all(axis=1).mean() >= min_share


# Backwards-compatible names used by the original notebooks.
NUMERICAL_FEATURES = ORIGINAL_NUMERICAL_FEATURES
FEATURE_COLUMNS = ["blood_type"] + ORIGINAL_NUMERICAL_FEATURES
TARGET_COLUMNS = ["target_date", "next_demand", "next_coverage", "next_risk"]


def feature_columns(numerical: list[str]) -> list[str]:
    return ["blood_type"] + list(numerical)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Human-readable result of checking a CSV against the expected schema."""

    rows: int = 0
    unique_dates: int = 0
    date_start: str | None = None
    date_end: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    converted_to_missing: dict[str, int] = field(default_factory=dict)
    negative_values: dict[str, int] = field(default_factory=dict)
    duplicate_dates: int = 0
    exact_duplicate_rows: int = 0
    unparsed_dates: int = 0
    irregular_gaps: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def can_train(self) -> bool:
        return self.ok and self.unique_dates >= MIN_DATES_FOR_TRAINING

    def as_frame(self) -> pd.DataFrame:
        rows = [("Rows", self.rows), ("Unique dates", self.unique_dates),
                ("Date range", f"{self.date_start} to {self.date_end}"),
                ("Unparsed dates (dropped)", self.unparsed_dates),
                ("Exact duplicate rows (dropped)", self.exact_duplicate_rows),
                ("Repeated dates (averaged)", self.duplicate_dates),
                ("Irregular gaps (no label created)", self.irregular_gaps),
                ("Non-numeric cells set to missing", sum(self.converted_to_missing.values())),
                ("Negative values set to missing", sum(self.negative_values.values())),
                ("Optional columns absent", len(self.missing_optional)),
                ("Usable for prediction", self.ok), ("Usable for training", self.can_train)]
        return pd.DataFrame(rows, columns=["Check", "Result"]).astype({"Result": "string"})

    def summary(self) -> str:
        lines = [f"Rows: {self.rows} | unique dates: {self.unique_dates} | range: {self.date_start} to {self.date_end}"]
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def read_raw_csv(source: CsvSource) -> pd.DataFrame:
    """Read a CSV path, uploaded file object or DataFrame with every column as text."""
    if isinstance(source, pd.DataFrame):
        df = source.astype("string")
    else:
        if isinstance(source, (str, Path)):
            if not Path(source).exists():
                raise FileNotFoundError(f"Dataset missing: {source}")
        elif hasattr(source, "seek"):
            source.seek(0)
        try:
            df = pd.read_csv(source, dtype="string", encoding="utf-8-sig")
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, dtype="string", encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    # Accept case differences (e.g. "Demand_A") by mapping to the canonical names.
    canonical = {c.lower(): c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
    df = df.rename(columns={c: canonical.get(c.lower(), c) for c in df.columns})
    return df


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%m/%d/%Y", errors="coerce")
    missing = parsed.isna() & text.notna()
    if missing.any():  # fall back to ISO dates such as 2024-06-17 for external files
        parsed.loc[missing] = pd.to_datetime(text[missing], format="ISO8601", errors="coerce")
    return parsed


def _to_number(values: pd.Series) -> pd.Series:
    text = (values.astype("string").str.replace("\u00a0", "", regex=False)
            .str.replace(",", "", regex=False).str.strip())
    return pd.to_numeric(text, errors="coerce").astype("float64")


def validate_raw(raw: pd.DataFrame) -> ValidationReport:
    """Check required columns, data types, dates and value ranges without changing data."""
    report = ValidationReport(rows=len(raw))
    absent = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if absent:
        report.errors.append(f"Missing required columns: {absent}")
        return report
    report.missing_optional = [c for c in OPTIONAL_COLUMNS if c not in raw.columns]
    if report.missing_optional:
        report.warnings.append("Optional columns absent (filled as missing or derived from per-type columns): "
                               + ", ".join(report.missing_optional))
    dates = _parse_dates(raw["date"])
    report.unparsed_dates = int(dates.isna().sum())
    if report.unparsed_dates:
        report.warnings.append(f"{report.unparsed_dates} rows have unreadable dates and will be dropped.")
    valid_dates = dates.dropna()
    report.unique_dates = int(valid_dates.nunique())
    if report.unique_dates:
        report.date_start, report.date_end = str(valid_dates.min().date()), str(valid_dates.max().date())
    report.duplicate_dates = int(valid_dates.duplicated().sum())
    report.exact_duplicate_rows = int(raw.duplicated().sum())
    gaps = pd.Series(sorted(valid_dates.unique())).diff().dt.days.dropna()
    report.irregular_gaps = int((~gaps.between(*TARGET_GAP_DAYS)).sum())

    for col in [c for c in raw.columns if c != "date" and (c in REQUIRED_COLUMNS or c in OPTIONAL_COLUMNS)]:
        original = raw[col].astype("string").str.strip()
        number = _to_number(raw[col])
        bad = int((number.isna() & original.notna() & original.ne("")).sum())
        neg = int((number < 0).sum())
        if bad:
            report.converted_to_missing[col] = bad
        if neg:
            report.negative_values[col] = neg
        if col in REQUIRED_COLUMNS and number.notna().sum() == 0:
            report.errors.append(f"Column '{col}' has no numeric values.")
    if report.converted_to_missing:
        report.warnings.append(f"Non-numeric cells (e.g. '#VALUE!') set to missing: {report.converted_to_missing}")
    if report.negative_values:
        report.warnings.append(f"Negative amounts set to missing: {report.negative_values}")
    if report.unique_dates < MIN_ROWS_FOR_PREDICTION:
        report.errors.append(f"At least {MIN_ROWS_FOR_PREDICTION} dated rows are needed (found {report.unique_dates}).")
    elif report.unique_dates < MIN_DATES_FOR_TRAINING:
        report.warnings.append(f"Only {report.unique_dates} dates: enough for prediction with an existing model, "
                               f"but training needs at least {MIN_DATES_FOR_TRAINING}.")
    return report


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean text exports, impossible negative values and duplicate dates.

    Duplicate *dates* are merged using the column means rather than summed because
    these entries appear to be conflicting measurements, not disjoint donations.
    This policy is a project assumption and should be reviewed with a data owner.
    Missing numeric inputs are left missing for train-only sklearn imputers.
    """
    absent = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if absent:
        raise ValueError(f"Missing expected columns: {sorted(absent)}")
    df = raw.copy()
    df["date"] = _parse_dates(df["date"])
    df = df.dropna(subset=["date"])
    for col in df.columns.drop("date"):
        number = _to_number(df[col])
        df[col] = number.mask(number < 0, np.nan)
    df = df.drop_duplicates()
    df = df.groupby("date", as_index=False, sort=True).mean(numeric_only=True)

    # External files may lack totals or platelet columns: derive or fill as missing.
    for kind in ("collected", "demand", "supply"):
        total = f"{kind}_total"
        if total not in df.columns:
            df[total] = df[[f"{kind}_{bt}" for bt in BLOOD_TYPES]].sum(axis=1, min_count=1)
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    ordered = REQUIRED_COLUMNS[:1] + [c for c in df.columns if c != "date"]
    return df[ordered].sort_values("date").reset_index(drop=True)


def load_and_clean(source: CsvSource = RAW_DATA) -> pd.DataFrame:
    """Read and clean a dataset (primary Zhejiang file by default)."""
    return clean_raw(read_raw_csv(source))


def save_cleaned(path: Path | None = None, source: CsvSource = RAW_DATA) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_PATHS.cleaned_data
    df = load_and_clean(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio; replace zero denominators with missing values."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def risk_from_coverage(coverage: float) -> str | None:
    """Project thresholds: HIGH <80%, MEDIUM 80%-<100%, LOW >=100%."""
    if pd.isna(coverage):
        return None
    if coverage < HIGH_RISK_COVERAGE:
        return "High"
    if coverage < LOW_RISK_COVERAGE:
        return "Medium"
    return "Low"


_SPRING_FESTIVALS = pd.to_datetime(list(SPRING_FESTIVAL_DATES))


def days_to_spring_festival(dates: pd.Series, clip: int = 60) -> pd.Series:
    """Signed days from each date to the nearest Spring Festival (negative = before)."""
    values = []
    for d in pd.to_datetime(dates):
        diffs = (d - _SPRING_FESTIVALS).days
        nearest = diffs[np.argmin(np.abs(diffs))]
        values.append(float(np.clip(nearest, -clip, clip)) if abs(nearest) <= 400 else np.nan)
    return pd.Series(values, index=dates.index)


def cross_type_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Same-week context shared by all four blood types (one row per date, aligned with ``cleaned``)."""
    cov = pd.DataFrame({f"coverage_type_{bt}": safe_ratio(cleaned[f"supply_{bt}"], cleaned[f"demand_{bt}"])
                        for bt in BLOOD_TYPES}, index=cleaned.index)
    out = cov.copy()
    out["total_coverage"] = safe_ratio(cleaned["supply_total"], cleaned["demand_total"])
    out["total_coverage_lag1"] = out.total_coverage.shift(1)
    out["min_type_coverage"] = cov.min(axis=1)
    out["types_high_count"] = (cov < HIGH_RISK_COVERAGE).sum(axis=1).astype(float).where(cov.notna().any(axis=1))
    out["types_high_count_rolling4"] = out.types_high_count.rolling(4, min_periods=1).mean()
    return out


def engineer_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create only information known at date t and next-week targets at date t+1.

    Non-weekly target gaps are discarded (target is only valid for 5-9 day gaps).
    Rows without next-week targets remain for dashboard future forecasting.
    """
    cross = cross_type_features(cleaned)
    pieces = []
    for blood_type in BLOOD_TYPES:
        weekly = cleaned[["date", "collected_total", "demand_total", "supply_total", "team_total", "Individual_donate"]].copy()
        weekly["blood_type"] = blood_type
        for item in ("collected", "demand", "supply", "platelet_collection", "platelet_supply"):
            weekly[item] = cleaned[f"{item}_{blood_type}"].to_numpy()
        weekly["coverage"] = safe_ratio(weekly["supply"], weekly["demand"])
        weekly["collection_efficiency"] = safe_ratio(weekly["supply"], weekly["collected"])
        weekly["observed_risk"] = weekly["coverage"].map(risk_from_coverage)

        # Calendar information about the observation date is available in advance.
        iso_week = weekly.date.dt.isocalendar().week.astype("int64")
        weekly["month"] = weekly.date.dt.month
        weekly["quarter"] = weekly.date.dt.quarter
        weekly["week_sin"] = np.sin(2 * np.pi * iso_week / 52.18)
        weekly["week_cos"] = np.cos(2 * np.pi * iso_week / 52.18)

        # Historical lags and rolling averages contain only data from t or earlier.
        for item in ("demand", "supply", "collected", "coverage"):
            for lag in (1, 2, 4):
                weekly[f"{item}_lag{lag}"] = weekly[item].shift(lag)
            weekly[f"{item}_rolling4"] = weekly[item].rolling(4, min_periods=1).mean()
        weekly["demand_growth"] = safe_ratio(weekly.demand - weekly.demand_lag1, weekly.demand_lag1)
        weekly["supply_growth"] = safe_ratio(weekly.supply - weekly.supply_lag1, weekly.supply_lag1)

        # Extended candidates (optimisation phase) - still t or earlier.
        weekly["coverage_rolling8"] = weekly.coverage.rolling(8, min_periods=1).mean()
        weekly["coverage_rolling12"] = weekly.coverage.rolling(12, min_periods=1).mean()
        weekly["coverage_std4"] = weekly.coverage.rolling(4, min_periods=2).std()
        weekly["coverage_ewm4"] = weekly.coverage.ewm(span=4, adjust=False).mean()
        weekly["collected_coverage"] = safe_ratio(weekly.collected, weekly.demand)
        weekly["collected_coverage_rolling4"] = weekly.collected_coverage.rolling(4, min_periods=1).mean()
        weekly["demand_vs_rolling12"] = safe_ratio(weekly.demand, weekly.demand.rolling(12, min_periods=1).mean())
        weekly["demand_share"] = safe_ratio(weekly.demand, weekly.demand_total)
        weekly["supply_share"] = safe_ratio(weekly.supply, weekly.supply_total)
        # Calendar of the week being forecast (known in advance, so not leakage).
        nominal_target = weekly.date + pd.Timedelta(days=7)
        target_week = nominal_target.dt.isocalendar().week.astype("int64")
        weekly["target_week_sin"] = np.sin(2 * np.pi * target_week / 52.18)
        weekly["target_week_cos"] = np.cos(2 * np.pi * target_week / 52.18)
        weekly["target_days_to_spring_festival"] = days_to_spring_festival(nominal_target)
        weekly["target_spring_festival_window"] = (weekly.target_days_to_spring_festival.abs() <= 14).astype(float)
        weekly.loc[weekly.target_days_to_spring_festival.isna(), "target_spring_festival_window"] = np.nan

        # Version 2 candidates (week t or earlier).
        net = weekly.collected - weekly.supply
        weekly["net_flow_ratio"] = safe_ratio(net, weekly.demand)
        for window in (4, 13, 26):
            weekly[f"stock_proxy{window}"] = safe_ratio(net.rolling(window, min_periods=1).sum(),
                                                        weekly.demand_rolling4)
        weekly["collected_coverage_lag1"] = weekly.collected_coverage.shift(1)
        for col in cross.columns:
            weekly[col] = cross[col].to_numpy()
        # Optional external weekly columns: same value for all four blood types of a week.
        for col in EXTERNAL_NUMERICAL_FEATURES:
            weekly[col] = cleaned[col].to_numpy() if col in cleaned.columns else np.nan

        # Labels use t+1, NEVER as input features. Limit to approximately weekly pairs.
        weekly["target_date"] = weekly.date.shift(-1)
        days_ahead = (weekly.target_date - weekly.date).dt.days
        valid_gap = days_ahead.between(*TARGET_GAP_DAYS)
        future_supply = weekly.supply.shift(-1).where(valid_gap)
        future_demand = weekly.demand.shift(-1).where(valid_gap)
        weekly["next_demand"] = future_demand
        weekly["next_coverage"] = safe_ratio(future_supply, future_demand)
        weekly["next_risk"] = weekly["next_coverage"].map(risk_from_coverage)
        weekly["target_date"] = weekly.target_date.where(valid_gap)
        pieces.append(weekly)

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["date", "blood_type"]).reset_index(drop=True)


def check_no_leakage(processed: pd.DataFrame, numerical: list[str]) -> None:
    """Raise if any target column could enter the model."""
    forbidden = [c for c in numerical if c.startswith("next_") or c in TARGET_COLUMNS or c == "observed_risk"]
    if forbidden:
        raise ValueError(f"Target/leaky columns found in features: {forbidden}")
    missing = [c for c in numerical if c not in processed.columns]
    if missing:
        raise ValueError(f"Feature columns missing from processed data: {missing}")


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------
def generate_processed(path: Path | None = None, source: CsvSource | None = None,
                       paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Clean + engineer features and save processed_data.csv (and cleaned_weekly.csv)."""
    paths = paths or DEFAULT_PATHS
    source = paths.raw_data if source is None else source
    cleaned = load_and_clean(source)
    processed = engineer_features(cleaned)
    path = Path(path) if path else paths.processed_data
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path.parent / paths.cleaned_data.name, index=False)
    processed.to_csv(path, index=False)
    return processed


def load_processed(path: Path | None = None, paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Load processed features, rebuilding them if missing or outdated (new columns)."""
    paths = paths or DEFAULT_PATHS
    path = Path(path) if path else paths.processed_data
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date", "target_date"])
        if set(ALL_NUMERICAL_FEATURES).issubset(df.columns):
            return df
    return generate_processed(path, paths=paths)


@dataclass
class PreparedDataset:
    report: ValidationReport
    cleaned: pd.DataFrame | None
    processed: pd.DataFrame | None


def prepare_dataset(source: CsvSource) -> PreparedDataset:
    """Validate, clean and engineer an external CSV in memory (writes nothing)."""
    raw = read_raw_csv(source)
    report = validate_raw(raw)
    if not report.ok:
        return PreparedDataset(report, None, None)
    cleaned = clean_raw(raw)
    processed = engineer_features(cleaned)
    labeled = int(processed.next_risk.notna().sum())
    if labeled == 0:
        report.warnings.append("No next-week labels could be created (no 5-9 day gaps); evaluation is impossible.")
    return PreparedDataset(report, cleaned, processed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blood Shortage ML data pipeline")
    parser.add_argument("--clean-only", action="store_true", help="only write data/cleaned_weekly.csv")
    parser.add_argument("--validate", metavar="CSV", help="validate an external CSV and print a report (writes nothing)")
    args = parser.parse_args()
    if args.validate:
        prepared = prepare_dataset(Path(args.validate))
        print(prepared.report.as_frame().to_string(index=False))
        print(prepared.report.summary())
        return
    if args.clean_only:
        frame = save_cleaned()
        print(f"Saved {len(frame)} unique weekly dates to {DEFAULT_PATHS.cleaned_data}")
        print(f"Observed date range: {frame.date.min().date()} to {frame.date.max().date()}")
        return
    df = generate_processed()
    print(f"Saved {len(df)} blood-type/date feature rows to {DEFAULT_PATHS.processed_data}")
    print(f"Rows with next-week labels: {df.next_risk.notna().sum()}")
    print("Risk distribution:")
    print(df.next_risk.value_counts().reindex(RISK_LABELS).to_string())


if __name__ == "__main__":
    main()
