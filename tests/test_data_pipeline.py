import io

import numpy as np
import pandas as pd
import pytest

from src.config import BLOOD_TYPES, RAW_DATA
from src.data_pipeline import (
    ALL_NUMERICAL_FEATURES, ORIGINAL_NUMERICAL_FEATURES, REQUIRED_COLUMNS, check_no_leakage, clean_raw,
    engineer_features, load_and_clean, prepare_dataset, read_raw_csv, risk_from_coverage, safe_ratio,
    validate_raw,
)


def synthetic_raw(n_weeks: int = 20, start: str = "2020-01-06", fmt: str = "%m/%d/%Y") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n_weeks, freq="7D")
    data = {"date": dates.strftime(fmt)}
    for bt in BLOOD_TYPES:
        data[f"demand_{bt}"] = rng.integers(300, 1500, n_weeks)
        data[f"supply_{bt}"] = rng.integers(250, 1500, n_weeks)
        data[f"collected_{bt}"] = rng.integers(250, 1600, n_weeks)
    return pd.DataFrame(data).astype("string")


# --- definitions -----------------------------------------------------------
@pytest.mark.parametrize("coverage, expected", [(0.0, "High"), (0.7999, "High"), (0.80, "Medium"),
                                                (0.9999, "Medium"), (1.0, "Low"), (1.7, "Low")])
def test_risk_thresholds_unchanged(coverage, expected):
    assert risk_from_coverage(coverage) == expected


def test_risk_of_missing_coverage_is_none():
    assert risk_from_coverage(np.nan) is None


def test_safe_ratio_handles_zero_denominator():
    out = safe_ratio(pd.Series([1.0, 2.0]), pd.Series([0.0, 4.0]))
    assert np.isnan(out[0]) and out[1] == 0.5


# --- cleaning --------------------------------------------------------------
def test_cleaning_nbsp_value_errors_negatives_and_duplicate_dates():
    raw = synthetic_raw(6)
    raw.loc[0, "demand_A"] = "1 200"
    raw.loc[1, "supply_B"] = "#VALUE!"
    raw.loc[2, "collected_O"] = "-5"
    duplicate = raw.iloc[[3]].copy()
    duplicate["demand_A"] = "1000"
    raw.loc[3, "demand_A"] = "800"
    raw = pd.concat([raw, duplicate], ignore_index=True)
    cleaned = clean_raw(raw)
    assert len(cleaned) == 6
    assert cleaned.loc[0, "demand_A"] == 1200
    assert np.isnan(cleaned.loc[1, "supply_B"])
    assert np.isnan(cleaned.loc[2, "collected_O"])
    assert cleaned.loc[3, "demand_A"] == 900  # conflicting same-date rows are averaged
    assert cleaned.date.is_monotonic_increasing


def test_missing_totals_are_derived_and_optional_columns_added():
    cleaned = clean_raw(synthetic_raw(8))
    expected = cleaned[[f"demand_{bt}" for bt in BLOOD_TYPES]].sum(axis=1)
    assert np.allclose(cleaned.demand_total, expected)
    assert "platelet_supply_A" in cleaned and cleaned.platelet_supply_A.isna().all()


def test_iso_dates_are_accepted():
    cleaned = clean_raw(synthetic_raw(8, fmt="%Y-%m-%d"))
    assert cleaned.date.notna().all() and len(cleaned) == 8


def test_primary_dataset_matches_documented_properties():
    raw = read_raw_csv(RAW_DATA)
    assert raw.shape == (639, 29)
    report = validate_raw(raw)
    assert report.ok and report.can_train
    assert report.duplicate_dates == 13
    cleaned = load_and_clean()
    assert len(cleaned) == 626
    assert str(cleaned.date.min().date()) == "2012-03-19" and str(cleaned.date.max().date()) == "2024-06-17"


# --- validation of external files -----------------------------------------
def test_validation_rejects_missing_required_columns():
    raw = synthetic_raw(10).drop(columns=["supply_AB"])
    report = validate_raw(raw)
    assert not report.ok and "supply_AB" in report.errors[0]


def test_validation_rejects_non_numeric_required_column():
    raw = synthetic_raw(10)
    raw["demand_B"] = "abc"
    assert not validate_raw(raw).ok


def test_short_file_can_predict_but_not_train():
    report = validate_raw(synthetic_raw(20))
    assert report.ok and not report.can_train


def test_prepare_dataset_from_uploaded_bytes():
    buffer = io.BytesIO(synthetic_raw(12).to_csv(index=False).encode("utf-8"))
    prepared = prepare_dataset(buffer)
    assert prepared.report.ok
    assert len(prepared.processed) == 12 * 4
    assert set(ALL_NUMERICAL_FEATURES).issubset(prepared.processed.columns)


def test_case_insensitive_column_names():
    raw = synthetic_raw(10)
    raw.columns = [c.upper() if c != "date" else "Date" for c in raw.columns]
    assert validate_raw(read_raw_csv(io.StringIO(raw.to_csv(index=False)))).ok


# --- features and leakage ------------------------------------------------------
@pytest.fixture(scope="module")
def processed():
    return engineer_features(load_and_clean())


def test_four_blood_types_per_date(processed):
    assert len(processed) == 2504
    assert processed.groupby("date").blood_type.nunique().eq(4).all()


def test_feature_lists_contain_no_targets(processed):
    for feats in (ORIGINAL_NUMERICAL_FEATURES, ALL_NUMERICAL_FEATURES):
        check_no_leakage(processed, feats)
        assert not any(c.startswith("next_") or c == "target_date" for c in feats)
    with pytest.raises(ValueError):
        check_no_leakage(processed, ["demand", "next_demand"])


def test_lags_and_targets_are_aligned(processed):
    a = processed[processed.blood_type == "A"].reset_index(drop=True)
    assert a.demand_lag1.iloc[10] == a.demand.iloc[9]
    assert a.demand_lag4.iloc[10] == a.demand.iloc[6]
    gap = (a.date.shift(-1) - a.date).dt.days
    valid = gap.between(5, 9)
    assert np.allclose(a.next_demand[valid], a.demand.shift(-1)[valid])
    assert a.next_demand[~valid].isna().all()
    assert a.next_risk.notna().sum() == 586


def test_features_at_t_do_not_change_when_future_changes():
    """Strong leakage test: rewriting every week after t must not alter any feature at or before t."""
    cleaned = load_and_clean()
    cut = 300
    base = engineer_features(cleaned)
    altered = cleaned.copy()
    numeric = altered.columns.drop("date")
    altered.loc[cut + 1:, numeric] = altered.loc[cut + 1:, numeric] * 3 + 17
    changed = engineer_features(altered)
    cutoff_date = cleaned.date.iloc[cut]
    left = base[base.date <= cutoff_date][ALL_NUMERICAL_FEATURES].reset_index(drop=True)
    right = changed[changed.date <= cutoff_date][ALL_NUMERICAL_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_required_columns_listed():
    assert REQUIRED_COLUMNS[0] == "date" and len(REQUIRED_COLUMNS) == 13


# ---------------------------------------------------------------------------
# Version 2 features
# ---------------------------------------------------------------------------
def test_cross_type_features_match_each_types_coverage(processed):
    from src.data_pipeline import BLOOD_TYPES
    one_date = processed[processed.date == processed.date.iloc[400]]
    for bt in BLOOD_TYPES:
        own = one_date.loc[one_date.blood_type == bt, "coverage"].iloc[0]
        assert np.allclose(one_date[f"coverage_type_{bt}"], own)  # same value on all four rows
    covs = one_date.set_index("blood_type").coverage
    assert one_date.types_high_count.iloc[0] == (covs < 0.80).sum()
    assert np.isclose(one_date.min_type_coverage.iloc[0], covs.min())


def test_stock_proxy_is_net_flow_over_recent_demand(processed):
    a = processed[processed.blood_type == "A"].reset_index(drop=True)
    i = 200
    net = (a.collected - a.supply).iloc[i - 12:i + 1].sum()
    assert np.isclose(a.stock_proxy13.iloc[i], net / a.demand_rolling4.iloc[i])
    assert np.isclose(a.collected_coverage_lag1.iloc[i], a.collected_coverage.iloc[i - 1])
