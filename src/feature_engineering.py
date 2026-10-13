"""Turn weekly wide-format observations into next-week modeling examples.

CMD: python -m src.feature_engineering
Purpose: create data/processed_data.csv with 4 rows per date (one per blood type).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessing import BLOOD_TYPES, ROOT, load_and_clean

PROCESSED_DATA = ROOT / "data" / "processed_data.csv"
NUMERICAL_FEATURES = [
    "collected", "supply", "demand", "platelet_collection", "platelet_supply",
    "coverage", "collection_efficiency", "collected_total", "demand_total",
    "supply_total", "team_total", "Individual_donate", "month", "quarter",
    "week_sin", "week_cos", "demand_growth", "supply_growth",
] + [
    f"{variable}_lag{lag}"
    for variable in ("demand", "supply", "collected", "coverage")
    for lag in (1, 2, 4)
] + [f"{variable}_rolling4" for variable in ("demand", "supply", "coverage", "collected")]
FEATURE_COLUMNS = ["blood_type"] + NUMERICAL_FEATURES
RISK_LABELS = ("High", "Medium", "Low")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio; replace zero denominators with missing values."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def risk_from_coverage(coverage: float) -> str | None:
    """Project thresholds: HIGH <80%, MEDIUM 80%-<100%, LOW >=100%."""
    if pd.isna(coverage):
        return None
    if coverage < 0.80:
        return "High"
    if coverage < 1.0:
        return "Medium"
    return "Low"


def engineer_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create only information known at date t and next-week targets at date t+1.

    Non-weekly target gaps are discarded (target is only valid for 5–9 day gaps).
    Rows without next-week targets remain for dashboard future forecasting.
    """
    pieces = []
    for blood_type in BLOOD_TYPES:
        weekly = cleaned[["date", "collected_total", "demand_total", "supply_total", "team_total", "Individual_donate"]].copy()
        weekly["blood_type"] = blood_type
        for item in ("collected", "demand", "supply"):
            weekly[item] = cleaned[f"{item}_{blood_type}"].to_numpy()
        for item in ("platelet_collection", "platelet_supply"):
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

        # Labels use t+1, NEVER as input features. Limit to approximately weekly pairs.
        weekly["target_date"] = weekly.date.shift(-1)
        days_ahead = (weekly.target_date - weekly.date).dt.days
        valid_gap = days_ahead.between(5, 9)
        future_supply = weekly.supply.shift(-1).where(valid_gap)
        future_demand = weekly.demand.shift(-1).where(valid_gap)
        weekly["next_demand"] = future_demand
        weekly["next_coverage"] = safe_ratio(future_supply, future_demand)
        weekly["next_risk"] = weekly["next_coverage"].map(risk_from_coverage)
        weekly["target_date"] = weekly.target_date.where(valid_gap)
        pieces.append(weekly)

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["date", "blood_type"]).reset_index(drop=True)


def generate_processed(path: Path = PROCESSED_DATA) -> pd.DataFrame:
    processed = engineer_features(load_and_clean())
    path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(path, index=False)
    return processed


def load_processed() -> pd.DataFrame:
    if not PROCESSED_DATA.exists():
        return generate_processed()
    df = pd.read_csv(PROCESSED_DATA, parse_dates=["date", "target_date"])
    return df


if __name__ == "__main__":
    df = generate_processed()
    print(f"Saved {len(df)} blood-type/date feature rows to {PROCESSED_DATA}")
    print(f"Rows with next-week labels: {df.next_risk.notna().sum()}")
    print("Risk distribution:")
    print(df.next_risk.value_counts().to_string())
