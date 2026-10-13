"""Read and clean the original weekly Zhejiang dataset.

CMD: python -m src.preprocessing
Purpose: clean raw CSV and create data/cleaned_weekly.csv (an optional intermediate file).
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "Zhejiang_blood_donation.csv"
CLEAN_DATA = ROOT / "data" / "cleaned_weekly.csv"
BLOOD_TYPES = ("A", "B", "O", "AB")


def load_and_clean(path: Path = RAW_DATA) -> pd.DataFrame:
    """Clean non-numeric Excel exports, impossible negative values and duplicate dates.

    Duplicate *dates* are merged using the column means rather than summed because
    these entries appear to be conflicting measurements, not disjoint donations.
    This policy is a project assumption and should be reviewed with a data owner.
    Missing numeric inputs are left missing for train-only sklearn imputers.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset missing: {path}")
    df = pd.read_csv(path, dtype="string")
    df.columns = df.columns.str.strip()
    required = {"date"} | {f"{kind}_{b}" for kind in ("demand", "supply", "collected") for b in BLOOD_TYPES}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Missing expected columns: {sorted(absent)}")

    # CMD concept: parse date strings like 3/19/2012 as March 19, 2012.
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"])

    # CMD concept: Excel cells may contain non-breaking spaces or '#VALUE!'.
    for col in df.columns.drop("date"):
        value = df[col].astype("string").str.replace("\u00a0", "", regex=False).str.replace(",", "", regex=False).str.strip()
        num = pd.to_numeric(value, errors="coerce").astype("float64")
        df[col] = num.mask(num < 0, np.nan)

    # Remove exactly identical records, then reconcile conflicting same-date rows.
    df = df.drop_duplicates()
    df = df.groupby("date", as_index=False, sort=True).mean(numeric_only=True)
    return df.sort_values("date").reset_index(drop=True)


def save_cleaned(path: Path = CLEAN_DATA) -> pd.DataFrame:
    df = load_and_clean()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    frame = save_cleaned()
    print(f"Saved {len(frame)} unique weekly dates to {CLEAN_DATA}")
    print(f"Observed date range: {frame.date.min().date()} to {frame.date.max().date()}")
