from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BLOOD_TYPES, DATA_DIR, RAW_DATA, RESULTS_DIR

CORRECTED_DATA = DATA_DIR / "Zhejiang_blood_donation_corrected.csv"
REPAIR_LOG = RESULTS_DIR / "data_repair_log.csv"
SHARE_HIGH, SHARE_LOW = 2.5, 0.4
LOOKAHEAD_ROWS = 6


def _number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(" ", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce")


def _fmt(d: pd.Timestamp) -> str:
    return f"{d.month}/{d.day}/{d.year}"


def repair_dates(dates: pd.Series) -> tuple[pd.Series, list[dict]]:
    """Return corrected dates (same index) and a log of changes."""
    parsed = pd.to_datetime(dates.astype("string").str.strip(), format="%m/%d/%Y", errors="coerce")
    fixed = parsed.copy()
    log: list[dict] = []
    last_good = None
    i, n = 0, len(parsed)
    while i < n:
        d = parsed.iloc[i]
        if pd.isna(d) or last_good is None or 5 <= (d - last_good).days <= 9:
            if not pd.isna(d):
                last_good = d
            i += 1
            continue
        candidate = last_good + pd.Timedelta(days=7)
        confirmed = None
        for j in range(i + 1, min(n, i + 1 + LOOKAHEAD_ROWS)):
            if parsed.iloc[j] == candidate + pd.Timedelta(days=7 * (j - i)):
                confirmed = j
                break
        if confirmed is None:            # genuine gap or unexplained: keep as is
            last_good = d
            i += 1
            continue
        block = range(i, confirmed)
        new = [candidate + pd.Timedelta(days=7 * k) for k in range(len(block))]
        old = [parsed.iloc[k] for k in block]
        if sorted(old) == sorted(new) or any(x.dayofweek != 0 for x in new):
            last_good = max(old)         # only the row order differs: dates are fine
            i = confirmed
            continue
        for k, value in zip(block, new):
            fixed.iloc[k] = value
            log.append({"row": int(k) + 2, "column": "date", "original": str(dates.iloc[k]).strip(),
                        "corrected": _fmt(value), "rule": "date out of weekly sequence; confirmed by later rows"})
        last_good = new[-1]
        i = confirmed
    return fixed, log


def repair_values(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Set clear entry errors to missing (or recompute broken totals). ``df`` holds raw text."""
    out = df.copy()
    log: list[dict] = []
    num = {c: _number(df[c]) for c in df.columns if c != "date"}

    def note(idx, col, new, rule):
        log.append({"row": int(idx) + 2, "column": col, "original": str(df.at[idx, col]).strip(),
                    "corrected": "" if pd.isna(new) else new, "rule": rule})
        out.at[idx, col] = "" if pd.isna(new) else str(new)

    for kind in ("collected", "demand", "supply"):
        cols = [f"{kind}_{bt}" for bt in BLOOD_TYPES]
        parts = pd.DataFrame({c: num[c] for c in cols})
        share = parts.div(parts.sum(axis=1), axis=0)
        ratio = share / share.median()
        flagged = (ratio > SHARE_HIGH) | (ratio < SHARE_LOW)
        for col in cols:
            for idx in flagged.index[flagged[col].fillna(False)]:
                note(idx, col, np.nan, f"share of {kind} total is {ratio.at[idx, col]:.2f}x its usual share "
                                        f"(limits {SHARE_LOW}-{SHARE_HIGH}x): entry error, set missing")
        total = f"{kind}_total"
        if total in num:
            for idx in parts.index:
                t, row = num[total].iloc[idx], parts.loc[idx]
                if row.isna().any() or pd.isna(t):
                    continue
                if t == 0 or any(np.isclose(t, v) for v in row):
                    note(idx, total, round(float(row.sum()), 2),
                         "total is 0 or a copy of one blood type: replaced by the sum of the four types")

    if {"team_total", "collected_total"} <= set(num):
        bad = num["team_total"] > num["collected_total"] * 1.05
        for idx in bad.index[bad.fillna(False)]:
            note(idx, "team_total", np.nan, "team donations exceed total collection: set missing")
            if "Individual_donate" in df:
                note(idx, "Individual_donate", np.nan, "derived from the invalid team_total: set missing")

    if {"platelet_collection_tx_units", "platelet_collection"} <= set(num):
        units, donations = num["platelet_collection_tx_units"], num["platelet_collection"]
        r = units / donations
        bad = (r > 3 * r.median()) | (r < r.median() / 3)
        for idx in bad.index[bad.fillna(False)]:
            # Blame whichever of the two is further from its own typical level.
            units_off = abs(np.log(units.iloc[idx] / units.median()))
            donations_off = abs(np.log(donations.iloc[idx] / donations.median()))
            if units_off >= donations_off:
                for col in ["platelet_collection_tx_units"] + [f"platelet_collection_{bt}" for bt in BLOOD_TYPES]:
                    if col in df:
                        note(idx, col, np.nan, "platelet units far from their usual ratio to donations: set missing")
            else:
                note(idx, "platelet_collection", np.nan,
                     "platelet donations far from their usual ratio to units: set missing")
    return out, log


def repair(source: Path = RAW_DATA, target: Path = CORRECTED_DATA, log_path: Path = REPAIR_LOG) -> pd.DataFrame:
    """Write the corrected copy and the change log; return the log."""
    if Path(source).resolve() == Path(target).resolve():
        raise ValueError("Refusing to overwrite the original dataset.")
    raw = pd.read_csv(source, dtype="string", keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    dates, date_log = repair_dates(raw["date"])
    fixed, value_log = repair_values(raw)
    fixed["date"] = [_fmt(d) if not pd.isna(d) else str(o) for d, o in zip(dates, raw["date"])]
    target.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(target, index=False)
    log = pd.DataFrame(date_log + value_log, columns=["row", "column", "original", "corrected", "rule"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.sort_values(["row", "column"]).to_csv(log_path, index=False)
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a corrected copy of the Zhejiang dataset")
    parser.add_argument("--source", type=Path, default=RAW_DATA)
    parser.add_argument("--target", type=Path, default=CORRECTED_DATA)
    args = parser.parse_args()
    log = repair(args.source, args.target)
    print(f"Wrote {args.target} ({len(log)} changes; log: {REPAIR_LOG})")
    print(log.groupby("column").size().to_string())


if __name__ == "__main__":
    main()
