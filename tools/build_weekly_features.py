"""Build weekly external features (holidays, approximate school breaks, COVID restrictions)
for every Monday-starting week from 2012 to 2026, and merge them into the weekly blood data.

Each blood row's date is the MONDAY that starts its week (checked: the weeks containing the
Spring Festival holiday have the lowest collection when the date is taken as the week start).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAY_EN = {"元旦": "new_year", "春节": "spring_festival", "清明节": "qingming", "劳动节": "labour_day",
              "端午节": "dragon_boat", "中秋节": "mid_autumn", "国庆节": "national_day",
              # Years when Mid-Autumn falls inside Golden Week use a combined name; counted as National Day.
              "中秋节、国庆节": "national_day", "国庆节、中秋节": "national_day"}
HOLIDAY_COLUMNS = list(dict.fromkeys(HOLIDAY_EN.values())) + ["other_holiday"]   # other = special one-off holidays (e.g. 3 Sep 2015)


def load_holidays(folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(folder.glob("holiday_*.json")):
        for d in json.loads(f.read_text(encoding="utf-8"))["days"]:
            rows.append({"date": pd.Timestamp(d["date"]), "name": HOLIDAY_EN.get(d["name"], "other_holiday"),
                         "is_off": bool(d["isOffDay"])})
    return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()


def daily_calendar(hol: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    days = pd.DataFrame(index=pd.date_range(start, end, freq="D"))
    days["holiday_off"] = hol.is_off.reindex(days.index).eq(True)
    days["makeup_workday"] = hol.is_off.reindex(days.index).eq(False)
    days["holiday_name"] = hol.name.where(hol.is_off).reindex(days.index)
    weekend = days.index.dayofweek >= 5
    days["off_day"] = days.holiday_off | (weekend & ~days.makeup_workday)
    # Approximate school breaks (Zhejiang): winter = 3 weeks before to 3 weeks after Spring Festival day 1;
    # summer = 1 July - 31 August.  An approximation, not an official school calendar.
    sf_first = days[days.holiday_name.eq("spring_festival")].groupby(days[days.holiday_name.eq("spring_festival")].index.year).apply(lambda g: g.index.min())
    winter = pd.Series(False, index=days.index)
    for first in sf_first:
        winter |= (days.index >= first - pd.Timedelta(days=21)) & (days.index <= first + pd.Timedelta(days=21))
    summer = (days.index.month.isin([7, 8]))
    days["school_break_approx"] = winter | summer
    return days


def weekly_features(days: pd.DataFrame, stringency: pd.Series) -> pd.DataFrame:
    mondays = days.index[days.index.dayofweek == 0]
    off_dates = days.index[days.holiday_off]
    rows = []
    for m in mondays:
        w = days.loc[m:m + pd.Timedelta(days=6)]
        if len(w) < 7:
            continue
        row = {"date": m,
               "holiday_days_off": int(w.holiday_off.sum()),
               "makeup_workdays": int(w.makeup_workday.sum()),
               "total_days_off": int(w.off_day.sum()),
               "school_break_days_approx": int(w.school_break_approx.sum())}
        for name in HOLIDAY_COLUMNS:
            row[f"{name}_days"] = int(w.holiday_name.eq(name).sum())
        nxt = off_dates[off_dates >= m]
        prv = off_dates[off_dates < m]
        row["days_to_next_holiday"] = int((nxt[0] - m).days) if len(nxt) else np.nan
        row["days_since_last_holiday"] = int((m - prv[-1]).days) if len(prv) else np.nan
        s = stringency.reindex(w.index)
        row["covid_stringency_mean"] = round(float(s.mean()), 2) if s.notna().any() else np.nan
        row["covid_stringency_max"] = round(float(s.max()), 2) if s.notna().any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("date")
    # Calendar of the NEXT week is known in advance, so it may be used when forecasting week t+1.
    for c in ["holiday_days_off", "makeup_workdays", "total_days_off", "school_break_days_approx"] + \
             [f"{n}_days" for n in HOLIDAY_COLUMNS]:
        out[f"next_week_{c}"] = out[c].shift(-1)
    return out.reset_index()


def load_stringency(path: Path) -> pd.Series:
    cols = ["CountryCode", "Jurisdiction", "Date", "StringencyIndex_Average"]
    df = pd.read_csv(path, usecols=cols, dtype={"Date": str})
    df = df[(df.CountryCode == "CHN") & (df.Jurisdiction == "NAT_TOTAL")]
    s = pd.Series(df.StringencyIndex_Average.to_numpy(), index=pd.to_datetime(df.Date, format="%Y%m%d")).sort_index()
    full = pd.Series(np.nan, index=pd.date_range("2012-01-01", "2026-12-31", freq="D"))
    full.loc[: "2019-12-31"] = 0.0            # no COVID before 2020
    full.loc[s.index] = s.to_numpy()
    full.loc["2023-01-01":] = 0.0             # China ended COVID restrictions in Jan 2023 (tracker ends 2022-12-31)
    return full


def main():
    hol_dir, ox_path, weekly_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    context_path = Path(sys.argv[4]) if len(sys.argv) == 6 else None
    out_dir = Path(sys.argv[-1]); out_dir.mkdir(parents=True, exist_ok=True)

    hol = load_holidays(hol_dir)
    days = daily_calendar(hol, "2012-01-01", "2027-01-10")
    feats = weekly_features(days, load_stringency(ox_path))
    feats = feats[(feats.date >= "2012-03-01") & (feats.date <= "2026-12-28")]
    feats_out = feats.copy(); feats_out["date"] = feats_out.date.dt.strftime("%Y-%m-%d")
    feats_out.to_csv(out_dir / "weekly_external_features_2012_2026.csv", index=False)

    weekly = pd.read_csv(weekly_path, dtype="string", keep_default_na=False)   # blood rows kept exactly as given
    dates = pd.to_datetime(weekly["date"].str.strip(), format="%m/%d/%Y")
    joined = pd.DataFrame({"date": dates}).merge(feats, on="date", how="left").drop(columns="date")
    merged = pd.concat([weekly, joined.reset_index(drop=True)], axis=1)
    if context_path is not None:   # optional: the quarterly Data_Sheet_2 columns already merged earlier
        ctx = pd.read_csv(context_path, dtype="string", keep_default_na=False)
        extra = [c for c in ctx.columns if c not in weekly.columns]
        merged = pd.concat([merged, ctx[extra]], axis=1)
    name = Path(weekly_path).stem + "_enriched.csv"
    merged.to_csv(out_dir / name, index=False)
    print(f"{name}: {merged.shape}; weeks matched to features: {int(joined.holiday_days_off.notna().sum())}/{len(weekly)}")


if __name__ == "__main__":
    main()
