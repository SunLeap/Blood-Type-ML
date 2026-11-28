import pandas as pd
import pytest

from src.config import RAW_DATA
from src.data_pipeline import read_raw_csv, validate_raw
from src.data_repair import repair, repair_dates, repair_values


def _dates(*values):
    return pd.Series(list(values), dtype="string")


def test_year_typo_is_fixed_and_confirmed_by_next_row():
    fixed, log = repair_dates(_dates("12/17/2012", "12/24/2012", "12/31/2013", "01/07/2013"))
    assert fixed.iloc[2] == pd.Timestamp("2012-12-31")
    assert len(log) == 1 and log[0]["original"] == "12/31/2013"


def test_october_block_typed_as_january_is_fixed():
    fixed, _ = repair_dates(_dates("9/21/2015", "9/28/2015", "01/05/2015", "01/12/2015", "1/19/2015",
                                   "1/26/2015", "11/02/2015"))
    assert list(fixed.iloc[2:6].dt.strftime("%Y-%m-%d")) == ["2015-10-05", "2015-10-12", "2015-10-19", "2015-10-26"]


def test_swapped_row_order_keeps_original_dates():
    fixed, log = repair_dates(_dates("08/05/2019", "08/12/2019", "8/26/2019", "8/19/2019", "09/02/2019"))
    assert log == []
    assert fixed.iloc[2] == pd.Timestamp("2019-08-26")


def test_genuine_missing_week_is_kept():
    fixed, log = repair_dates(_dates("8/13/2012", "8/20/2012", "09/03/2012", "09/10/2012"))
    assert log == [] and fixed.iloc[2] == pd.Timestamp("2012-09-03")


def test_value_errors_become_missing_not_guessed():
    rows = []
    for k in range(12):
        rows.append({"date": f"1/{k + 1}/2020", "collected_A": "1000", "collected_B": "900", "collected_O": "1200",
                     "collected_AB": "300", "collected_total": "3400", "demand_A": "1400", "demand_B": "1200",
                     "demand_O": "1500", "demand_AB": "300", "demand_total": "4400", "supply_A": "1400",
                     "supply_B": "1200", "supply_O": "1500", "supply_AB": "300", "supply_total": "4400"})
    df = pd.DataFrame(rows, dtype="string")
    df.loc[5, "supply_A"] = "5"            # typing error
    df.loc[6, "supply_AB"] = "1500"        # copy of O
    df.loc[7, "supply_total"] = "0"        # broken total
    out, log = repair_values(df)
    assert out.loc[5, "supply_A"] == "" and out.loc[6, "supply_AB"] == ""
    assert float(out.loc[7, "supply_total"]) == 4400
    assert {e["column"] for e in log} == {"supply_A", "supply_AB", "supply_total"}


def test_repair_never_overwrites_original(tmp_path):
    with pytest.raises(ValueError):
        repair(RAW_DATA, RAW_DATA, tmp_path / "log.csv")


def test_repaired_primary_dataset_has_no_duplicate_dates(tmp_path):
    before = RAW_DATA.read_bytes()
    target = tmp_path / "corrected.csv"
    log = repair(RAW_DATA, target, tmp_path / "log.csv")
    assert RAW_DATA.read_bytes() == before                     # original untouched
    report = validate_raw(read_raw_csv(target))
    assert report.duplicate_dates == 0 and report.unique_dates == 639
    assert report.irregular_gaps == 1                          # the genuinely missing week in Aug 2012
    assert (log.column == "date").sum() == 36
