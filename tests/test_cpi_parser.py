import pandas as pd
import pytest

from src.cpi_parser import prepare_series, to_long, clean
from src.quality import find_gaps, find_start_mismatch, find_stale


def make_wide():
    return pd.DataFrame({
        "H03": ["CPA01100", "CPS00000", "CPS00000", "CPSD0001",
                "CPI60005", "CPS06201", "CPR00000", "CPI60051"],
        "H04": ["Food", "CPI Headline", "All Items", "Deciles - All urban areas",
                "Pensioners", "Medical industry", "All Items", "Geographic indices"],
        "H05": [None, None, None, "Expenditure deciles", None, "Doctors", None, None],
        "H06": [None, None, None, "1", None, None, None, None],
        "H13": ["Western Cape", "All urban areas", "All urban areas", "All urban areas",
                "Total country", "All urban areas", "Rural Areas", "Rural areas"],
        "H18": ["Dec 2024 = 100"] * 8,
        "H24": ["2008 01", "2008 01", "2008 01", "2017 01",
                "2013 01", "2008 01", "2013 01", "2008 01"],
        "MO012008": ["33,8", "41,9", "41,9", None, None, "30,0", None, "41"],
        "MO022008": ["33,9", "42,2", "42,2", None, None, "30,5", None, "42"],
    })


def test_identical_duplicates_dropped_but_different_ones_raise():
    assert len(prepare_series(make_wide())) == 7
    bad = make_wide()
    bad.loc[2, "MO012008"] = "99,9"            # same code, different number
    with pytest.raises(ValueError):
        prepare_series(bad)


def test_geography_is_standardised():
    s = prepare_series(make_wide())
    assert set(s["geography"]) >= {"Rural areas"}
    assert "Rural Areas" not in set(s["geography"])
    assert s.set_index("series_code").loc["CPR00000", "geography_code"] == "RUR"


def test_unknown_geography_raises():
    bad = make_wide()
    bad.loc[0, "H13"] = "Atlantis"
    with pytest.raises(ValueError):
        prepare_series(bad)


def test_series_types():
    t = prepare_series(make_wide()).set_index("series_code")["series_type"]
    assert t["CPSD0001"] == "decile"
    assert t["CPA01100"] == "category"
    assert t["CPI60005"] == "analytical"       # pensioners: not a normal category
    assert t["CPS06201"] == "supplementary"    # 'Medical industry / Doctors'
    assert t["CPI60051"] == "geographic"


def test_category_code():
    s = prepare_series(make_wide()).set_index("series_code")
    assert s.loc["CPA01100", "category_code"] == "01100"
    assert pd.isna(s.loc["CPI60005", "category_code"])


def test_decimal_comma_period_and_empty_cells():
    fact = clean(to_long(prepare_series(make_wide())), "t.xlsx")
    assert set(fact["period_text"]) == {"2008-01", "2008-02"}
    row = fact[(fact.series_code == "CPA01100") & (fact.period_text == "2008-01")]
    assert row["index_value"].iloc[0] == 33.8
    assert not fact.duplicated(subset=["series_code", "period_text"]).any()


def test_quality_checks():
    dim = prepare_series(make_wide())
    fact = clean(to_long(dim), "t.xlsx")
    # CPI60005 and CPR00000 are declared from 2013 but have no values here
    mism = find_start_mismatch(dim, fact)
    assert {"CPI60005", "CPR00000", "CPSD0001"} <= set(mism["series_code"])
    assert find_gaps(fact).empty
    assert find_stale(fact).empty


def test_gap_detection():
    fact = pd.DataFrame({
        "series_code": ["X", "X", "X"],
        "period_text": ["2024-01", "2024-02", "2024-05"],
        "index_value": [1.0, 2.0, 3.0],
    })
    assert find_gaps(fact).iloc[0]["missing_months"] == 2