import pandas as pd
from src.cpi_parser import clean, to_long
from pathlib import Path
from src.cpi_parser import load_wide


def make_wide():
    return pd.DataFrame({
        "province_code": ["GP", "GP"],
        "category_code": ["01", "07"],
        "2024-01": [100.0, 99.0],
        "2024-02": [101.0, "..."],      # junk value on purpose
    })


def test_melt_produces_one_row_per_month():
    long = to_long(make_wide())
    assert len(long) == 4           # 2 series x 2 months


def test_junk_values_are_dropped():
    cleaned = clean(to_long(make_wide()), "test.csv")
    assert len(cleaned) == 3        # the "..." row is gone
    assert cleaned["index_value"].notna().all()


def test_no_duplicate_keys():
    cleaned = clean(to_long(make_wide()), "test.csv")
    key = ["period_text", "province_code", "category_code"]
    assert not cleaned.duplicated(subset=key).any()


def test_period_format():
    cleaned = clean(to_long(make_wide()), "test.csv")
    assert cleaned["period_text"].str.match(r"^\d{4}-\d{2}$").all()

def test_category_codes_keep_leading_zeros(tmp_path: Path):
    f = tmp_path / "sample.csv"
    f.write_text("province_code,category_code,2024-01\nGP,01,100.0\n")
    df = load_wide(f)
    assert df["category_code"].iloc[0] == "01"