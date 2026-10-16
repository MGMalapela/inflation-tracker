
"""Parse the Stats SA P0141 CPI sheet ('Excel table from 2008') into tidy tables."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SHEET_NAME = "Excel table from 2008"
MONTH_PATTERN = r"^MO(\d{2})(\d{4})$"        # MO012008 -> January 2008

# Stats SA header code -> our column name
RENAME = {
    "H03": "series_code",
    "H04": "series_label",
    "H05": "sub_label",
    "H06": "decile",
    "H13": "geography",
    "H18": "base_period",
    "H24": "series_start",
}

# lower-case source spelling -> (our code, display name)
GEOGRAPHY_MAP = {
    "total country":   ("ZA",  "Total country"),
    "all urban areas": ("URB", "All urban areas"),
    "rural areas":     ("RUR", "Rural areas"),
    "western cape":    ("WC",  "Western Cape"),
    "eastern cape":    ("EC",  "Eastern Cape"),
    "northern cape":   ("NC",  "Northern Cape"),
    "free state":      ("FS",  "Free State"),
    "kwazulu-natal":   ("KZN", "KwaZulu-Natal"),
    "north-west":      ("NW",  "North West"),
    "gauteng":         ("GP",  "Gauteng"),
    "mpumalanga":      ("MP",  "Mpumalanga"),
    "limpopo":         ("LP",  "Limpopo"),
}

SERIES_COLS = [
    "series_code", "series_type", "category_code", "series_label", "sub_label",
    "decile", "geography_code", "geography", "base_period", "series_start",
]


def month_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if re.match(MONTH_PATTERN, str(c))]


def load_wide(path: Path) -> pd.DataFrame:
    """Read the raw file exactly as it is. Everything is read as text."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=SHEET_NAME, dtype=str)
    return pd.read_csv(path, dtype=str, sep=None, engine="python")


def prepare_series(wide: pd.DataFrame) -> pd.DataFrame:
    """Tidy descriptor columns, standardise geography, classify, drop true duplicates."""
    df = wide.rename(columns=RENAME)
    for col in RENAME.values():
        df[col] = df[col].str.strip()

    # --- duplicates: only drop them if the numbers are identical ---
    months = month_columns(df)
    by_code = len(df.drop_duplicates(subset="series_code"))
    by_code_and_values = len(df.drop_duplicates(subset=["series_code"] + months))
    if by_code != by_code_and_values:
        raise ValueError("A repeated series_code carries different values - inspect the file.")
    repeated = sorted(df.loc[df["series_code"].duplicated(), "series_code"].unique())
    df = df.drop_duplicates(subset="series_code", keep="first")
    print(f"Dropped {len(repeated)} repeated series codes: {repeated}")

    # --- geography: fix 'Rural Areas' vs 'Rural areas' etc., fail loudly on surprises ---
    key = df["geography"].str.lower()
    bad = df.loc[~key.isin(GEOGRAPHY_MAP), "geography"].unique()
    if len(bad):
        raise ValueError(f"Unmapped geography values: {list(bad)}")
    df["geography_code"] = key.map({k: v[0] for k, v in GEOGRAPHY_MAP.items()})
    df["geography"] = key.map({k: v[1] for k, v in GEOGRAPHY_MAP.items()})

    # --- series type ---
    label = df["series_label"].fillna("")
    sub = df["sub_label"].fillna("")
    code = df["series_code"]
    df["series_type"] = np.select(
        [
            label.eq("Geographic indices"),
            sub.eq("Expenditure deciles"),
            label.str.startswith("Analytical series") | code.str.startswith("CPI"),
            sub.ne(""),
        ],
        ["geographic", "decile", "analytical", "supplementary"],
        default="category",
    )
    # CPA01100 -> '01100' (only meaningful for the category rows)
    df["category_code"] = code.str[3:].where(df["series_type"] == "category")
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Wide (one column per month) -> long (one row per series per month)."""
    return df.melt(
        id_vars="series_code",
        value_vars=month_columns(df),
        var_name="month_code",
        value_name="index_value",
    )


def clean(long: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = long.copy()
    parts = out["month_code"].str.extract(MONTH_PATTERN)
    out["period_text"] = parts[1] + "-" + parts[0]          # '2008-01'
    out["index_value"] = pd.to_numeric(
        out["index_value"].str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    out = out.dropna(subset=["index_value"])
    out["source_file"] = source_file
    return out[["series_code", "period_text", "index_value", "source_file"]]


def run(raw_path: Path, out_dir: Path):
    wide = prepare_series(load_wide(raw_path))
    dim = wide[SERIES_COLS]
    fact = clean(to_long(wide), raw_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    dim.to_csv(out_dir / "dim_series.csv", index=False)
    fact.to_csv(out_dir / "fact_cpi.csv", index=False)
    return dim, fact


if __name__ == "__main__":
    from src.quality import find_gaps, find_start_mismatch, find_stale

    dim, fact = run(Path("data/raw/Excel - CPI (COICOP) from January 2008 (202608).xlsx"), Path("data/processed"))
    print(dim["series_type"].value_counts())
    print(dim["geography"].value_counts())
    print(f"{len(dim)} series, {len(fact)} observations")
    print(f"Periods: {fact['period_text'].min()} to {fact['period_text'].max()}")

    print("\n-- Series with gaps inside their range --")
    print(find_gaps(fact))
    print("\n-- Declared start differs from first observed month --")
    mm = find_start_mismatch(dim, fact).merge(dim[["series_code", "series_label", "geography"]])
    summary = (
        mm.groupby(["series_label", "declared_start", "first_observed"])
        .size()
        .reset_index(name="n_series")
    )
    print(summary.to_string())
    print("\n-- Series that stop before the latest month --")
    print(find_stale(fact))