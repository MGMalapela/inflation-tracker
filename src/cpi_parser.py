"""Parse a wide-format CPI sheet into a clean long-format table."""
from pathlib import Path
import pandas as pd

# ---- CONFIG: adjust these once you've opened the real file ----
SHEET_NAME = 0                 # sheet index or name
HEADER_ROW = 0                 # which row holds the column headers
ID_COLUMNS = ["province_code", "category_code"]
BASE_PERIOD = "Dec 2021=100"   # CHECK the release notes for the real value
# ---------------------------------------------------------------


def load_wide(path: Path) -> pd.DataFrame:
    """Read the raw Excel/CSV file exactly as it is."""
    id_types = {col: str for col in ID_COLUMNS}   # keep codes as text
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW, dtype=id_types)
    return pd.read_csv(path, header=HEADER_ROW, dtype=id_types)

def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide (one column per month) -> long (one row per month)."""
    long = wide.melt(id_vars=ID_COLUMNS, var_name="period", value_name="index_value")
    return long


def clean(long: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = long.copy()

    # 1. Parse dates. errors="coerce" turns junk into NaT instead of crashing
    df["period"] = pd.to_datetime(df["period"], errors="coerce")

    # 2. Parse numbers. Stats SA sometimes uses text like "..." or "-"
    df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")

    # 3. Tidy the text keys
    for col in ID_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    # 4. Drop rows we can't use
    df = df.dropna(subset=["period", "index_value"])

    # 5. Add lineage columns (so you can trace any row back to its source)
    df["base_period"] = BASE_PERIOD
    df["source_file"] = source_file

    # 6. Format the period the way our SQL staging table expects: '2024-03'
    df["period_text"] = df["period"].dt.strftime("%Y-%m")

    return df[["period_text", "province_code", "category_code",
               "index_value", "base_period", "source_file"]]


def run(raw_path: Path, out_path: Path) -> pd.DataFrame:
    wide = load_wide(raw_path)
    long = to_long(wide)
    cleaned = clean(long, source_file=raw_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    return cleaned


if __name__ == "__main__":
    result = run(Path("data/raw/cpi_wide.csv"), Path("data/processed/cpi_long.csv"))
    print(f"Wrote {len(result)} rows")
    print(result.head())