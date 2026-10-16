"""Data-quality checks on the cleaned CPI tables. Each returns a DataFrame of problems."""
import pandas as pd


def _month_number(period_text: pd.Series) -> pd.Series:
    """'2024-03' -> a running month number, so gaps can be counted."""
    return period_text.str[:4].astype(int) * 12 + period_text.str[5:7].astype(int)


def find_gaps(fact: pd.DataFrame) -> pd.DataFrame:
    """Series with missing months between their first and last observation."""
    m = fact.assign(m=_month_number(fact["period_text"]))
    g = m.groupby("series_code")["m"].agg(["min", "max", "count"])
    g["missing_months"] = (g["max"] - g["min"] + 1) - g["count"]
    return g[g["missing_months"] > 0].reset_index()[["series_code", "missing_months"]]


def find_start_mismatch(dim: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    """Series whose declared start (H24) differs from their first real value."""
    first = fact.groupby("series_code")["period_text"].min().rename("first_observed")
    out = dim[["series_code", "series_start"]].merge(first, on="series_code", how="left")
    out["declared_start"] = out["series_start"].str.replace(" ", "-", regex=False)
    return out[out["declared_start"] != out["first_observed"]][
        ["series_code", "declared_start", "first_observed"]
    ]


def find_stale(fact: pd.DataFrame) -> pd.DataFrame:
    """Series whose last observation is earlier than the latest month in the data."""
    last = fact.groupby("series_code")["period_text"].max().rename("last_observed")
    latest = fact["period_text"].max()
    return last[last < latest].reset_index()