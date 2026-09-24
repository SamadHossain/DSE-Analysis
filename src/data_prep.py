"""Loading and cleaning of the raw DSE daily price file."""

from __future__ import annotations

import pandas as pd

from .config import CLEANED_DATA_FILE, NON_EQUITY_CODES, OHLCV_COLUMNS, RAW_DATA_FILE


def load_raw(path=RAW_DATA_FILE) -> pd.DataFrame:
    """Read the raw export and normalise the Date column.

    The rest of the pipeline assumes the frame is sorted by security and then
    by date, because every lag, rolling window and return depends on that
    ordering. The raw file is not guaranteed to be in that order, so we sort
    immediately rather than relying on it later.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Trading_Code", "Date"]).reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the state of a frame on a few simple dimensions.

    Run this on the raw frame and again on the cleaned frame. Running it only
    after cleaning is not informative: every check is one the cleaning step
    has already enforced, so the result is 100% by construction.
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    checks = {
        "Rows": n_rows,
        "Missing cells (%)": 100 * df.isna().sum().sum() / total_cells,
        "Exact duplicate rows (%)": 100 * df.duplicated().mean(),
        "Duplicate (code, date) pairs (%)": 100
        * df.duplicated(subset=["Trading_Code", "Date"]).mean(),
        "Rows with a non-positive price (%)": 100
        * (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).mean(),
        "Rows with zero volume (%)": 100 * (df["Volume"] == 0).mean(),
        "Rows violating High >= Low (%)": 100 * (df["High"] < df["Low"]).mean(),
        "Rows with Close outside [Low, High] (%)": 100
        * ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).mean(),
        "Rows dated in the future (%)": 100 * (df["Date"] > pd.Timestamp.today()).mean(),
    }

    report = pd.DataFrame(checks.items(), columns=["Check", "Value"])
    report["Value"] = report["Value"].round(4)
    return report


def clean(df: pd.DataFrame, drop_non_equity: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the cleaning rules and return the cleaned frame plus a log.

    Steps, in order:
      1. drop rows with any missing value
      2. drop exact duplicate rows
      3. drop repeated (Trading_Code, Date) pairs, keeping the first
      4. drop rows where any OHLCV field is zero
      5. optionally drop indices, bonds and mutual funds

    Step 3 was not in the original notebook. Two rows sharing a security and a
    date would silently corrupt every lag and rolling window downstream, so it
    is worth checking explicitly even if the count turns out to be zero.
    """
    log = []

    def record(step: str, before: int, after: int) -> None:
        log.append({"Step": step, "Rows removed": before - after, "Rows remaining": after})

    n = len(df)
    df = df.dropna()
    record("Drop rows with missing values", n, len(df))

    n = len(df)
    df = df.drop_duplicates()
    record("Drop exact duplicate rows", n, len(df))

    n = len(df)
    df = df.drop_duplicates(subset=["Trading_Code", "Date"], keep="first")
    record("Drop repeated (code, date) pairs", n, len(df))

    n = len(df)
    df = df[(df[OHLCV_COLUMNS] != 0).all(axis=1)]
    record("Drop rows with a zero OHLCV value", n, len(df))

    if drop_non_equity:
        n = len(df)
        df = df[~df["Trading_Code"].isin(NON_EQUITY_CODES)]
        record("Drop indices, bonds and mutual funds", n, len(df))

    df = df.sort_values(["Trading_Code", "Date"]).reset_index(drop=True)
    return df, pd.DataFrame(log)


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage change in Close from the previous trading day, per security.

    pct_change() takes the difference against the preceding row, so the frame
    must already be in ascending date order within each security. Applying it
    to a descending frame produces the inverse return, which is what happened
    in the original notebook.
    """
    df = df.sort_values(["Trading_Code", "Date"]).copy()
    df["Daily_Return"] = df.groupby("Trading_Code")["Close"].pct_change() * 100
    return df


def save_cleaned(df: pd.DataFrame, path=CLEANED_DATA_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
