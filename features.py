"""Feature construction for the next-day closing price task."""

from __future__ import annotations

import pandas as pd

LAGS = [1, 2, 3, 5, 10]
WINDOWS = [5, 10, 20]

ID_COLUMNS = ["Trading_Code", "Date"]
TARGET = "Target_Close"


def build_features(df: pd.DataFrame, keep_latest_incomplete: bool = False) -> pd.DataFrame:
    """Add lags, moving averages, calendar fields and the prediction target.

    Every rolling and shifting operation is grouped by Trading_Code so that
    one security never borrows values from another. The frame is sorted first
    because groupby preserves the order it is given.

    The target is the *next* trading day's close, so features computed from
    day t (including day t's own OHLCV) are all known at the time the
    prediction would be made. There is no look-ahead in the feature set.

    By default, the most recent row of each security has no known target
    (there is no "next day" yet in the data) and is dropped along with the
    other incomplete rows at the start of each history. Pass
    ``keep_latest_incomplete=True`` to keep that final row — its inputs are
    complete even though its target is not, which is exactly what a genuine
    forward prediction (rather than a backtest) needs to run on.
    """
    out = df.sort_values(ID_COLUMNS).copy()
    grouped = out.groupby("Trading_Code", sort=False)

    out[TARGET] = grouped["Close"].shift(-1)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        for lag in LAGS:
            out[f"{col}_Lag_{lag}"] = grouped[col].shift(lag)

    for window in WINDOWS:
        out[f"SMA_Close_{window}"] = grouped["Close"].transform(
            lambda s, w=window: s.rolling(window=w).mean()
        )
        out[f"EMA_Close_{window}"] = grouped["Close"].transform(
            lambda s, w=window: s.ewm(span=w, adjust=False).mean()
        )
        out[f"Volatility_{window}"] = grouped["Daily_Return"].transform(
            lambda s, w=window: s.rolling(window=w).std()
        )

    out["Volume_Ratio_20"] = out["Volume"] / grouped["Volume"].transform(
        lambda s: s.rolling(window=20).mean()
    )

    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["Month"] = out["Date"].dt.month
    out["Year"] = out["Date"].dt.year
    out["DayOfYear"] = out["Date"].dt.dayofyear
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["Day"] = out["Date"].dt.day

    out["Daily_Range"] = out["High"] - out["Low"]
    out["Open_Close_Diff"] = out["Open"] - out["Close"]

    input_cols = [c for c in out.columns if c != TARGET]
    if keep_latest_incomplete:
        # Every input feature must still be present; only a missing target is tolerated.
        return out.dropna(subset=input_cols).reset_index(drop=True)

    # Lags and rolling windows leave NaNs at the start of each security's
    # history, and shift(-1) leaves one at the end. Both are dropped.
    return out.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame, drop_year: bool = True) -> list[str]:
    """Model inputs: everything except identifiers and the target.

    Year is excluded by default. Under a chronological split the test years
    never appear in training, so a tree can only ever send them down the
    right-most branch of any split on that column. It adds nothing at
    prediction time and invites the model to memorise period effects.
    """
    excluded = ID_COLUMNS + [TARGET] + (["Year"] if drop_year else [])
    return [c for c in df.columns if c not in excluded]
