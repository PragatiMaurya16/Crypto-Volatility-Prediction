"""Data cleaning: make the raw table consistent before any feature is built."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PRICE_COLS = ["open", "high", "low", "close"]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). `report` records what was changed."""
    rep: dict = {"rows_in": int(len(df)), "coins_in": int(df["symbol"].nunique())}

    # 1. unusable keys
    before = len(df)
    df = df.dropna(subset=["date", "symbol"])
    rep["dropped_bad_key"] = before - len(df)

    # 2. duplicates on (symbol, date): keep the last record
    before = len(df)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    rep["duplicates_removed"] = before - len(df)

    # 3. impossible values -> NaN (they get repaired or dropped below)
    for c in PRICE_COLS:
        rep[f"nonpositive_{c}"] = int((df[c] <= 0).sum())
        df.loc[df[c] <= 0, c] = np.nan
    for c in ["volume", "market_cap"]:
        df.loc[df[c] <= 0, c] = np.nan

    # 4. OHLC consistency: high must be the max, low the min of the four prices
    hi = df[PRICE_COLS].max(axis=1)
    lo = df[PRICE_COLS].min(axis=1)
    rep["ohlc_rows_repaired"] = int(((df["high"] != hi) | (df["low"] != lo)).sum())
    df["high"], df["low"] = hi, lo

    # 5. per coin: regular daily calendar + limited forward fill
    out, gaps_filled, gaps_left = [], 0, 0
    for sym, g in df.groupby("symbol", sort=True):
        name = g["name"].iloc[-1]
        g = g.set_index("date").drop(columns=["symbol", "name"])
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full)
        missing_before = g[PRICE_COLS + ["volume", "market_cap"]].isna().any(axis=1).sum()
        g = g.ffill(limit=config.MAX_FFILL_DAYS)
        missing_after = g[PRICE_COLS + ["volume", "market_cap"]].isna().any(axis=1).sum()
        gaps_filled += int(missing_before - missing_after)
        gaps_left += int(missing_after)
        g = g.dropna(subset=PRICE_COLS + ["volume", "market_cap"])
        g.insert(0, "symbol", sym)
        g.insert(1, "name", name)
        g.index.name = "date"
        out.append(g.reset_index())
    df = pd.concat(out, ignore_index=True)
    rep["missing_cells_forward_filled"] = gaps_filled
    rep["rows_dropped_unfillable_gaps"] = gaps_left

    # 6. drop coins with too little history to build long-window features
    counts = df.groupby("symbol")["date"].transform("count")
    dropped = df.loc[counts < config.MIN_HISTORY_DAYS, "symbol"].nunique()
    df = df[counts >= config.MIN_HISTORY_DAYS].reset_index(drop=True)
    rep["coins_dropped_short_history"] = int(dropped)

    rep["rows_out"] = int(len(df))
    rep["coins_out"] = int(df["symbol"].nunique())
    rep["date_min"] = str(df["date"].min().date())
    rep["date_max"] = str(df["date"].max().date())
    return df, rep
