"""Feature engineering.

Every feature at row t only uses data from day t or earlier. The target is the
realised volatility over days t+1 ... t+HORIZON, so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LN2 = np.log(2)

FEATURE_GROUPS = {
    "Returns": ["ret_1", "abs_ret_1", "ret_3", "ret_7", "ret_30"],
    "Rolling volatility": ["vol_3", "vol_7", "vol_14", "vol_30", "vol_90",
                           "vol_ratio_7_30", "vol_ratio_3_14", "downside_vol_14"],
    "Range-based volatility": ["range_hl", "range_ma_7", "range_ma_14",
                               "parkinson_14", "garman_klass_14", "atr_pct_7", "atr_pct_14"],
    "Bollinger Bands": ["bb_width_20", "bb_pctb_20"],
    "Trend and momentum": ["sma_ratio_7", "sma_ratio_30", "ema_ratio_12_26", "rsi_14"],
    "Volume and liquidity": ["log_volume", "volume_chg_1", "volume_ratio_7", "volume_ratio_30",
                             "volume_std_14", "liq_ratio", "liq_ratio_ma_7", "liq_ratio_chg_7",
                             "amihud_14"],
    "Size and history": ["log_mcap", "log_age_days"],
    "Calendar": ["dow_sin", "dow_cos"],
    "Market-wide": ["market_vol_7", "market_ret_7", "rel_vol_7"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
TARGET = "target_vol"
BASELINE = "vol_7"   # naive forecast: "next week will look like last week"


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _coin_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features for ONE coin. Reindexed to a daily calendar so windows are real days."""
    g = g.set_index("date").sort_index()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    o, h, l, c, v, m = (g[k] for k in ["open", "high", "low", "close", "volume", "market_cap"])
    f = pd.DataFrame(index=g.index)

    r = np.log(c / c.shift(1))
    f["ret_1"] = r
    f["abs_ret_1"] = r.abs()
    for n in (3, 7, 30):
        f[f"ret_{n}"] = np.log(c / c.shift(n))

    for n in (3, 7, 14, 30, 90):
        f[f"vol_{n}"] = r.rolling(n, min_periods=n).std()
    f["vol_ratio_7_30"] = f["vol_7"] / f["vol_30"]
    f["vol_ratio_3_14"] = f["vol_3"] / f["vol_14"]
    f["downside_vol_14"] = r.clip(upper=0).rolling(14, min_periods=14).std()

    hl = np.log(h / l)
    f["range_hl"] = (h - l) / c
    f["range_ma_7"] = f["range_hl"].rolling(7, min_periods=7).mean()
    f["range_ma_14"] = f["range_hl"].rolling(14, min_periods=14).mean()
    f["parkinson_14"] = np.sqrt((hl**2).rolling(14, min_periods=14).mean() / (4 * LN2))
    gk = 0.5 * hl**2 - (2 * LN2 - 1) * np.log(c / o) ** 2
    f["garman_klass_14"] = np.sqrt(gk.rolling(14, min_periods=14).mean().clip(lower=0))
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_pct_7"] = tr.rolling(7, min_periods=7).mean() / c
    f["atr_pct_14"] = tr.rolling(14, min_periods=14).mean() / c

    sma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std()
    upper, lower = sma20 + 2 * sd20, sma20 - 2 * sd20
    f["bb_width_20"] = (upper - lower) / sma20
    f["bb_pctb_20"] = (c - lower) / (upper - lower).replace(0, np.nan)

    f["sma_ratio_7"] = c / c.rolling(7, min_periods=7).mean() - 1
    f["sma_ratio_30"] = c / c.rolling(30, min_periods=30).mean() - 1
    f["ema_ratio_12_26"] = c.ewm(span=12, min_periods=12).mean() / c.ewm(span=26, min_periods=26).mean() - 1
    f["rsi_14"] = _rsi(c)

    lv = np.log(v)
    f["log_volume"] = lv
    f["volume_chg_1"] = lv.diff()
    f["volume_ratio_7"] = v / v.rolling(7, min_periods=7).mean()
    f["volume_ratio_30"] = v / v.rolling(30, min_periods=30).mean()
    f["volume_std_14"] = lv.rolling(14, min_periods=14).std()
    liq = v / m
    f["liq_ratio"] = liq
    f["liq_ratio_ma_7"] = liq.rolling(7, min_periods=7).mean()
    f["liq_ratio_chg_7"] = liq / liq.shift(7) - 1
    f["amihud_14"] = np.log1p((r.abs() / v).rolling(14, min_periods=14).mean() * 1e9)

    f["log_mcap"] = np.log(m)
    f["log_age_days"] = np.log1p(np.arange(len(f)))

    dow = f.index.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # target: realised vol over the next HORIZON days (t+1 .. t+HORIZON)
    fwd = r.rolling(config.HORIZON, min_periods=config.HORIZON).std().shift(-config.HORIZON)
    f[TARGET] = fwd
    f["close"] = c
    return f


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model table from cleaned data (all coins)."""
    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        f = _coin_features(g)
        f.insert(0, "symbol", sym)
        parts.append(f)
    out = pd.concat(parts)
    out.index.name = "date"
    out = out.reset_index()

    # market-wide context: information available on day t across all coins
    daily = out.groupby("date")
    out["market_vol_7"] = daily["vol_7"].transform("mean")
    out["market_ret_7"] = daily["ret_7"].transform("mean")
    out["rel_vol_7"] = out["vol_7"] / out["market_vol_7"]

    out = out.replace([np.inf, -np.inf], np.nan)
    # keep rows with a complete feature vector; target may be NaN (latest days)
    out = out.dropna(subset=FEATURES).reset_index(drop=True)
    return out
