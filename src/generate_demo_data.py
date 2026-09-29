"""Create a synthetic daily OHLCV + market-cap dataset with the same schema as
the Kaggle file. It has volatility clustering, fat tails, a shared market
factor and a few deliberate data problems (gaps, NaNs, duplicates) so that the
cleaning code is exercised. Tickers are fictional on purpose.

Usage:  python -m src.generate_demo_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TICKERS = [
    "AURX", "BLTC", "CIVN", "DRFT", "EMBR", "FLUX", "GLDN", "HRZN", "IONA", "JUNO",
    "KRYO", "LMNA", "MRDN", "NOVA", "ORBT", "PLSR", "QNTA", "RVLT", "SLCE", "TRNX",
    "UMBR", "VRTX", "WAVE", "XNDR", "YLDS",
]


def _garch(n, omega, alpha, beta, rng, dof=5):
    sig2 = np.empty(n)
    eps = np.empty(n)
    sig2[0] = omega / (1 - alpha - beta)
    z = rng.standard_t(dof, n) / np.sqrt(dof / (dof - 2))
    for t in range(n):
        if t > 0:
            sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
    return eps, np.sqrt(sig2)


def make(start="2018-01-01", end="2022-12-31", seed=config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    market, _ = _garch(n, 4e-6, 0.09, 0.86, rng)
    rows = []
    for i, sym in enumerate(TICKERS):
        base_vol = rng.uniform(0.025, 0.06)
        alpha, beta = rng.uniform(0.08, 0.14), rng.uniform(0.78, 0.84)
        omega = base_vol**2 * (1 - alpha - beta)
        idio, sigma = _garch(n, omega, alpha, beta, rng)
        r = 0.6 * market * rng.uniform(0.7, 1.3) + idio
        # each coin lists at a different date -> unequal history lengths
        offset = int(rng.integers(0, 500)) if i >= 6 else 0
        close = rng.uniform(0.05, 2000) * np.exp(np.cumsum(r))
        openp = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, sigma * 0.15))
        hi = np.maximum(openp, close) * np.exp(np.abs(rng.normal(0, sigma * 0.55)))
        lo = np.minimum(openp, close) * np.exp(-np.abs(rng.normal(0, sigma * 0.55)))
        vol_level = rng.uniform(5e6, 8e8)
        volume = vol_level * np.exp(6 * (sigma - base_vol) + 3 * np.abs(r) + rng.normal(0, 0.35, n))
        supply = rng.uniform(5e7, 2e9) * (1 + 0.0004 * np.arange(n))
        mcap = close * supply
        df = pd.DataFrame(dict(Date=dates, Symbol=sym, Name=f"{sym.title()} Coin", Open=openp,
                               High=hi, Low=lo, Close=close, Volume=volume, Marketcap=mcap))
        rows.append(df.iloc[offset:])
    out = pd.concat(rows, ignore_index=True)

    # inject realistic mess
    idx = rng.choice(len(out), 250, replace=False)
    out.loc[idx[:60], "Volume"] = np.nan
    out.loc[idx[60:120], "Marketcap"] = np.nan
    out.loc[idx[120:150], "Close"] = np.nan
    out = out.drop(index=rng.choice(len(out), 120, replace=False))          # missing days
    out = pd.concat([out, out.sample(40, random_state=1)], ignore_index=True)  # duplicates
    out = out.sample(frac=1, random_state=3).reset_index(drop=True)
    out = out.round({"Open": 6, "High": 6, "Low": 6, "Close": 6, "Volume": 0, "Marketcap": 0})
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d 23:59:59")
    return out


if __name__ == "__main__":
    df = make()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / "demo_crypto_prices.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows for {df['Symbol'].nunique()} coins -> {path}")
