"""Load raw cryptocurrency price data.

Works with the Kaggle "Cryptocurrency Historical Prices" layout (one CSV per
coin, e.g. coin_Bitcoin.csv) and also with a single combined CSV that has a
symbol/name column. Column names are normalised so small naming differences
(Marketcap / Market Cap / market_cap) do not matter.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

REQUIRED = ["date", "symbol", "open", "high", "low", "close", "volume", "market_cap"]

_ALIASES = {
    "marketcap": "market_cap",
    "market cap": "market_cap",
    "market_cap": "market_cap",
    "mcap": "market_cap",
    "timestamp": "date",
    "time": "date",
    "ticker": "symbol",
    "coin": "name",
    "currency": "name",
    "vol": "volume",
}


def _normalise(df: pd.DataFrame, fallback_symbol: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: _ALIASES[c] for c in df.columns if c in _ALIASES})

    if "symbol" not in df.columns:
        if "name" in df.columns:
            df["symbol"] = df["name"]
        else:
            df["symbol"] = fallback_symbol or "COIN"
    if "name" not in df.columns:
        df["name"] = df["symbol"]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    for c in ["open", "high", "low", "close", "volume", "market_cap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "symbol", "name", "open", "high", "low", "close", "volume", "market_cap"]]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read every CSV under `path` (file or folder). Defaults to data/raw."""
    path = Path(path) if path else config.RAW_DIR
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.csv"))
        # Prefer real data over the bundled demo file when both are present.
        real = [f for f in files if not f.name.startswith("demo_")]
        files = real or files
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {path}. Put the Kaggle files in data/raw/ "
            "or run `python -m src.generate_demo_data` to create demo data."
        )
    frames = [_normalise(pd.read_csv(f), fallback_symbol=f.stem.replace("coin_", "")) for f in files]
    return pd.concat(frames, ignore_index=True)


def is_demo_data() -> bool:
    files = sorted(config.RAW_DIR.glob("*.csv"))
    return bool(files) and all(f.name.startswith("demo_") for f in files)
