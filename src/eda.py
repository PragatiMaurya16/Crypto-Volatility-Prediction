"""Exploratory data analysis: summary statistics, figures and a written report."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config, data_loader, features as F, style

style.apply()


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / f"eda_{name}.png")
    plt.close(fig)


def run(clean: pd.DataFrame, feats: pd.DataFrame, cleaning_report: dict) -> dict:
    s: dict = {"cleaning": cleaning_report}
    clean = clean.sort_values(["symbol", "date"])
    clean["log_ret"] = clean.groupby("symbol")["close"].transform(lambda x: np.log(x / x.shift(1)))

    # ---- dataset statistics ----
    per_coin = clean.groupby("symbol").agg(
        first=("date", "min"), last=("date", "max"), days=("date", "count"),
        avg_close=("close", "mean"), avg_volume=("volume", "mean"), avg_mcap=("market_cap", "mean"),
        daily_vol=("log_ret", "std"))
    per_coin["worst_day"] = clean.groupby("symbol")["log_ret"].min()
    per_coin["best_day"] = clean.groupby("symbol")["log_ret"].max()
    per_coin.sort_values("daily_vol", ascending=False).to_csv(config.REPORT_DIR / "eda_per_coin_stats.csv")
    s["per_coin_top_vol"] = per_coin["daily_vol"].nlargest(3).round(4).to_dict()
    s["per_coin_low_vol"] = per_coin["daily_vol"].nsmallest(3).round(4).to_dict()

    desc = clean[["open", "high", "low", "close", "volume", "market_cap", "log_ret"]].describe().T
    desc.to_csv(config.REPORT_DIR / "eda_summary_stats.csv")

    r = clean["log_ret"].dropna()
    s["returns"] = {"mean": float(r.mean()), "std": float(r.std()), "skew": float(stats.skew(r)),
                    "excess_kurtosis": float(stats.kurtosis(r)), "min": float(r.min()), "max": float(r.max()),
                    "share_abs_gt_10pct": float((r.abs() > 0.10).mean())}

    # ---- 1. price paths (rebased to 100) ----
    fig, ax = plt.subplots(figsize=(9, 3.6))
    top = clean.groupby("symbol")["market_cap"].mean().nlargest(5).index
    for sym in top:
        g = clean[clean["symbol"] == sym]
        ax.plot(g["date"], g["close"] / g["close"].iloc[0] * 100, lw=1, label=sym)
    ax.set_yscale("log")
    ax.set(title="Price paths of the 5 largest coins (rebased to 100, log scale)", ylabel="Index")
    ax.legend(ncol=5, loc="upper left")
    _save(fig, "prices")

    # ---- 2. market volatility through time ----
    mv = feats.groupby("date")["vol_7"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(mv.index, mv.values, color=style.BLUE, lw=1)
    ax.fill_between(mv.index, mv.values, color=style.BLUE, alpha=0.12)
    ax.set(title="Average 7-day volatility across all coins", ylabel="Std of daily log return")
    _save(fig, "market_volatility")
    s["market_vol"] = {"peak_date": str(mv.idxmax().date()), "peak": float(mv.max()),
                       "median": float(mv.median()), "min": float(mv.min())}

    # ---- 3. return distribution vs normal ----
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rr = r.clip(-0.4, 0.4)
    ax.hist(rr, bins=120, density=True, color=style.BLUE, alpha=0.8, label="Daily log returns")
    xs = np.linspace(-0.4, 0.4, 400)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=style.AMBER, lw=1.5, label="Normal fit")
    ax.set(title="Returns have fat tails", xlabel="Daily log return", ylabel="Density")
    ax.legend()
    _save(fig, "return_distribution")

    # ---- 4. volatility clustering: autocorrelation of |returns| ----
    lags = np.arange(1, 61)
    ac = []
    for k in lags:
        vals = [g["log_ret"].abs().autocorr(k) for _, g in clean.groupby("symbol") if len(g) > 200]
        ac.append(np.nanmean(vals))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(lags, ac, color=style.TEAL, width=0.8)
    ax.set(title="Volatility clusters: |return| autocorrelation", xlabel="Lag (days)", ylabel="Mean autocorrelation")
    _save(fig, "volatility_clustering")
    s["abs_ret_autocorr_lag1"] = float(ac[0])
    s["abs_ret_autocorr_lag30"] = float(ac[29])

    # ---- 5. average volatility per coin ----
    pc = per_coin["daily_vol"].sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(pc.index, pc.values, color=style.BLUE)
    ax.grid(axis="y", visible=False)
    ax.set(title="Average daily volatility by coin", xlabel="Std of daily log return")
    _save(fig, "vol_by_coin")

    # ---- 6. correlations ----
    cols = ["target_vol", "vol_7", "vol_30", "atr_pct_14", "parkinson_14", "bb_width_20", "range_ma_7",
            "abs_ret_1", "log_volume", "volume_ratio_7", "liq_ratio", "amihud_14", "log_mcap", "ret_7", "rsi_14", "market_vol_7"]
    corr = feats[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else style.INK)
    fig.colorbar(im, fraction=0.04)
    ax.set_title("Spearman correlation of key features with next-7-day volatility")
    _save(fig, "correlations")
    tc = corr["target_vol"].drop("target_vol").sort_values(ascending=False)
    s["corr_with_target"] = tc.round(3).to_dict()

    # ---- 7. volume / liquidity vs volatility ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    samp = feats.sample(min(6000, len(feats)), random_state=1)
    axes[0].scatter(samp["liq_ratio"], samp["target_vol"], s=4, alpha=0.3, color=style.BLUE, edgecolors="none")
    axes[0].set_xscale("log")
    axes[0].set(title="Liquidity ratio vs future volatility", xlabel="Volume / market cap (log)", ylabel="Next-7-day volatility")
    axes[1].scatter(samp["volume_ratio_7"], samp["target_vol"], s=4, alpha=0.3, color=style.AMBER, edgecolors="none")
    axes[1].set(title="Volume surge vs future volatility", xlabel="Volume / 7-day average volume")
    fig.tight_layout()
    _save(fig, "liquidity")

    # ---- 8. weekday effect ----
    wd = clean.dropna(subset=["log_ret"]).assign(dow=lambda d: d["date"].dt.dayofweek).groupby("dow")["log_ret"].apply(
        lambda x: x.abs().mean())
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], wd.values, color=style.TEAL)
    ax.set(title="Average absolute return by weekday", ylabel="Mean |log return|")
    ax.set_ylim(wd.min() * 0.9, wd.max() * 1.05)
    _save(fig, "weekday")
    s["weekday_abs_ret"] = wd.round(5).tolist()

    # ---- 9. target distribution ----
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(feats["target_vol"].dropna(), bins=80, color=style.BLUE, alpha=0.85)
    ax.set(title="Target: next-7-day volatility", xlabel="Volatility", ylabel="Rows")
    _save(fig, "target_distribution")
    t = feats["target_vol"].dropna()
    s["target"] = {"mean": float(t.mean()), "median": float(t.median()), "std": float(t.std()),
                   "skew": float(stats.skew(t)), "p05": float(t.quantile(.05)), "p95": float(t.quantile(.95))}
    s["persistence_corr_vol7_target"] = float(feats[["vol_7", "target_vol"]].corr().iloc[0, 1])

    s["n_rows"] = int(len(clean))
    s["n_coins"] = int(clean["symbol"].nunique())
    s["date_min"], s["date_max"] = str(clean["date"].min().date()), str(clean["date"].max().date())
    s["is_demo"] = data_loader.is_demo_data()
    (config.REPORT_DIR / "eda_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s
