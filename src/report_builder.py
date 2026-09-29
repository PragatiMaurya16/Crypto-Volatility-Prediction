"""Fills the EDA report, feature guide and final report with the real numbers
produced by the pipeline, so the documents never drift from the results."""
from __future__ import annotations

import json

import pandas as pd

from . import config, features as F

FEATURE_DOCS = {
    "ret_1": "Log return over the last day.",
    "abs_ret_1": "Absolute size of the last day's move; a direct shock signal.",
    "ret_3": "Log return over 3 days.", "ret_7": "Log return over 7 days.", "ret_30": "Log return over 30 days.",
    "vol_3": "Std-dev of daily log returns over 3 days (very short-term).",
    "vol_7": "Same over 7 days; also the naive baseline forecast.",
    "vol_14": "Same over 14 days.", "vol_30": "Same over 30 days.",
    "vol_90": "Same over 90 days; a coin's long-run volatility level.",
    "vol_ratio_7_30": "Short vs medium volatility; above 1 means volatility is rising.",
    "vol_ratio_3_14": "Very short vs 2-week volatility; catches sudden changes.",
    "downside_vol_14": "Volatility of negative returns only (14 days).",
    "range_hl": "(High - Low) / Close for the day; intraday range.",
    "range_ma_7": "7-day average of the intraday range.", "range_ma_14": "14-day average of the intraday range.",
    "parkinson_14": "Parkinson estimator: volatility from High/Low, more efficient than close-to-close.",
    "garman_klass_14": "Garman-Klass estimator: volatility using Open, High, Low and Close.",
    "atr_pct_7": "Average True Range over 7 days as a share of price.",
    "atr_pct_14": "Average True Range over 14 days as a share of price.",
    "bb_width_20": "Bollinger Band width: (upper - lower) / 20-day mean; wide bands mean high volatility.",
    "bb_pctb_20": "%B: where the close sits inside the Bollinger Bands (0 = lower, 1 = upper).",
    "sma_ratio_7": "Close relative to its 7-day average.", "sma_ratio_30": "Close relative to its 30-day average.",
    "ema_ratio_12_26": "12-day EMA over 26-day EMA (MACD-style trend signal).",
    "rsi_14": "Relative Strength Index: overbought / oversold momentum.",
    "log_volume": "Log of daily traded volume.", "volume_chg_1": "1-day change in log volume.",
    "volume_ratio_7": "Volume vs its 7-day average; surges often precede big moves.",
    "volume_ratio_30": "Volume vs its 30-day average.", "volume_std_14": "How erratic volume has been (14 days).",
    "liq_ratio": "Volume / market cap; how much of the coin trades each day.",
    "liq_ratio_ma_7": "7-day average liquidity ratio.", "liq_ratio_chg_7": "Change in liquidity ratio vs 7 days ago.",
    "amihud_14": "Amihud illiquidity: price move per unit of volume (14-day mean, log scaled).",
    "log_mcap": "Log market capitalisation (size).", "log_age_days": "Log of days since the coin's first record in the data.",
    "dow_sin": "Day of week, sine encoding.", "dow_cos": "Day of week, cosine encoding.",
    "market_vol_7": "Average 7-day volatility across all coins on that date (market stress).",
    "market_ret_7": "Average 7-day return across all coins (market direction).",
    "rel_vol_7": "The coin's 7-day volatility relative to the market average.",
}


def _demo_note(is_demo: bool) -> str:
    if not is_demo:
        return ""
    return ("> **Note:** the figures in this document were produced from the bundled synthetic demo dataset "
            "(fictional tickers), because the Kaggle file is not included in the repository. Put the real "
            "CSV files in `data/raw/` and run `python run_pipeline.py` to regenerate every number, table and "
            "chart from the real data.\n\n")


def _md_table(df: pd.DataFrame, fmt="{:.4f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [fmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build():
    m = json.loads((config.REPORT_DIR / "metrics.json").read_text())
    e = json.loads((config.REPORT_DIR / "eda_summary.json").read_text())
    c = e["cleaning"]
    demo = _demo_note(e.get("is_demo", False))
    imp = pd.read_csv(config.REPORT_DIR / "feature_importance.csv")
    per_coin = pd.read_csv(config.REPORT_DIR / "per_coin_metrics.csv", index_col=0)

    # ---------------- feature guide ----------------
    lines = ["# Feature Guide\n", "Every feature is computed from data available on day *t* or earlier. "
             "The target is the volatility (std-dev of daily log returns) over days *t+1 to t+7*.\n"]
    for group, feats in F.FEATURE_GROUPS.items():
        lines += [f"\n## {group}\n", "| Feature | Meaning |", "|---|---|"]
        lines += [f"| `{f}` | {FEATURE_DOCS[f]} |" for f in feats]
    (config.DOCS_DIR / "FEATURES.md").write_text("\n".join(lines) + "\n")

    # ---------------- EDA report ----------------
    r, t, mv = e["returns"], e["target"], e["market_vol"]
    corr = e["corr_with_target"]
    top_corr = list(corr.items())[:4]
    wd = e["weekday_abs_ret"]
    eda = f"""# Exploratory Data Analysis Report

{demo}## 1. Dataset summary

| Item | Value |
|---|---|
| Cryptocurrencies | {e['n_coins']} |
| Daily records (after cleaning) | {e['n_rows']:,} |
| Period | {e['date_min']} to {e['date_max']} |
| Columns | date, symbol, open, high, low, close, volume, market cap |
| Model-ready rows (after feature windows) | see `data/processed/features.csv.gz` |

Full per-column statistics are in `reports/eda_summary_stats.csv`; per-coin statistics are in
`reports/eda_per_coin_stats.csv`.

## 2. Data quality and cleaning

| Check | Result |
|---|---|
| Rows read | {c['rows_in']:,} |
| Duplicate (coin, date) rows removed | {c['duplicates_removed']:,} |
| Non-positive prices set to missing | {c['nonpositive_open'] + c['nonpositive_high'] + c['nonpositive_low'] + c['nonpositive_close']:,} |
| Rows with inconsistent High/Low repaired | {c['ohlc_rows_repaired']:,} |
| Missing cells filled (forward fill, max {config.MAX_FFILL_DAYS} days) | {c['missing_cells_forward_filled']:,} |
| Rows dropped (gap longer than {config.MAX_FFILL_DAYS} days) | {c['rows_dropped_unfillable_gaps']:,} |
| Coins dropped (< {config.MIN_HISTORY_DAYS} days of history) | {c['coins_dropped_short_history']} |
| Rows kept | {c['rows_out']:,} |

## 3. Price behaviour

![prices](../reports/figures/eda_prices.png)

Prices span several orders of magnitude between coins, so raw price levels are not comparable. All
volatility measures in this project are therefore built from **log returns** and price ratios, never
from price levels.

## 4. Returns are fat-tailed

![returns](../reports/figures/eda_return_distribution.png)

Daily log returns have a standard deviation of {r['std']:.3f}, an excess kurtosis of {r['excess_kurtosis']:.1f} and
a skew of {r['skew']:.2f}. {r['share_abs_gt_10pct']:.1%} of all coin-days move more than 10%. A normal distribution
badly underestimates these extremes, which is why we predict volatility with flexible models and evaluate
on RMSE and MAE rather than assuming Gaussian errors.

## 5. Volatility clusters and is persistent

![clustering](../reports/figures/eda_volatility_clustering.png)

The average autocorrelation of absolute returns is {e['abs_ret_autocorr_lag1']:.2f} at lag 1 and still
{e['abs_ret_autocorr_lag30']:.2f} at lag 30. Calm periods are followed by calm periods and turbulent
periods by turbulent ones. The correlation between the last 7 days' volatility and the next 7 days' volatility
is {e['persistence_corr_vol7_target']:.2f}, which is why a "same as last week" forecast is a strong baseline that
the model has to beat.

![market](../reports/figures/eda_market_volatility.png)

Market-wide volatility peaked on {mv['peak_date']} (average 7-day volatility {mv['peak']:.3f}, versus a median of
{mv['median']:.3f}). Coins tend to become volatile together, so the model gets a market-wide volatility feature.

## 6. Differences between coins

![by coin](../reports/figures/eda_vol_by_coin.png)

Highest average daily volatility: {', '.join(f'{k} ({v:.3f})' for k, v in e['per_coin_top_vol'].items())}.
Lowest: {', '.join(f'{k} ({v:.3f})' for k, v in e['per_coin_low_vol'].items())}.

## 7. Correlations

![corr](../reports/figures/eda_correlations.png)

Features most correlated (Spearman) with next-7-day volatility: {', '.join(f'`{k}` ({v:.2f})' for k, v in top_corr)}.
The volatility estimators are strongly correlated with each other, which is fine for tree models and is
handled by regularisation in the linear model.

## 8. Volume and liquidity

![liquidity](../reports/figures/eda_liquidity.png)

Liquidity ratio (volume / market cap) and volume surges are related to future volatility but much more
weakly than past volatility itself. They add information on top of the volatility features rather than replacing them.

## 9. Weekday effect

![weekday](../reports/figures/eda_weekday.png)

Average absolute return by weekday (Mon to Sun): {', '.join(f'{x:.4f}' for x in wd)}. Weekday is included as a
cyclical feature.

## 10. The target

![target](../reports/figures/eda_target_distribution.png)

Next-7-day volatility has a mean of {t['mean']:.4f}, a median of {t['median']:.4f} and is right-skewed
(skew {t['skew']:.2f}; 5th to 95th percentile: {t['p05']:.3f} to {t['p95']:.3f}). Because of the skew, the models
are trained on log(volatility) and converted back, so calm periods are not drowned out by spikes.

## 11. Takeaways

1. Volatility is persistent and clustered, so recent volatility is the strongest signal.
2. Returns are fat-tailed, so errors are judged on the original volatility scale (RMSE, MAE, R2).
3. Range-based estimators (Parkinson, Garman-Klass, ATR) use the intraday range and often carry information that close-to-close volatility misses.
4. Market-wide stress matters, so cross-coin features are included.
5. Chronological splitting is mandatory: shuffling would leak future volatility into training.
"""
    (config.DOCS_DIR / "EDA_REPORT.md").write_text(eda)

    # ---------------- final report ----------------
    tm = pd.DataFrame(m["test_metrics"]).T
    tm.index.name = "Model"
    tm = tm.sort_values("RMSE")
    best = m["model_name"]
    naive = m["test_metrics"]["Naive (last 7 days)"]
    b = m["test_metrics"][best]
    gain = (naive["RMSE"] - b["RMSE"]) / naive["RMSE"]
    reg = m["regime_test"]
    sp = m["split"]
    top10 = ", ".join(f"`{x}`" for x in m["top_features"][:8])
    pc_best, pc_worst = per_coin.head(3), per_coin.tail(3)
    params = ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                       for k, v in m["best_params"][best].items())
    verdict = ("beats" if gain > 0 else "does not beat")
    med_r2 = float(per_coin["R2"].median())
    n_pos = int((per_coin["R2"] > 0).sum())
    hi_err = ", ".join(per_coin.tail(3).index)
    lo_vol = ", ".join(per_coin.head(3).index)
    persist = e["persistence_corr_vol7_target"]
    final = f"""# Final Report: Cryptocurrency Volatility Prediction

{demo}## 1. Objective

Forecast how volatile each cryptocurrency will be over the next {m['horizon_days']} days using only
information available today (OHLC prices, volume and market capitalisation). Volatility is measured as the standard
deviation of daily log returns. The forecast supports risk management, position sizing and spotting turbulent periods early.

## 2. Data

{e['n_coins']} coins, {e['n_rows']:,} daily records, {e['date_min']} to {e['date_max']}. Cleaning removed duplicates,
repaired inconsistent High/Low values, forward-filled short gaps and dropped coins with too little history
(details in the EDA report).

## 3. Method

![pipeline](../reports/figures/diagram_pipeline.png)

**Features.** {m['n_features']} engineered features in nine groups: returns, rolling volatility, range-based volatility
(Parkinson, Garman-Klass, ATR), Bollinger Bands, trend and momentum, volume and liquidity, size, calendar and
market-wide context. See `docs/FEATURES.md`.

**Target.** Realised volatility over days t+1 to t+{m['horizon_days']}.

**Preprocessing.** Standard scaling (fitted on training data only), clipping at 8 standard deviations, and a log transform
of the target.

**Splitting.** Strictly chronological, with a {config.GAP_DAYS}-day gap before each boundary so no training target overlaps a
later period.

| Set | From | To | Rows |
|---|---|---|---|
| Train | {sp['train'][0]} | {sp['train'][1]} | {sp['train'][2]:,} |
| Validation | {sp['validation'][0]} | {sp['validation'][1]} | {sp['validation'][2]:,} |
| Test | {sp['test'][0]} | {sp['test'][1]} | {sp['test'][2]:,} |

**Models.** Ridge regression, Random Forest and Gradient Boosting, each tuned with randomised search and forward-chaining
cross-validation on the training set, compared against a naive baseline (next week's volatility = last 7 days' volatility). The
winner was picked on the validation set. All candidates were then refitted on train + validation and scored **once** on the untouched test set.

## 4. Results

Selected model: **{best}** ({params}).

{_md_table(tm[['RMSE','MAE','R2']], '{:.4f}')}

![comparison](../reports/figures/model_comparison.png)

On the hold-out test period, {best} has an RMSE of {b['RMSE']:.4f}, an MAE of {b['MAE']:.4f} and an R2 of {b['R2']:.3f}.
Compared with the naive baseline (RMSE {naive['RMSE']:.4f}), it {verdict} the baseline, with RMSE
{'reduced' if gain > 0 else 'changed'} by {abs(gain):.1%}.

![forecast](../reports/figures/forecast_timeseries.png)

![scatter](../reports/figures/pred_vs_actual.png) ![residuals](../reports/figures/residuals.png)

### Volatility regimes

Forecasts were also bucketed into Low, Medium and High volatility using training-set terciles
(thresholds {m['regime_thresholds'][0]:.4f} and {m['regime_thresholds'][1]:.4f}). The model puts a coin in the
correct regime {reg['accuracy']:.1%} of the time (random guessing would give about 33%).

![regimes](../reports/figures/regime_confusion.png)

### What drives the forecast

![importance](../reports/figures/feature_importance.png)

Permutation importance on the test set ranks {top10} highest (features that raise RMSE most when shuffled).

### Accuracy by coin

Lowest RMSE: {', '.join(f'{i} ({v:.4f})' for i, v in pc_best['RMSE'].items())}.
Highest RMSE: {', '.join(f'{i} ({v:.4f})' for i, v in pc_worst['RMSE'].items())}.
Full table: `reports/per_coin_metrics.csv`.

**Read the R2 carefully.** The pooled R2 above ({b['R2']:.2f}) mixes two things: telling calm coins from wild coins, and
timing when a single coin gets calmer or wilder. Inside individual coins the median R2 is {med_r2:.2f} and it is positive for
{n_pos} of {len(per_coin)} coins. So a large part of the pooled score comes from ranking coins by their usual volatility level, and the
week-to-week timing signal is smaller. RMSE and MAE per coin are the fairer view of within-coin accuracy.

## 5. Key insights

1. Volatility is persistent and clustered (correlation between last week's and next week's volatility: {persist:.2f}), which is what makes it forecastable at all, unlike returns.
2. The naive "same as last week" forecast is the bar to beat. {best} {verdict} it, with RMSE {'lower' if gain > 0 else 'different'} by {abs(gain):.1%}. The gain likely comes from blending several volatility estimators with volume, liquidity and market-wide context, and from shrinking noisy 7-day readings toward each coin's longer-run level.
3. Errors are largest for coins with the highest volatility ({hi_err}) and, as with any price-based model, around sudden shocks that history cannot anticipate.
4. Volatility level (Low / Medium / High) is coarser than an exact forecast, and is often the more practical output for risk monitoring. Accuracy here is {reg['accuracy']:.0%} against about 33% for guessing.

## 6. Limitations

- Forecasts use only price, volume and market cap. News, regulation, on-chain data and macro events are not included.
- Daily data cannot capture intraday volatility.
- Sudden shocks (exchange failures, regulatory news) are unpredictable from history.
- The model was evaluated on one hold-out period; performance in a very different market regime may differ.
- Nothing here is financial advice.

## 7. Possible next steps

Add a GARCH benchmark and an LSTM or temporal CNN, include sentiment and on-chain features, forecast several horizons (1, 7, 30 days),
and predict intervals (quantile regression) instead of a single number.

## 8. Reproducing

```
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```
"""
    (config.DOCS_DIR / "FINAL_REPORT.md").write_text(final)


if __name__ == "__main__":
    build()
