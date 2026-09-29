# Feature Guide

Every feature is computed from data available on day *t* or earlier. The target is the volatility (std-dev of daily log returns) over days *t+1 to t+7*.


## Returns

| Feature | Meaning |
|---|---|
| `ret_1` | Log return over the last day. |
| `abs_ret_1` | Absolute size of the last day's move; a direct shock signal. |
| `ret_3` | Log return over 3 days. |
| `ret_7` | Log return over 7 days. |
| `ret_30` | Log return over 30 days. |

## Rolling volatility

| Feature | Meaning |
|---|---|
| `vol_3` | Std-dev of daily log returns over 3 days (very short-term). |
| `vol_7` | Same over 7 days; also the naive baseline forecast. |
| `vol_14` | Same over 14 days. |
| `vol_30` | Same over 30 days. |
| `vol_90` | Same over 90 days; a coin's long-run volatility level. |
| `vol_ratio_7_30` | Short vs medium volatility; above 1 means volatility is rising. |
| `vol_ratio_3_14` | Very short vs 2-week volatility; catches sudden changes. |
| `downside_vol_14` | Volatility of negative returns only (14 days). |

## Range-based volatility

| Feature | Meaning |
|---|---|
| `range_hl` | (High - Low) / Close for the day; intraday range. |
| `range_ma_7` | 7-day average of the intraday range. |
| `range_ma_14` | 14-day average of the intraday range. |
| `parkinson_14` | Parkinson estimator: volatility from High/Low, more efficient than close-to-close. |
| `garman_klass_14` | Garman-Klass estimator: volatility using Open, High, Low and Close. |
| `atr_pct_7` | Average True Range over 7 days as a share of price. |
| `atr_pct_14` | Average True Range over 14 days as a share of price. |

## Bollinger Bands

| Feature | Meaning |
|---|---|
| `bb_width_20` | Bollinger Band width: (upper - lower) / 20-day mean; wide bands mean high volatility. |
| `bb_pctb_20` | %B: where the close sits inside the Bollinger Bands (0 = lower, 1 = upper). |

## Trend and momentum

| Feature | Meaning |
|---|---|
| `sma_ratio_7` | Close relative to its 7-day average. |
| `sma_ratio_30` | Close relative to its 30-day average. |
| `ema_ratio_12_26` | 12-day EMA over 26-day EMA (MACD-style trend signal). |
| `rsi_14` | Relative Strength Index: overbought / oversold momentum. |

## Volume and liquidity

| Feature | Meaning |
|---|---|
| `log_volume` | Log of daily traded volume. |
| `volume_chg_1` | 1-day change in log volume. |
| `volume_ratio_7` | Volume vs its 7-day average; surges often precede big moves. |
| `volume_ratio_30` | Volume vs its 30-day average. |
| `volume_std_14` | How erratic volume has been (14 days). |
| `liq_ratio` | Volume / market cap; how much of the coin trades each day. |
| `liq_ratio_ma_7` | 7-day average liquidity ratio. |
| `liq_ratio_chg_7` | Change in liquidity ratio vs 7 days ago. |
| `amihud_14` | Amihud illiquidity: price move per unit of volume (14-day mean, log scaled). |

## Size and history

| Feature | Meaning |
|---|---|
| `log_mcap` | Log market capitalisation (size). |
| `log_age_days` | Log of days since the coin's first record in the data. |

## Calendar

| Feature | Meaning |
|---|---|
| `dow_sin` | Day of week, sine encoding. |
| `dow_cos` | Day of week, cosine encoding. |

## Market-wide

| Feature | Meaning |
|---|---|
| `market_vol_7` | Average 7-day volatility across all coins on that date (market stress). |
| `market_ret_7` | Average 7-day return across all coins (market direction). |
| `rel_vol_7` | The coin's 7-day volatility relative to the market average. |
