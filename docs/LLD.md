# Low-Level Design (LLD)

How each component is implemented. All paths are relative to the project root.

## 1. `src/config.py`

Holds every setting: folder paths, forecast horizon (`HORIZON = 7`), cleaning limits, split fractions, regime quantiles and the random seed.

## 2. `src/data_loader.py`

| Function | Behaviour |
|---|---|
| `_normalise(df, fallback_symbol)` | Lower-cases column names, maps aliases (`Marketcap` -> `market_cap`, `Ticker` -> `symbol` ...), parses dates, converts numbers, returns the 9 standard columns. Raises a clear error listing missing columns. |
| `load_raw(path=None)` | Reads one CSV or every CSV in `data/raw/`. Real files take priority over the bundled `demo_*.csv`. Works with the Kaggle per-coin layout (`coin_Bitcoin.csv` ...) or one combined file. |
| `is_demo_data()` | True when only the demo file is present. Saved in the model metadata so the app can show a banner. |

## 3. `src/preprocessing.py`

`clean(df) -> (df, report)` runs these steps in order:

1. Drop rows without a date or symbol.
2. Remove duplicate (symbol, date) rows, keeping the last.
3. Turn non-positive prices, volume and market cap into missing values.
4. Repair OHLC consistency: `high = max(O,H,L,C)`, `low = min(O,H,L,C)`.
5. For each coin: reindex to a continuous daily calendar and forward-fill up to `MAX_FFILL_DAYS` (3). Longer gaps stay missing and those rows are dropped.
6. Drop coins with fewer than `MIN_HISTORY_DAYS` (120) rows.

The returned `report` dictionary counts every change and is written to `reports/cleaning_report.json`.

## 4. `src/features.py`

`build_features(df)` calls `_coin_features(g)` for each coin and then adds market-wide columns.

Inside `_coin_features`:

- The coin is reindexed to a daily calendar so a "7-day window" always means 7 real days.
- Rolling statistics use `min_periods = window`, so rows without a full window become missing and are dropped at the end.
- Target: `r.rolling(7).std().shift(-7)`, which is the std of returns on days t+1 to t+7. Rows in the last 7 days of each coin keep a missing target; they are used for live forecasts but not for training.
- Formulas: Parkinson `sqrt(mean(ln(H/L)^2) / (4 ln 2))`; Garman-Klass `0.5 ln(H/L)^2 - (2 ln2 - 1) ln(C/O)^2`; ATR uses the true range `max(H-L, |H-C_prev|, |L-C_prev|)`; Bollinger Bands use a 20-day mean +/- 2 standard deviations; RSI uses Wilder smoothing (alpha = 1/14).
- Market-wide columns: for each date, the mean of `vol_7` and `ret_7` over all coins, plus the coin's `vol_7` relative to that mean.

`FEATURE_GROUPS` and `FEATURES` are the single source of truth for the model, the docs and the app. The full list is in `FEATURES.md`.

## 5. `src/modeling.py`

| Item | Detail |
|---|---|
| `make_pipeline(est)` | `TransformedTargetRegressor(Pipeline[StandardScaler -> clip(-8, 8) -> est], func=log, inverse=exp)` |
| `candidates()` | Ridge, Random Forest, HistGradientBoosting, each with a search space |
| `chronological_split(df)` | Sorts unique dates, cuts at 70% / 80%, and removes `GAP_DAYS` before each boundary from the earlier set |
| `tune(name, n_coins)` | `RandomizedSearchCV` with `TimeSeriesSplit` (gap = horizon x number of coins, so no fold's targets overlap the next fold); scoring is negative RMSE |
| `N_ITER` | Number of search iterations per model |

## 6. `src/train.py`

1. Load `features.csv.gz`, keep rows with a target, sort by date.
2. Split with `chronological_split`.
3. For each candidate: tune on train, refit with the best parameters, score on validation.
4. Pick the best on validation RMSE.
5. Refit every candidate on train + validation and score once on test (with the naive baseline).
6. Build regime thresholds (33rd and 66th percentile of the training target), permutation importance and all plots.
7. Save `models/volatility_model.joblib`, `models/metadata.json`, `reports/metrics.json`, test predictions and per-coin metrics.

## 7. `src/evaluate.py`

`regression_metrics` (RMSE, MAE, R2), `regime_report` (accuracy and confusion matrix on Low / Medium / High), `feature_importance`
(permutation importance, 3 repeats on up to 4,000 test rows) and all figure functions.

## 8. `src/eda.py`

Computes dataset statistics and saves the exploratory figures (`reports/figures/eda_*.png`) and `eda_summary.json`.

## 9. `src/report_builder.py`

Reads the JSON / CSV outputs and writes `docs/EDA_REPORT.md`, `docs/FEATURES.md` and `docs/FINAL_REPORT.md` so the text always matches the numbers.

## 10. `app/streamlit_app.py`

| Part | Behaviour |
|---|---|
| Loading | `st.cache_resource` for the model, `st.cache_data` for features and predictions |
| Sidebar | Pick a coin from the dataset, or upload a CSV for one coin (cleaned and featurised by the same code as training) |
| Forecast tab | Headline forecast, Low / Medium / High level, a ruler comparing forecast with last week, a chart of past forecasts vs realised volatility, and a table of the signals behind it |
| Compare coins tab | Latest forecast for every coin, sortable, downloadable |
| Model quality tab | Test-set metrics and saved figures |
| About the data tab | Dataset facts and a plain-language definition of volatility |

Design notes: the forecast made on day t is drawn at day t+7, where it lines up with the volatility that was then realised; the hold-out start is marked so users can see where the model was not trained.
For uploaded files, market-wide volatility is taken from the latest date in the training data (a single coin has no cross-section).

## 11. Error handling

- Missing columns in a CSV: clear message naming the missing columns.
- Too little history: the app reports that about 150 days are needed.
- No trained model: the app tells the user to run `python run_pipeline.py`.
