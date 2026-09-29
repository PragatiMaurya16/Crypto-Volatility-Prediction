# Cryptocurrency Volatility Prediction

Predicts how volatile a cryptocurrency will be over the **next 7 days** from daily OHLC prices, trading volume and market
capitalisation, and serves the forecast in a Streamlit app.

Volatility here is the standard deviation of daily log returns. The model also labels each forecast as Low, Medium or High
compared with the coins it was trained on.

> **About the data in this repo.** The Kaggle file is not bundled, so `data/raw/demo_crypto_prices.csv` contains a synthetic
> dataset (fictional tickers) with the same columns. Every number and chart currently in `docs/` and `reports/` comes from that demo
> data. Follow "Use the real dataset" below and re-run the pipeline to replace them with real results.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py                                  # clean -> features -> EDA -> train -> reports (a few minutes)
streamlit run app/streamlit_app.py
```

`python run_pipeline.py --quick` does a fast smoke test with less tuning.

## Light and dark mode

Both themes are built in. Open the menu at the top right, choose **Settings**, then **Theme** (Light, Dark, or Use system setting).
The charts and colours follow the choice. The sidebar can be collapsed and re-opened with the arrow at the top left.

## Opening the app from another device (Network URL)

Streamlit already listens on all interfaces. If the `Network URL` works on the PC but not from a phone or another computer:

1. Both devices must be on the same Wi-Fi/LAN (guest networks and some campus/office Wi-Fi block device-to-device traffic).
2. Allow the port through the firewall. Windows (run PowerShell as Administrator):
   `netsh advfirewall firewall add rule name="Streamlit 8501" dir=in action=allow protocol=TCP localport=8501`
   When Windows asks whether Python may use the network, choose **Private networks**.
3. Start with an explicit address if needed: `streamlit run app/streamlit_app.py --server.address 0.0.0.0`
4. Use the IP shown as `Network URL` in the terminal (for example `http://192.168.1.20:8501`), not `localhost`.

To share outside your network, deploy to Streamlit Community Cloud (below).

## Use the real dataset

1. Download **Cryptocurrency Historical Prices** from Kaggle and unzip it.
2. Copy the CSV files (e.g. `coin_Bitcoin.csv`, `coin_Ethereum.csv`, ...) into `data/raw/`. A single combined CSV also works.
   Column names are matched flexibly (`Marketcap`, `Market Cap`, `market_cap` are all fine); the file needs date, open, high, low,
   close, volume, market cap and a symbol or name column.
3. Run `python run_pipeline.py`. Real files take priority over the demo file automatically.
4. Everything is regenerated: `models/`, `data/processed/`, `reports/` and the documents in `docs/`.

## Deploy on Streamlit Community Cloud

1. Run `python run_pipeline.py` locally first, so the saved model matches your scikit-learn version. Then push this folder to a GitHub repository, committing `models/` and `data/processed/features.csv.gz` (the app needs them).
2. On share.streamlit.io choose the repo, branch `main` and main file `app/streamlit_app.py`.
3. Streamlit installs from `requirements.txt` automatically.

## Deliverables and where they are

| Requirement from the brief | Location |
|---|---|
| Trained model + evaluation metrics (RMSE, MAE, R2) | `models/volatility_model.joblib`, `models/metadata.json`, `reports/metrics.json`, `docs/FINAL_REPORT.md` |
| Cleaned and prepared dataset | `data/processed/clean_data.csv.gz`, `data/processed/features.csv.gz` |
| Explanation of the new features | `docs/FEATURES.md` |
| EDA report (statistics, trends, correlations, distributions) | `docs/EDA_REPORT.md`, `reports/eda_*.csv`, `reports/figures/eda_*.png` |
| High-Level Design document | `docs/HLD.md` |
| Low-Level Design document | `docs/LLD.md` |
| Pipeline architecture | `docs/PIPELINE.md`, `reports/figures/diagram_pipeline.png` |
| Final report | `docs/FINAL_REPORT.md` |
| Well-commented source code | `src/`, `run_pipeline.py` |
| Local deployment (Streamlit) | `app/streamlit_app.py` |

## Project layout

```
app/streamlit_app.py      web app
src/config.py             all settings (horizon, split, paths)
src/data_loader.py        read Kaggle CSVs, normalise columns
src/preprocessing.py      cleaning
src/features.py           feature engineering + target
src/modeling.py           pipelines, chronological split, tuning
src/train.py              train, select, test, save
src/evaluate.py           metrics and plots
src/eda.py                exploratory analysis
src/report_builder.py     writes the reports from real results
src/diagrams.py           pipeline and architecture diagrams
src/generate_demo_data.py synthetic demo dataset
docs/                     HLD, LLD, pipeline, EDA report, feature guide, final report
reports/                  metrics, tables and figures
models/                   saved model and metadata
data/raw, data/processed  input and prepared data
```

## Method in brief

- **Target:** volatility over days t+1 to t+7, using only information up to day t.
- **Features (42):** returns, rolling volatility (3 to 90 days), Parkinson and Garman-Klass estimators, ATR, Bollinger Bands,
  moving-average ratios, RSI, volume and liquidity ratios (volume / market cap, Amihud), size, calendar and market-wide context.
- **Split:** by date into train, validation and test, with a 7-day gap so no target window overlaps a later period.
- **Models:** Ridge, Random Forest and Gradient Boosting, tuned with forward-chaining cross-validation and compared with a naive
  "same as last week" baseline. The winner is picked on validation data and scored once on the test data.

## Limitations

Only prices, volume and market cap are used. News, regulation and on-chain data are not. Daily data cannot see intraday swings, and
sudden shocks cannot be predicted from history. This is a statistical estimate and not financial advice.
