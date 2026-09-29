# Cryptocurrency Volatility Prediction

## Project Overview

The **Cryptocurrency Volatility Prediction** project is an end-to-end machine learning project that forecasts how volatile a cryptocurrency will be over the **next 7 days**, using historical daily prices, trading volume, and market capitalization.

This project combines **Python, Scikit-learn, and Streamlit** to clean data, engineer features, train and compare models, and serve the forecasts in an interactive web app.

The objective is to help traders, investors, and financial institutions anticipate periods of heightened volatility for better risk management, portfolio allocation, and trading decisions.

*Live App:
https://your-app-name.streamlit.app](https://crypto-volatility-prediction-07.streamlit.app/

---
# Project Highlights

- End-to-end machine learning workflow
- Data cleaning and preprocessing using Python
- Exploratory Data Analysis (EDA)
- **42 engineered features** (rolling volatility, ATR, Bollinger Bands, liquidity ratios, and more)
- Time-aware model training with no data leakage
- Comparison of 3 models against a naive baseline
- Volatility regime labels: Low, Medium, High
- Interactive Streamlit app with light and dark mode
- Analysis of **XX,XXX records** across **XX cryptocurrencies**

---

# Project Objectives

The main goals of this project are:

- Predict cryptocurrency volatility for the next 7 days
- Identify which coins are heading into turbulent periods
- Engineer features that capture volatility and liquidity trends
- Compare regression models and pick the best one on validation data
- Evaluate on unseen data using RMSE, MAE, and R² score
- Deploy the model in a simple app for testing predictions

---

# Dataset Description

The dataset contains daily historical records for multiple cryptocurrencies, including:

- Date
- Coin name / symbol
- Open, High, Low, Close prices
- Trading volume
- Market capitalization

**Source:** Cryptocurrency Historical Prices dataset (Kaggle)

The raw dataset was cleaned and transformed before building features.

**Target variable:** volatility, defined as the standard deviation of daily log returns over the next 7 days.

---

# Tech Stack

## Programming & Data Processing

- Python
- Pandas
- NumPy
- SciPy

## Machine Learning

- Scikit-learn (Ridge, Random Forest, Gradient Boosting)
- Time-series cross-validation
- Randomized hyperparameter search

## Visualization & Deployment

- Matplotlib
- Seaborn
- Plotly
- Streamlit

## Development Tools

- Git
- GitHub
- VS Code

---

# Project Structure

```
crypto-volatility-prediction/

│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   ├── PIPELINE.md
│   ├── EDA_REPORT.md
│   ├── FEATURES.md
│   └── FINAL_REPORT.md
│
├── models/
│   ├── volatility_model.joblib
│   └── metadata.json
│
├── reports/
│   └── figures/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modeling.py
│   ├── train.py
│   ├── evaluate.py
│   ├── eda.py
│   └── report_builder.py
│
├── .streamlit/
│   └── config.toml
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# Project Workflow

```
Raw Dataset (Kaggle CSVs)
      ↓
Data Cleaning using Python
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering (42 features + 7-day target)
      ↓
Chronological Train / Validation / Test Split
      ↓
Model Training & Hyperparameter Tuning
      ↓
Model Evaluation (RMSE, MAE, R²)
      ↓
Streamlit App Deployment
```

---

# Data Cleaning & Preparation

Performed data preprocessing steps including:

- Removing duplicate records
- Fixing inconsistent High / Low values
- Handling missing values (forward fill up to 3 days)
- Building a continuous daily calendar for each coin
- Dropping coins with too little history
- Scaling numerical features (fitted on training data only)

---

# Exploratory Data Analysis

Performed analysis to understand:

- Price trends across coins
- Return distributions and fat tails
- Volatility clustering and persistence
- Volatility differences between coins
- Correlations between features and future volatility
- Volume and liquidity effects
- Weekday effects

The full report is in `docs/EDA_REPORT.md`.

---

# Feature Engineering

Created features in nine groups:

- **Returns:** 1, 3, 7, and 30-day log returns
- **Rolling volatility:** 3, 7, 14, 30, and 90-day windows
- **Range-based volatility:** Parkinson, Garman-Klass, ATR
- **Bollinger Bands:** band width and %B
- **Trend and momentum:** moving-average ratios, EMA ratio, RSI
- **Volume and liquidity:** volume ratios, volume / market cap, Amihud illiquidity
- **Size and history:** log market cap, coin age
- **Calendar:** day-of-week encoding
- **Market-wide context:** average market volatility and returns

Every feature uses only data available up to day *t*, so there is no look-ahead. Details are in `docs/FEATURES.md`.

---

# Model Training & Evaluation

Three models were tuned with forward-chaining cross-validation and compared with a naive baseline (next week's volatility equals last 7 days' volatility):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | X.XXXX | X.XXXX | X.XX |
| Random Forest | X.XXXX | X.XXXX | X.XX |
| Gradient Boosting | X.XXXX | X.XXXX | X.XX |
| Naive baseline (last 7 days) | X.XXXX | X.XXXX | X.XX |

> Replace the values above with the results from `docs/FINAL_REPORT.md` after running the pipeline on the real dataset.

**How leakage is prevented:**

- Split by date, never shuffled
- A 7-day gap separates train, validation, and test sets
- Scaler is fitted on training data only
- The test set is scored once, after model selection

---

# Streamlit App

The interactive app includes:

## Tabs

- **Forecast:** expected daily swing, volatility level (Low / Medium / High), comparison with the last 7 days, and a chart of past forecasts vs realised volatility
- **Compare coins:** latest forecast for every coin, sortable and downloadable as CSV
- **Model quality:** test-set metrics and evaluation charts
- **About the data:** dataset summary and definitions

## Extra Features

- Upload your own CSV for a single coin and get a forecast
- Light and dark mode (menu → Settings → Theme)

---

# Key Insights

- Volatility is persistent and clustered: calm periods follow calm periods, and turbulent periods follow turbulent ones.
- Crypto returns have fat tails, so extreme moves are much more common than a normal distribution suggests.
- Recent volatility is the strongest signal, and range-based measures add extra information.
- Coins tend to become volatile together, so market-wide context improves forecasts.
- The best model beats the naive "same as last week" baseline by **XX%** in RMSE.

> Update these numbers from your real results.

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add the Dataset

Download the Kaggle dataset and place the CSV files inside:

```
data/raw/
```

---

## 5. Run the Pipeline

```bash
python run_pipeline.py
```

This cleans the data, builds features, runs EDA, trains the models, and writes the reports.

---

## 6. Launch the App

```bash
streamlit run app/streamlit_app.py
```

---

# Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Time-Series Modeling
- Machine Learning (Regression)
- Hyperparameter Tuning
- Model Evaluation
- Preventing Data Leakage
- Streamlit App Development
- Technical Documentation (HLD, LLD, Pipeline)

---

# Future Improvements

Possible improvements:

- Add a GARCH benchmark model
- Try deep learning models (LSTM, temporal CNN)
- Add news sentiment and on-chain data
- Forecast multiple horizons (1, 7, 30 days)
- Predict uncertainty intervals with quantile regression
- Connect live price APIs for automatic updates

---

# Disclaimer

This project is for educational purposes. Forecasts are statistical estimates and are not financial advice.

---

# Author

**Pragati Maurya**

B.Tech Student | Machine Learning & Data Science Enthusiast | Aspiring Python Developer & Data Scientist

Passionate about building data-driven solutions using Python, SQL, Machine Learning, and Data Visualization.

Interested in solving real-world problems through analytics and intelligent systems.

---
## Support

If you found this project useful, consider giving it a star!

It helps others discover the project and supports future development.
