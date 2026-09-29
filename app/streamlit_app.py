"""Streamlit front end for the volatility model.

Run locally:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from string import Template
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, data_loader, features as F, preprocessing

st.set_page_config(page_title="Volatility outlook", layout="wide", initial_sidebar_state="expanded")


def _is_dark() -> bool:
    """True when the viewer picked the dark theme (menu > Settings > Theme)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:  # older Streamlit or a test harness
        return False


DARK = _is_dark()
P = (
    dict(ink="#E6EAEE", muted="#A3ADB8", rule="#3A4552", grid="#2C3641", blue="#7FAEDB", amber="#E3A552",
         teal="#5FB39A", red="#DB7A70", grey="#8E99A6", banner_bg="#3A2F1E", banner_ink="#F0DDB8", panel="#F4F6F8")
    if DARK else
    dict(ink="#1B2430", muted="#4A5561", rule="#C9D0D8", grid="#DCE1E6", blue="#2F5D8A", amber="#C7791F",
         teal="#3E7C6B", red="#A8433A", grey="#8A94A0", banner_bg="#FBF1E3", banner_ink="#4A3A22", panel="#FFFFFF")
)
INK, BLUE, AMBER, TEAL, RED, GREY = P["ink"], P["blue"], P["amber"], P["teal"], P["red"], P["grey"]
REGIME_COLORS = {"Low": TEAL, "Medium": AMBER, "High": RED}

_CSS = Template(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'IBM Plex Sans', sans-serif; }
/* Hide only the footer and Deploy button. The header stays so the sidebar toggle and the
   menu (Settings > Theme) remain available. */
footer, [data-testid="stAppDeployButton"] { display: none; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1120px; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; font-weight: 600; letter-spacing: -0.01em; color: $ink; }
h1 { font-size: 2.05rem !important; margin-bottom: .1rem; }
h3 { font-size: 1.15rem !important; margin-top: 1.6rem; }
.lede { color: $muted; font-size: 1.0rem; max-width: 62ch; margin: 0 0 1.4rem 0; line-height: 1.5; }
.readout { display: flex; gap: 3.2rem; flex-wrap: wrap; padding: 1.1rem 0 1.3rem 0; border-top: 2px solid $ink; border-bottom: 1px solid $rule; margin-bottom: .8rem; }
.readout .label { font-size: .82rem; color: $muted; margin-bottom: .15rem; }
.readout .value { font-family: 'Source Serif 4', Georgia, serif; font-size: 2.5rem; line-height: 1.05; font-variant-numeric: tabular-nums; color: $ink; }
.readout .note { font-size: .82rem; color: $muted; margin-top: .2rem; max-width: 26ch; }
.banner { background: $banner_bg; border-left: 3px solid $amber; padding: .65rem .9rem; font-size: .9rem; margin-bottom: 1.2rem; color: $banner_ink; }
.small { font-size: .82rem; color: $muted; }
[data-testid="stSidebar"] { border-right: 1px solid $rule; }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid $rule; }
.stTabs [data-baseweb="tab"] { padding: .5rem 0; height: auto; }
[data-testid="stImage"] img { background: $panel; border-radius: 6px; padding: 6px; }
</style>
"""
)
st.markdown(_CSS.substitute(P), unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner=False)
def load_model():
    path = config.MODEL_DIR / "volatility_model.joblib"
    if not path.exists():
        return None, None
    return joblib.load(path), json.loads((config.MODEL_DIR / "metadata.json").read_text())


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "features.csv.gz"
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def score(df: pd.DataFrame) -> pd.DataFrame:
    model, _ = load_model()
    out = df.copy()
    out["pred"] = model.predict(out[F.FEATURES])
    return out


def regime_of(value: float, thresholds, labels) -> str:
    return labels[int(np.digitize(value, thresholds))]


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


model, meta = load_model()
feats = load_features()
if model is None or feats.empty:
    st.title("Volatility outlook")
    st.error("No trained model found. Run `python run_pipeline.py` from the project folder, then reload this page.")
    st.stop()

thr, labels = meta["regime_thresholds"], meta["regime_labels"]
scored = score(feats)
latest_all = scored.sort_values("date").groupby("symbol").tail(1).set_index("symbol")
market_latest = float(latest_all["market_vol_7"].iloc[-1])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Choose a coin")
    source = st.radio("Data", ["Coins in the dataset", "Upload my own CSV"], label_visibility="collapsed")
    upload_frame = None
    if source == "Coins in the dataset":
        symbols = sorted(latest_all.index)
        default = "BTC" if "BTC" in symbols else symbols[0]
        symbol = st.selectbox("Coin", symbols, index=symbols.index(default))
    else:
        up = st.file_uploader("Daily prices for one coin (CSV)", type="csv")
        st.markdown(
            "<p class='small'>Needs columns: date, open, high, low, close, volume, market cap. "
            "At least 150 days of history.</p>", unsafe_allow_html=True)
        symbol = None
        if up is not None:
            try:
                raw = data_loader._normalise(pd.read_csv(up), fallback_symbol="UPLOAD")
                raw["symbol"] = "UPLOAD"
                clean, _ = preprocessing.clean(raw)
                f = F.build_features(clean)
                if f.empty:
                    raise ValueError("Not enough clean history to build features (need about 150 days).")
                f["market_vol_7"] = market_latest  # market context comes from the training dataset
                f["rel_vol_7"] = f["vol_7"] / market_latest
                upload_frame = score(f)
                symbol = "UPLOAD"
            except Exception as exc:  # show a readable message, not a stack trace
                st.error(f"Could not read that file: {exc}")
    st.markdown("---")
    st.markdown("<p class='small'>Light or dark: open the menu at the top right, then Settings, then Theme.</p>",
                unsafe_allow_html=True)
    st.markdown(f"<p class='small'>Model: {meta['model_name']}<br>Forecast horizon: {meta['horizon_days']} days<br>"
                f"Trained on data up to {meta['split']['test'][0]}</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------ header
st.title("Volatility outlook")
st.markdown(
    "<p class='lede'>How large price swings are likely to be over the next "
    f"{meta['horizon_days']} days, estimated from daily prices, trading volume and market cap.</p>",
    unsafe_allow_html=True)
if meta.get("is_demo"):
    st.markdown("<div class='banner'>This app is running on a synthetic demo dataset with fictional tickers. "
                "Run the pipeline on the real Kaggle files to see real coins.</div>", unsafe_allow_html=True)

tab_fc, tab_cmp, tab_model, tab_data = st.tabs(["Forecast", "Compare coins", "Model quality", "About the data"])

# ------------------------------------------------------------------ forecast
with tab_fc:
    if symbol is None:
        st.info("Upload a CSV in the sidebar to get a forecast for your own coin.")
    else:
        hist = (upload_frame if symbol == "UPLOAD" else scored[scored["symbol"] == symbol]).sort_values("date")
        row = hist.iloc[-1]
        pred, last = float(row["pred"]), float(row["vol_7"])
        reg = regime_of(pred, thr, labels)
        change = pred / last - 1
        wk = pred * np.sqrt(meta["horizon_days"])
        arrow = "higher" if change > 0.02 else "lower" if change < -0.02 else "about the same"

        st.markdown(
            f"""
<div class="readout">
  <div><div class="label">Expected daily swing</div>
       <div class="value">{pct(pred)}</div>
       <div class="note">Typical size of a daily move over the next {meta['horizon_days']} days (one standard deviation).</div></div>
  <div><div class="label">Volatility level</div>
       <div class="value" style="color:{REGIME_COLORS[reg]}">{reg}</div>
       <div class="note">Compared with all coins in the training data.</div></div>
  <div><div class="label">Versus the last 7 days</div>
       <div class="value">{change:+.0%}</div>
       <div class="note">Last week's realised swing was {pct(last)}, so the outlook is {arrow}.</div></div>
</div>
<p class="small">As of {row['date'].date()}. A one-week move of roughly ±{pct(wk, 0)} would be a normal outcome at this level.</p>
""",
            unsafe_allow_html=True)

        # regime ruler
        top = max(thr[1] * 2.0, pred * 1.25, last * 1.25)
        fig = go.Figure()
        for (a, b, name) in [(0, thr[0], "Low"), (thr[0], thr[1], "Medium"), (thr[1], top, "High")]:
            fig.add_shape(type="rect", x0=a, x1=b, y0=0, y1=1, fillcolor=REGIME_COLORS[name], opacity=0.22, line_width=0)
            fig.add_annotation(x=(a + b) / 2, y=0.5, text=name, showarrow=False, font=dict(size=12, color=INK))
        fig.add_trace(go.Scatter(x=[last], y=[0.13], mode="markers", marker=dict(symbol="line-ns", size=20, line=dict(width=2, color=GREY)),
                                 name="Last 7 days", hovertemplate="Last 7 days: %{x:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[pred], y=[0.87], mode="markers", marker=dict(symbol="diamond", size=13, color=INK),
                                 name="Forecast", hovertemplate="Forecast: %{x:.1%}<extra></extra>"))
        fig.update_layout(height=120, margin=dict(l=0, r=0, t=6, b=0), showlegend=True,
                          legend=dict(orientation="h", y=-0.35, x=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0, top], tickformat=".0%", showgrid=False), yaxis=dict(visible=False, range=[0, 1]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # history: forecast made 7 days earlier lines up with the volatility that was then realised
        st.markdown("### How past forecasts compared with what happened")
        span = st.select_slider("Window", options=[90, 180, 365, 730], value=180, format_func=lambda d: f"Last {d} days")
        h = hist.tail(span + meta["horizon_days"]).copy()
        h["forecast_for"] = h["date"] + pd.Timedelta(days=meta["horizon_days"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["date"], y=h["vol_7"], name="Realised (trailing 7 days)", line=dict(color=INK, width=1.6)))
        fig.add_trace(go.Scatter(x=h["forecast_for"], y=h["pred"], name="Forecast made 7 days earlier", line=dict(color=AMBER, width=1.6)))
        test_start = pd.Timestamp(meta["split"]["test"][0])
        if symbol != "UPLOAD" and h["date"].min() < test_start < h["forecast_for"].max():
            fig.add_vline(x=test_start, line_dash="dot", line_color=GREY)
            fig.add_annotation(x=test_start, y=1, yref="paper", text="Hold-out period starts", showarrow=False,
                               xanchor="left", yanchor="top", font=dict(size=11, color=GREY))
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12, x=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                          yaxis=dict(tickformat=".0%", gridcolor=P["grid"], title=None), xaxis=dict(gridcolor=P["grid"], title=None))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("<p class='small'>Before the dotted line the model had seen this data while training, so the fit looks "
                    "better there than it will in practice. Judge accuracy on the part after the line.</p>", unsafe_allow_html=True)

        # signals
        st.markdown("### Signals behind the forecast")
        sig = [("vol_7", "Volatility, last 7 days", True), ("vol_30", "Volatility, last 30 days", True),
               ("vol_90", "Volatility, last 90 days", True), ("volume_ratio_7", "Volume vs 7-day average", False),
               ("liq_ratio", "Volume as share of market cap", True), ("market_vol_7", "Market-wide volatility", True)]
        rows = []
        for key, label, is_pct in sig:
            now, med = float(row[key]), float(hist[key].median())
            rank = float((hist[key] <= now).mean())
            fmt = (lambda v: pct(v, 2)) if is_pct else (lambda v: f"{v:.2f}x")
            reading = "High for this coin" if rank > 0.8 else "Low for this coin" if rank < 0.2 else "Normal"
            rows.append({"Signal": label, "Now": fmt(now), "Typical": fmt(med), "Reading": reading})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if symbol == "UPLOAD":
            st.markdown("<p class='small'>For uploaded files, market-wide volatility is taken from the latest date in the "
                        "training dataset.</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------ compare
with tab_cmp:
    st.markdown("### Latest forecast for every coin")
    t = latest_all.reset_index()[["symbol", "date", "pred", "vol_7"]].copy()
    t["regime"] = [regime_of(v, thr, labels) for v in t["pred"]]
    t = t.sort_values("pred")
    fig = go.Figure(go.Bar(x=t["pred"], y=t["symbol"], orientation="h", marker_color=[REGIME_COLORS[r] for r in t["regime"]],
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(height=max(320, 22 * len(t)), margin=dict(l=0, r=0, t=6, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(tickformat=".0%", gridcolor=P["grid"]), yaxis=dict(title=None))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    out = t.sort_values("pred", ascending=False).rename(columns={
        "symbol": "Coin", "date": "As of", "pred": "Forecast", "vol_7": "Last 7 days", "regime": "Level"})
    out["As of"] = out["As of"].dt.date
    out["Change"] = out["Forecast"] / out["Last 7 days"] - 1
    st.dataframe(out, hide_index=True, width="stretch", column_config={
        "Forecast": st.column_config.NumberColumn(format="percent"),
        "Last 7 days": st.column_config.NumberColumn(format="percent"),
        "Change": st.column_config.NumberColumn(format="percent")})
    st.download_button("Download as CSV", out.to_csv(index=False), "volatility_forecasts.csv", "text/csv")

# ------------------------------------------------------------------ model quality
with tab_model:
    st.markdown("### Accuracy on data the model never saw")
    sp = meta["split"]
    st.markdown(f"<p class='lede'>Scores below come from {sp['test'][0]} to {sp['test'][1]} "
                f"({sp['test'][2]:,} coin-days), which was kept out of training and tuning.</p>", unsafe_allow_html=True)
    tm = pd.DataFrame(meta["test_metrics"]).T.sort_values("RMSE")
    tm.index.name = "Model"
    st.dataframe(tm.style.format("{:.4f}").apply(lambda r: ["font-weight:600" if r.name == meta["model_name"] else "" for _ in r], axis=1),
                 width="stretch")
    st.markdown("<p class='small'>RMSE and MAE are in volatility units (0.01 = 1 percentage point of daily swing); lower is better. "
                "R² is the share of variation explained; higher is better. The naive row assumes next week will look like last week.</p>",
                unsafe_allow_html=True)
    reg = meta["regime_test"]
    st.markdown(f"The model puts a coin in the right level (Low, Medium or High) **{reg['accuracy']:.0%}** of the time.")
    c1, c2 = st.columns(2)
    for col, name in [(c1, "forecast_timeseries.png"), (c2, "regime_confusion.png"),
                      (c1, "feature_importance.png"), (c2, "pred_vs_actual.png")]:
        p = config.FIG_DIR / name
        if p.exists():
            col.image(str(p), width="stretch")

# ------------------------------------------------------------------ data
with tab_data:
    st.markdown("### What the model was built from")
    st.markdown(
        f"- **Coins:** {meta['n_coins']}\n- **Rows used:** {len(feats):,}\n"
        f"- **Period:** {feats['date'].min().date()} to {feats['date'].max().date()}\n"
        f"- **Inputs per coin per day:** open, high, low, close, volume, market cap\n"
        f"- **Engineered features:** {meta['n_features']} (rolling volatility, ATR, Bollinger Bands, liquidity ratios, momentum, market-wide context)\n"
        f"- **What is predicted:** the standard deviation of daily returns over the next {meta['horizon_days']} days")
    st.markdown("<p class='small'>Volatility describes how large price moves are, not which direction they go. "
                "This is a statistical estimate, not financial advice.</p>", unsafe_allow_html=True)
