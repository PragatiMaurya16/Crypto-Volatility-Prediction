"""Metrics and evaluation plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config, style

style.apply()


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def to_regime(values, thresholds):
    return np.digitize(values, thresholds)  # 0 = Low, 1 = Medium, 2 = High


def regime_report(y_true, y_pred, thresholds) -> dict:
    a, p = to_regime(y_true, thresholds), to_regime(y_pred, thresholds)
    cm = pd.crosstab(pd.Series(a, name="actual"), pd.Series(p, name="predicted")).reindex(
        index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    return {"accuracy": float((a == p).mean()), "confusion": cm.values.tolist()}


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name)
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        t = table.sort_values(metric, ascending=(metric != "R2"))
        colors = [style.BLUE if i == 0 else style.GREY for i in range(len(t))]
        ax.barh(t.index, t[metric], color=colors)
        ax.invert_yaxis()
        ax.set_title(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"))
        ax.grid(axis="y", visible=False)
        for y, v in enumerate(t[metric]):
            ax.text(v, y, f" {v:.4f}", va="center", fontsize=8.5)
        ax.margins(x=0.25)
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def plot_pred_vs_actual(y, p):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y, p, s=4, alpha=0.25, color=style.BLUE, edgecolors="none")
    lim = [0, float(np.quantile(np.r_[y, p], 0.995))]
    ax.plot(lim, lim, color=style.AMBER, lw=1.2, label="Perfect forecast")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual next-7-day volatility", ylabel="Predicted",
           title="Predicted vs actual (test set)")
    ax.legend(loc="upper left")
    _save(fig, "pred_vs_actual.png")


def plot_residuals(y, p):
    res = p - y
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.hist(res, bins=70, color=style.BLUE, alpha=0.85)
    ax.axvline(0, color=style.AMBER, lw=1.2)
    ax.set(xlabel="Prediction error (predicted - actual)", ylabel="Count", title="Residual distribution")
    _save(fig, "residuals.png")


def plot_timeseries(test: pd.DataFrame, pred: np.ndarray):
    t = test.assign(pred=pred)
    sym = t.groupby("symbol")["date"].count().idxmax()
    s = t[t["symbol"] == sym].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.plot(s["date"], s["target_vol"], color=style.INK, lw=1.2, label="Actual")
    ax.plot(s["date"], s["pred"], color=style.AMBER, lw=1.2, label="Model")
    ax.plot(s["date"], s["vol_7"], color=style.GREY, lw=0.9, ls="--", label="Naive (last 7 days)")
    ax.set(ylabel="Volatility (std of daily log return)", title=f"Forecast vs actual on hold-out data: {sym}")
    ax.legend(ncol=3, loc="upper left")
    _save(fig, "forecast_timeseries.png")
    return sym


def plot_confusion(cm, labels=config.REGIME_LABELS):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else style.INK, fontsize=9)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set(xlabel="Predicted regime", ylabel="Actual regime", title="Volatility regime accuracy")
    _save(fig, "regime_confusion.png")


def feature_importance(model, X: pd.DataFrame, y: pd.Series, n: int = 4000) -> pd.DataFrame:
    idx = np.random.default_rng(config.RANDOM_STATE).choice(len(X), min(n, len(X)), replace=False)
    r = permutation_importance(model, X.iloc[idx], y.iloc[idx], n_repeats=3,
                               scoring="neg_root_mean_squared_error",
                               random_state=config.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                        "std": r.importances_std}).sort_values("importance", ascending=False)
    return imp


def plot_importance(imp: pd.DataFrame, top: int = 15):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(t["feature"], t["importance"], xerr=t["std"], color=style.BLUE, ecolor=style.GREY)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Increase in RMSE when the feature is shuffled", title="Most important features")
    _save(fig, "feature_importance.png")
