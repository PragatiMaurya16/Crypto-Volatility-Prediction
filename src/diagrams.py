"""Draws the pipeline and system-architecture diagrams used in the docs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from . import config, style


def _box(ax, x, y, w, h, title, sub="", fc="#FFFFFF", ec=style.BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.3))
    multi = "\n" in sub
    ax.text(x + w / 2, y + h * ((0.74 if multi else 0.62) if sub else 0.5), title, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=style.INK)
    if sub:
        ax.text(x + w / 2, y + h * (0.30 if multi else 0.28), sub, ha="center", va="center", fontsize=7.6, color="#4A5561")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=style.GREY, lw=1.3))


def pipeline():
    fig, ax = plt.subplots(figsize=(11.4, 3.6))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = [
        ("Raw CSVs", "OHLC, volume,\nmarket cap", "#F3F5F7"),
        ("Cleaning", "duplicates, gaps,\nOHLC repair", "#FFFFFF"),
        ("Features", "42 features +\n7-day target", "#FFFFFF"),
        ("Split", "train | val | test\nby date + gap", "#FFFFFF"),
        ("Train + tune", "scale, log-target,\ntime-series CV", "#FFFFFF"),
        ("Evaluate", "RMSE, MAE, R2,\nregimes", "#FFFFFF"),
    ]
    w, gap = 1.5, 0.4
    for i, (t, s, fc) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        _box(ax, x, 1.7, w, 1.3, t, s, fc)
        if i:
            _arrow(ax, x - gap + 0.02, 2.35, x - 0.02, 2.35)
    _box(ax, 6.7, 0.1, 2.0, 1.0, "Saved model", "volatility_model.joblib\nmetadata.json", "#EAF1F7")
    _box(ax, 9.0, 0.1, 1.8, 1.0, "Streamlit app", "forecasts + explorer", "#EAF1F7", style.AMBER)
    _arrow(ax, 8.9 - 8.0 + 6.7 + 2.0 - 1.0, 1.7, 7.7, 1.12)
    _arrow(ax, 8.7, 0.6, 9.0, 0.6)
    ax.set_title("Pipeline: from raw prices to a deployed forecast", loc="left", fontsize=11, fontweight="bold")
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / "diagram_pipeline.png"); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.set_xlim(0, 9.5); ax.set_ylim(0, 5); ax.axis("off")
    layers = [
        (3.9, "Data layer", [("data/raw", "Kaggle CSVs"), ("data_loader.py", "schema mapping"), ("preprocessing.py", "cleaning")]),
        (2.6, "Feature layer", [("features.py", "rolling vol, ATR,\nBollinger, liquidity"), ("data/processed", "features.csv.gz")]),
        (1.3, "Model layer", [("modeling.py", "pipelines, splits, tuning"), ("train.py", "fit + select"), ("evaluate.py", "metrics + plots")]),
        (0.0, "Serving layer", [("models/", "joblib + metadata"), ("app/streamlit_app.py", "forecasts, upload, explorer")]),
    ]
    for y, name, boxes in layers:
        ax.add_patch(FancyBboxPatch((0.1, y + 0.05), 9.3, 1.05, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="#F3F5F7", ec="none"))
        ax.text(0.25, y + 0.95, name, fontsize=9, fontweight="bold", color=style.BLUE, va="top")
        n = len(boxes)
        bw = 2.3
        for i, (t, s) in enumerate(boxes):
            x = 1.7 + i * (bw + 0.3)
            _box(ax, x, y + 0.14, bw, 0.72, t, s)
    for y in (3.9, 2.6, 1.3):
        _arrow(ax, 4.75, y + 0.02, 4.75, y - 0.2)
    ax.set_title("System architecture", loc="left", fontsize=11, fontweight="bold")
    fig.savefig(config.FIG_DIR / "diagram_architecture.png"); plt.close(fig)


if __name__ == "__main__":
    style.apply(); pipeline(); architecture()
