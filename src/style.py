"""Shared plot style so every figure in the reports looks the same."""
import matplotlib as mpl

INK, BLUE, AMBER, TEAL, RED, GREY, LIGHT = "#1B2430", "#2F5D8A", "#C7791F", "#3E7C6B", "#A8433A", "#8A94A0", "#E6EAEE"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.8,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, AMBER, TEAL, RED, GREY]),
        "legend.frameon": False,
    })
