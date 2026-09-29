"""Central configuration. Change values here, not inside the pipeline code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DOCS_DIR = ROOT / "docs"

# ---- Target definition -------------------------------------------------
# Target = realised volatility (std-dev of daily log returns) over the NEXT
# HORIZON days. Features only ever use information up to day t.
HORIZON = 7

# ---- Data cleaning -----------------------------------------------------
MIN_HISTORY_DAYS = 120        # coins with less history are dropped
MAX_FFILL_DAYS = 3            # longest gap we are willing to forward fill

# ---- Train / test split (chronological, no shuffling) ------------------
TEST_FRACTION = 0.20          # last 20% of the calendar is the hold-out set
VAL_FRACTION = 0.10           # the 10% before that is used as validation
GAP_DAYS = HORIZON            # embargo so target windows never overlap the split

# ---- Volatility regimes (quantiles of the training target) -------------
REGIME_QUANTILES = (0.33, 0.66)
REGIME_LABELS = ("Low", "Medium", "High")

RANDOM_STATE = 42
