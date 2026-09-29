"""Model definitions, chronological splitting and time-aware tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config

_EPS = 1e-6

# randomised-search budget per model (tree ensembles are slower to fit)
N_ITER = {"Ridge": 15, "Random Forest": 4, "Gradient Boosting": 12}


def log_target(y):
    return np.log(np.asarray(y) + _EPS)


def exp_target(y):
    return np.exp(np.asarray(y)) - _EPS


def _clip(x):
    """Guard against extreme outliers after scaling (keeps linear models stable)."""
    return np.clip(x, -8, 8)


def make_pipeline(estimator) -> TransformedTargetRegressor:
    """scale -> clip -> estimator, predicting log(volatility)."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clip", FunctionTransformer(_clip)),
        ("model", estimator),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=log_target, inverse_func=exp_target)


# name -> (estimator, search space over the inner "model" step)
def candidates(quick: bool = False):
    rs = config.RANDOM_STATE
    return {
        "Ridge": (
            Ridge(),
            {"regressor__model__alpha": loguniform(1e-2, 1e3)},
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=rs),
            {
                "regressor__model__max_depth": randint(7, 12),
                "regressor__model__min_samples_leaf": randint(25, 70),
                "regressor__model__max_features": uniform(0.25, 0.2),
            },
        ),
        "Gradient Boosting": (
            HistGradientBoostingRegressor(random_state=rs, early_stopping=False),
            {
                "regressor__model__learning_rate": loguniform(0.02, 0.2),
                "regressor__model__max_iter": randint(150, 450),
                "regressor__model__max_depth": randint(3, 8),
                "regressor__model__min_samples_leaf": randint(20, 120),
                "regressor__model__l2_regularization": loguniform(1e-3, 10),
            },
        ),
    }


def chronological_split(df: pd.DataFrame):
    """Split by calendar date: train | (gap) | validation | (gap) | test.

    Because a row's target looks GAP_DAYS into the future, we leave a gap of
    that many days before each boundary so no training target overlaps the
    period it is later evaluated on.
    """
    dates = np.sort(df["date"].unique())
    test_start = dates[int(len(dates) * (1 - config.TEST_FRACTION))]
    val_start = dates[int(len(dates) * (1 - config.TEST_FRACTION - config.VAL_FRACTION))]
    gap = np.timedelta64(config.GAP_DAYS, "D")
    train = df[df["date"] < val_start - gap]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start - gap)]
    test = df[df["date"] >= test_start]
    info = {
        "train": [str(pd.Timestamp(train["date"].min()).date()), str(pd.Timestamp(train["date"].max()).date()), int(len(train))],
        "validation": [str(pd.Timestamp(val["date"].min()).date()), str(pd.Timestamp(val["date"].max()).date()), int(len(val))],
        "test": [str(pd.Timestamp(test["date"].min()).date()), str(pd.Timestamp(test["date"].max()).date()), int(len(test))],
    }
    return train, val, test, info


def tune(name: str, n_coins: int, n_iter: int = 12, n_splits: int = 4):
    """Randomised search with forward-chaining CV. Rows must be sorted by date."""
    est, space = candidates()[name]
    n_coins = max(1, int(n_coins))
    cv = TimeSeriesSplit(n_splits=n_splits, gap=config.GAP_DAYS * n_coins)
    search = RandomizedSearchCV(
        make_pipeline(est), space, n_iter=n_iter, cv=cv,
        scoring="neg_root_mean_squared_error", random_state=config.RANDOM_STATE,
        n_jobs=1, refit=False,
    )
    return search
