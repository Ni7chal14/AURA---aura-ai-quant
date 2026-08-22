"""ML pipeline construction.

Builds (features, labels) from raw price data and instantiates models from
config. Centralizing this prevents the train-prod skew problem: the same
function builds features for training, walk-forward testing, and live
inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from quant_ml.features.technical import build_feature_matrix


def make_classifier(model_type: str, params: dict[str, Any] | None = None) -> ClassifierMixin:
    """Factory for sklearn classifiers."""
    params = params or {}
    if model_type == "random_forest":
        defaults = {"n_estimators": 200, "max_depth": 5, "random_state": 42, "n_jobs": -1}
        return RandomForestClassifier(**{**defaults, **params})
    if model_type == "logistic_regression":
        defaults = {"max_iter": 1000, "random_state": 42}
        return LogisticRegression(**{**defaults, **params})
    if model_type == "gradient_boosting":
        defaults = {"n_estimators": 200, "max_depth": 3, "random_state": 42}
        return GradientBoostingClassifier(**{**defaults, **params})
    raise ValueError(f"Unknown model_type: {model_type}")


def build_dataset(
    prices: pd.DataFrame,
    horizon: int = 1,
    close_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for ML training.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLCV data.
    horizon : int
        Forecast horizon in bars. y_t = 1 if return from t to t+horizon > 0.
    close_col : str
        Price column to use for return calculation.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, NaN rows dropped.
    y : pd.Series
        Binary target aligned with X.
    """
    features = build_feature_matrix(prices, close_col=close_col)
    forward_return = prices[close_col].shift(-horizon) / prices[close_col] - 1
    target = (forward_return > 0).astype(int)
    target.name = "target"

    df = features.join(target).dropna()
    # Drop the last `horizon` rows where target is NaN (no future to look at)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Replace any inf from divisions with NaN, then drop
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    return X, y

def build_multiclass_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    neutral_threshold: float = 0.01,
    close_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a three-class ML dataset.

    Target classes
    --------------
    -1 : bearish
         Future return < -neutral_threshold

     0 : neutral
         -neutral_threshold <= future return <= neutral_threshold

     1 : bullish
         Future return > neutral_threshold

    Parameters
    ----------
    prices : pd.DataFrame
        OHLCV data.

    horizon : int
        Forecast horizon in bars.

    neutral_threshold : float
        Return range considered neutral.

        Example:
            0.01 = +/- 1%

    close_col : str
        Price column used for calculating future returns.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Multiclass target.
    """

    features = build_feature_matrix(
        prices,
        close_col=close_col,
    )

    forward_return = (
        prices[close_col].shift(-horizon)
        / prices[close_col]
        - 1
    )

    target = pd.Series(
        0,
        index=prices.index,
        dtype=int,
        name="target",
    )

    # Bearish
    target.loc[
        forward_return < -neutral_threshold
    ] = -1

    # Bullish
    target.loc[
        forward_return > neutral_threshold
    ] = 1

    # Neutral remains 0

    df = features.join(target)

    # Rows where the future return doesn't exist
    # must not be used for training.
    df = df.loc[
        forward_return.notna()
    ]

    df = df.dropna()

    X = df.drop(
        columns=["target"]
    )

    y = df["target"].astype(int)

    # Replace infinities caused by feature calculations.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = X.notna().all(axis=1)

    X = X.loc[valid]
    y = y.loc[X.index]

    return X, y

def make_regressor(
    model_type: str,
    params: dict[str, Any] | None = None,
):
    """Factory for sklearn regression models."""

    params = params or {}

    if model_type == "random_forest":
        defaults = {
            "n_estimators": 200,
            "max_depth": 5,
            "random_state": 42,
            "n_jobs": -1,
        }

        return RandomForestRegressor(
            **{**defaults, **params}
        )

    if model_type == "gradient_boosting":
        defaults = {
            "n_estimators": 200,
            "max_depth": 3,
            "random_state": 42,
        }

        return GradientBoostingRegressor(
            **{**defaults, **params}
        )

    if model_type == "ridge":
        defaults = {
            "alpha": 1.0,
        }

        return Ridge(
            **{**defaults, **params}
        )

    raise ValueError(
        f"Unknown regressor type: {model_type}"
    )

def build_regression_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    close_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a regression dataset for future return prediction.

    Target:
        future percentage return over `horizon` bars.

    Example:
        horizon=5
        y_t = return from t to t+5
    """

    features = build_feature_matrix(
        prices,
        close_col=close_col,
    )

    forward_return = (
        prices[close_col].shift(-horizon)
        / prices[close_col]
        - 1
    )

    forward_return.name = "target"

    df = features.join(
        forward_return
    )

    df = df.dropna()

    X = df.drop(
        columns=["target"]
    )

    y = df["target"].astype(float)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = X.notna().all(axis=1)

    X = X.loc[valid]
    y = y.loc[X.index]

    return X, y

def build_strong_move_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.02,
    close_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a binary dataset containing only meaningful future moves.

    Target:
        0 = bearish: future return < -threshold
        1 = bullish: future return > +threshold

    Returns between -threshold and +threshold are excluded.
    """

    features = build_feature_matrix(
        prices,
        close_col=close_col,
    )

    forward_return = (
        prices[close_col].shift(-horizon)
        / prices[close_col]
        - 1
    )

    target = pd.Series(
        np.nan,
        index=prices.index,
        dtype=float,
        name="target",
    )

    target.loc[
        forward_return < -threshold
    ] = 0

    target.loc[
        forward_return > threshold
    ] = 1

    df = features.join(target)

    # Remove:
    # 1. rows without a future return
    # 2. ambiguous movements inside ±threshold
    df = df.dropna()

    X = df.drop(columns=["target"])

    y = df["target"].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = X.notna().all(axis=1)

    X = X.loc[valid]
    y = y.loc[X.index]

    return X, y

def build_market_enhanced_dataset(
    prices: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    horizon: int = 5,
    threshold: float = 0.02,
    close_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a strong-move classification dataset using both
    technical and market-context features.

    Classes:
        0 = bearish: future return < -threshold
        1 = bullish: future return > +threshold

    Ambiguous returns inside +/- threshold are excluded.
    """

    from quant_ml.features.market import build_market_context

    technical = build_feature_matrix(
        prices,
        close_col=close_col,
    )

    market = build_market_context(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    # Align market features with the stock's trading dates.
    features = technical.join(
        market,
        how="left",
    )

    forward_return = (
        prices[close_col].shift(-horizon)
        / prices[close_col]
        - 1
    )

    target = pd.Series(
        np.nan,
        index=prices.index,
        dtype=float,
        name="target",
    )

    target.loc[
        forward_return < -threshold
    ] = 0

    target.loc[
        forward_return > threshold
    ] = 1

    dataset = features.join(target)

    dataset = dataset.dropna()

    X = dataset.drop(
        columns=["target"]
    )

    y = dataset["target"].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = X.notna().all(axis=1)

    X = X.loc[valid]
    y = y.loc[X.index]

    return X, y