from __future__ import annotations

import pandas as pd

from quant_ml.backtest import BacktestEngine, CostModel
from quant_ml.data import PriceLoader
from quant_ml.ml.walkforward import WalkForwardValidator
from quant_ml.ml.pipeline import (
    build_dataset,
    make_classifier,
)
from quant_ml.strategies import MACrossover
from quant_ml.strategies.base import Strategy
from quant_ml.features.technical import build_feature_matrix


# ---------------------------------------------------------
# BASELINE SMA STRATEGY
# ---------------------------------------------------------

def run_strategy_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    fast: int = 10,
    slow: int = 50,
    commission_pct: float = 0.0005,
    slippage_bps: float = 2.0,
):
    """
    Run a MA crossover backtest for a selected ticker.

    Returns the backtest result and price data.
    """

    loader = PriceLoader(cache_dir="data/cache")

    prices = loader.load(
        ticker,
        start_date,
        end_date,
    )

    strategy = MACrossover(
        fast=fast,
        slow=slow,
    )

    cost_model = CostModel(
        commission_pct=commission_pct,
        slippage_bps=slippage_bps,
    )

    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_model=cost_model,
        position_sizing="fixed_fractional",
        fraction=0.95,
    )

    result = engine.run(
        strategy,
        prices,
    )

    return result, prices


# ---------------------------------------------------------
# ML WALK-FORWARD STRATEGY
# ---------------------------------------------------------

class MLWalkForwardStrategy(Strategy):
    """
    Strategy that converts walk-forward ML predictions
    into long/flat trading positions.

    +1 = long
     0 = flat

    The predictions are generated strictly out-of-sample
    using expanding-window walk-forward validation.
    """

    name = "ml_walkforward"

    def __init__(
        self,
        predictions: pd.Series,
    ) -> None:
        self.predictions = predictions

    def generate_signals(
        self,
        prices: pd.DataFrame,
    ) -> pd.Series:

        signals = self.predictions.reindex(
            prices.index
        ).fillna(0)

        # Ensure long-only positions.
        signals = signals.astype(int).clip(0, 1)

        return signals


def run_ml_strategy_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    horizon: int = 1,
    model_type: str = "random_forest",
    model_params: dict | None = None,
    n_splits: int = 5,
    train_months: int = 36,
    test_months: int = 6,
    embargo_days: int = 5,
    confidence_threshold: float = 0.50,
    commission_pct: float = 0.0005,
    slippage_bps: float = 2.0,
):
    """
    Run a walk-forward ML strategy backtest.

    The model is trained only on historical data available
    before each test period.

    Parameters
    ----------
    horizon:
        1  = next trading day
        5  = approximately next trading week
        21 = approximately next trading month.

    Returns
    -------
    result:
        BacktestResult from the existing BacktestEngine.

    prices:
        Historical market data.

    wf_result:
        WalkForwardResult containing OOS predictions,
        probabilities and fold metrics.
    """

    if horizon not in (1, 5, 21):
        raise ValueError(
            "horizon must be 1, 5, or 21 trading days"
        )

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    loader = PriceLoader(
        cache_dir="data/cache"
    )

    prices = loader.load(
        ticker,
        start_date,
        end_date,
    )

    # ---------------------------------------------------------
    # 2. BUILD ML DATASET
    # ---------------------------------------------------------

    X, y = build_dataset(
        prices,
        horizon=horizon,
        close_col="Adj Close",
    )
    
    # ---------------------------------------------------------
    # 3. CREATE MODEL
    # ---------------------------------------------------------

    model = make_classifier(
        model_type=model_type,
        params=model_params,
    )

    # ---------------------------------------------------------
    # 4. WALK-FORWARD VALIDATION
    # ---------------------------------------------------------

    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_months=train_months,
        test_months=test_months,
        embargo_days=embargo_days,
    )

    wf_result = validator.evaluate(
        model,
        X,
        y,
    )

    # ---------------------------------------------------------
    # 5. CONVERT OOS PREDICTIONS TO TRADING SIGNALS
    # ---------------------------------------------------------

    predictions = wf_result.predictions.copy()
    probabilities = wf_result.probabilities.copy()

    # ---------------------------------------------------------
    # CONFIDENCE-BASED SIGNAL
    # ---------------------------------------------------------

    # Only enter a long position when:
    # 1. Model predicts an upward move
    # 2. Probability of upward move >= threshold

    signals = pd.Series(
        0,
        index=prices.index,
        dtype=int,
    )

    valid_indices = probabilities.index

    signals.loc[valid_indices] = (
        (predictions.loc[valid_indices] == 1)
        & (
            probabilities.loc[valid_indices]
            >= confidence_threshold
        )
    ).astype(int)

    # ---------------------------------------------------------
    # 6. STRATEGY
    # ---------------------------------------------------------

    strategy = MLWalkForwardStrategy(
        predictions=signals,
    )

    # ---------------------------------------------------------
    # 7. TRANSACTION COSTS
    # ---------------------------------------------------------

    cost_model = CostModel(
        commission_pct=commission_pct,
        slippage_bps=slippage_bps,
    )

    # ---------------------------------------------------------
    # 8. BACKTEST ENGINE
    # ---------------------------------------------------------

    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_model=cost_model,
        position_sizing="fixed_fractional",
        fraction=0.95,
    )

    # ---------------------------------------------------------
    # 9. RUN BACKTEST
    # ---------------------------------------------------------

    result = engine.run(
        strategy,
        prices,
    )

    return result, prices, wf_result

# ---------------------------------------------------------
# FINAL OUT-OF-SAMPLE TEST
# ---------------------------------------------------------

def run_final_holdout_test(
    ticker: str,
    development_start: str,
    test_start: str,
    test_end: str,
    horizon: int = 1,
    confidence_threshold: float = 0.50,
    initial_capital: float = 100000,
    model_type: str = "random_forest",
    model_params: dict | None = None,
    commission_pct: float = 0.0005,
    slippage_bps: float = 2.0,
):
    """
    Evaluate a frozen ML configuration on an unseen period.

    Development data is used to train the model.

    The test period is kept completely separate from model
    training and is used only for final evaluation.

    Parameters
    ----------
    development_start:
        Beginning of historical training data.

    test_start:
        Beginning of untouched final test period.

    test_end:
        End of final test period.

    horizon:
        Prediction horizon in trading days.

    confidence_threshold:
        Minimum P(up) required to enter a long position.
    """

    if horizon not in (1, 5, 21):
        raise ValueError(
            "horizon must be 1, 5, or 21 trading days"
        )

    # ---------------------------------------------------------
    # 1. LOAD DEVELOPMENT DATA
    # ---------------------------------------------------------

    loader = PriceLoader(
        cache_dir="data/cache"
    )

    development_prices = loader.load(
        ticker,
        development_start,
        test_start,
    )

    # ---------------------------------------------------------
    # 2. BUILD DEVELOPMENT DATASET
    # ---------------------------------------------------------

    development_features = build_feature_matrix(
        development_prices,
        close_col="Adj Close",
    )

    development_forward_return = (
        development_prices["Adj Close"].shift(-horizon)
        / development_prices["Adj Close"]
        - 1
    )

    development_target = (
        development_forward_return > 0
    ).astype(int)

    development_dataset = development_features.copy()

    development_dataset["target"] = (
        development_target
    )

    development_dataset = (
        development_dataset.dropna()
    )

    X_train = development_dataset.drop(
        columns=["target"]
    )

    y_train = development_dataset["target"]

    valid_train = (
        development_forward_return
        .loc[X_train.index]
        .notna()
    )

    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]

    # ---------------------------------------------------------
    # 3. TRAIN FINAL MODEL
    # ---------------------------------------------------------

    model = make_classifier(
        model_type=model_type,
        params=model_params,
    )

    model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
    )

    # ---------------------------------------------------------
    # 4. LOAD COMPLETELY UNSEEN TEST DATA
    # ---------------------------------------------------------

    test_prices = loader.load(
        ticker,
        test_start,
        test_end,
    )

    # ---------------------------------------------------------
    # 5. BUILD TEST FEATURES
    # ---------------------------------------------------------

    test_features = build_feature_matrix(
        test_prices,
        close_col="Adj Close",
    )

    # Only rows with valid technical indicators
    valid_features = test_features.dropna()

    # ---------------------------------------------------------
    # 6. GENERATE TEST PREDICTIONS
    # ---------------------------------------------------------

    predictions = model.predict(
        valid_features.to_numpy()
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            valid_features.to_numpy()
        )[:, 1]
    else:
        probabilities = predictions.astype(float)

    prediction_series = pd.Series(
        predictions,
        index=valid_features.index,
        dtype=int,
    )

    probability_series = pd.Series(
        probabilities,
        index=valid_features.index,
        dtype=float,
    )

    # ---------------------------------------------------------
    # 7. CREATE FINAL TEST SIGNALS
    # ---------------------------------------------------------

    signals = pd.Series(
        0,
        index=test_prices.index,
        dtype=int,
    )

    signals.loc[valid_features.index] = (
        (prediction_series == 1)
        &
        (
            probability_series
            >= confidence_threshold
        )
    ).astype(int)

    # ---------------------------------------------------------
    # 8. CREATE STRATEGY
    # ---------------------------------------------------------

    strategy = MLWalkForwardStrategy(
        predictions=signals
    )

    # ---------------------------------------------------------
    # 9. TRANSACTION COSTS
    # ---------------------------------------------------------

    cost_model = CostModel(
        commission_pct=commission_pct,
        slippage_bps=slippage_bps,
    )

    # ---------------------------------------------------------
    # 10. BACKTEST ENGINE
    # ---------------------------------------------------------

    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_model=cost_model,
        position_sizing="fixed_fractional",
        fraction=0.95,
    )

    # ---------------------------------------------------------
    # 11. RUN FINAL TEST
    # ---------------------------------------------------------

    result = engine.run(
        strategy,
        test_prices,
    )

    return result, test_prices