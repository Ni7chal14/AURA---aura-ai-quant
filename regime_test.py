from quant_ml.data import PriceLoader
from quant_ml.ml.pipeline import (
    build_strong_move_dataset,
    make_classifier,
)
from quant_ml.ml.walkforward import WalkForwardValidator

import pandas as pd


# ---------------------------------------------------------
# LOAD AAPL DATA
# ---------------------------------------------------------

loader = PriceLoader(cache_dir="data/cache")

prices = loader.load(
    "AAPL",
    "2020-01-01",
    "2026-08-22",
)


# ---------------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------------

X, y = build_strong_move_dataset(
    prices,
    horizon=5,
    threshold=0.02,
)


# ---------------------------------------------------------
# WALK-FORWARD ML
# ---------------------------------------------------------

model = make_classifier(
    "gradient_boosting"
)

validator = WalkForwardValidator(
    n_splits=5,
    train_months=24,
    test_months=6,
    embargo_days=5,
)

wf = validator.evaluate(
    model,
    X,
    y,
)


# ---------------------------------------------------------
# BUILD MARKET REGIMES
# ---------------------------------------------------------

spy = loader.load(
    "SPY",
    "2020-01-01",
    "2026-08-22",
)

spy_close = spy["Adj Close"]

sma_200 = spy_close.rolling(200).mean()
spy_return_20d = spy_close.pct_change(20)

regime = pd.Series(
    "mixed",
    index=spy.index,
)

regime[
    (spy_close > sma_200)
    & (spy_return_20d > 0)
] = "bull"

regime[
    (spy_close <= sma_200)
    & (spy_return_20d < 0)
] = "bear"


# ---------------------------------------------------------
# ALIGN EVERYTHING
# ---------------------------------------------------------

regime = regime.reindex(
    wf.predictions.index
)

actual = y.reindex(
    wf.predictions.index
)

pred = wf.predictions

forward_return = (
    prices["Adj Close"].shift(-5)
    / prices["Adj Close"]
    - 1
).reindex(
    wf.predictions.index
)


# ---------------------------------------------------------
# CREATE ANALYSIS DATAFRAME
# ---------------------------------------------------------

results = pd.DataFrame(
    {
        "regime": regime,
        "pred": pred,
        "actual": actual,
        "return": forward_return,
    }
).dropna()


results["correct"] = (
    results["pred"]
    == results["actual"]
)

results["strategy_return"] = results[
    "return"
].where(
    results["pred"] == 1,
    0,
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print(
    "=== REGIME TRADING PERFORMANCE ==="
)

summary = results.groupby(
    "regime"
).agg(
    samples=("return", "size"),
    accuracy=("correct", "mean"),
    avg_market_return=("return", "mean"),
    strategy_return=("strategy_return", "sum"),
    avg_strategy_return=("strategy_return", "mean"),
)

print(
    summary.round(4).to_string()
)