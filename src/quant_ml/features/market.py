from __future__ import annotations

import pandas as pd

from quant_ml.data import PriceLoader


def build_market_context(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build market-context features for a stock.

    Features:
        - SPY 1D return
        - SPY 5D return
        - QQQ 1D return
        - QQQ 5D return
        - VIX level
        - VIX 5D change
        - Stock relative strength vs SPY over 5D
    """

    loader = PriceLoader(cache_dir="data/cache")

    stock = loader.load(
        ticker,
        start_date,
        end_date,
    )

    spy = loader.load(
        "SPY",
        start_date,
        end_date,
    )

    qqq = loader.load(
        "QQQ",
        start_date,
        end_date,
    )

    vix = loader.load(
        "^VIX",
        start_date,
        end_date,
    )

    stock_close = stock["Adj Close"]
    spy_close = spy["Adj Close"]
    qqq_close = qqq["Adj Close"]
    vix_close = vix["Adj Close"]

    context = pd.DataFrame(index=stock.index)

    # ---------------------------------------------------------
    # SPY MARKET RETURNS
    # ---------------------------------------------------------

    context["spy_ret_1d"] = (
        spy_close.pct_change(1)
    )

    context["spy_ret_5d"] = (
        spy_close.pct_change(5)
    )

    # ---------------------------------------------------------
    # QQQ MARKET RETURNS
    # ---------------------------------------------------------

    context["qqq_ret_1d"] = (
        qqq_close.pct_change(1)
    )

    context["qqq_ret_5d"] = (
        qqq_close.pct_change(5)
    )

    # ---------------------------------------------------------
    # VIX
    # ---------------------------------------------------------

    context["vix_level"] = vix_close

    context["vix_change_5d"] = (
        vix_close.pct_change(5)
    )

    # ---------------------------------------------------------
    # RELATIVE STRENGTH
    # ---------------------------------------------------------

    stock_ret_5d = stock_close.pct_change(5)
    spy_ret_5d = spy_close.pct_change(5)

    context["relative_strength_5d"] = (
        stock_ret_5d - spy_ret_5d
    )

    return context