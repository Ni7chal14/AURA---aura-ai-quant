from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


class PriceLoader:
    """Load historical OHLCV price data with optional local caching."""

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        ticker: str,
        start_date,
        end_date,
    ) -> pd.DataFrame:
        """Load historical price data for a ticker."""

        cache_file = self.cache_dir / (
            f"{ticker}_{start_date}_{end_date}.csv"
        )

        # Try local cache first
        if cache_file.exists():
            prices = pd.read_csv(
                cache_file,
                index_col=0,
                parse_dates=True,
            )

            return prices

        # Download from Yahoo Finance
        prices = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if prices.empty:
            raise ValueError(
                f"No price data found for {ticker} "
                f"between {start_date} and {end_date}."
            )

        # yfinance can return MultiIndex columns
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

        prices.index.name = "Date"

        # Save cache
        prices.to_csv(cache_file)

        return prices