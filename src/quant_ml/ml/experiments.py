from __future__ import annotations

import pandas as pd

from quant_ml.ml.performance import run_ml_strategy_backtest


def run_experiment(
    tickers: list[str],
    horizons: list[int] = [1, 5, 21],
    confidence_thresholds: list[float] = [0.50, 0.55, 0.60, 0.65],
    start_date: str = "2020-01-01",
    end_date: str = "2026-08-22",
) -> pd.DataFrame:
    """
    Run AURA ML backtests across multiple assets,
    forecast horizons and confidence thresholds.

    Returns
    -------
    pd.DataFrame
        One row per experiment configuration.
    """

    results = []

    total_experiments = (
        len(tickers)
        * len(horizons)
        * len(confidence_thresholds)
    )

    experiment_number = 0

    for ticker in tickers:

        for horizon in horizons:

            for threshold in confidence_thresholds:

                experiment_number += 1

                print(
                    f"[{experiment_number}/{total_experiments}] "
                    f"{ticker} | "
                    f"horizon={horizon} | "
                    f"threshold={threshold:.2f}"
                )

                try:

                    result, prices, wf_result = (
                        run_ml_strategy_backtest(
                            ticker=ticker,
                            start_date=start_date,
                            end_date=end_date,
                            horizon=horizon,
                            confidence_threshold=threshold,
                        )
                    )

                    metrics = result.metrics

                    results.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "confidence_threshold": threshold,

                            "total_return": metrics.total_return,
                            "cagr": metrics.cagr,
                            "volatility": metrics.volatility,
                            "sharpe": metrics.sharpe,
                            "sortino": metrics.sortino,
                            "max_drawdown": metrics.max_drawdown,
                            "calmar": metrics.calmar,

                            "win_rate": metrics.win_rate,
                            "profit_factor": metrics.profit_factor,
                            "trades": metrics.n_trades,
                            "exposure": metrics.exposure,

                            "oos_accuracy": (
                                wf_result.mean_accuracy
                            ),

                            "oos_accuracy_std": (
                                wf_result.std_accuracy
                            ),

                            "status": "success",
                        }
                    )

                except Exception as e:

                    print(
                        f"ERROR: {ticker} | "
                        f"horizon={horizon} | "
                        f"threshold={threshold:.2f} | "
                        f"{e}"
                    )

                    results.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "confidence_threshold": threshold,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

    return pd.DataFrame(results)


def save_experiment_results(
    results: pd.DataFrame,
    path: str = "reports/ml_experiments.csv",
) -> None:
    """
    Save experiment results to CSV.
    """

    results.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved experiment results to: {path}"
    )