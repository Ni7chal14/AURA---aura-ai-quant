from __future__ import annotations

import pandas as pd


def rank_experiments(
    path: str = "reports/ml_experiments.csv",
    min_trades: int = 10,
) -> pd.DataFrame:
    """
    Rank ML experiments using a robustness-oriented score.

    The score rewards:
        - Sharpe ratio
        - CAGR
        - OOS accuracy above 50%
        - Profit factor

    The score penalizes:
        - Maximum drawdown
        - Very low trade counts
    """

    results = pd.read_csv(path)

    results = results[
        results["status"] == "success"
    ].copy()

    # Remove configurations with too few trades.
    results = results[
        results["trades"] >= min_trades
    ].copy()

    if results.empty:
        raise ValueError(
            "No experiments remain after trade-count filtering."
        )

    # ---------------------------------------------------------
    # NORMALIZED COMPONENTS
    # ---------------------------------------------------------

    def normalize(series):
        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(
                1.0,
                index=series.index,
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    results["sharpe_score"] = normalize(
        results["sharpe"]
    )

    results["cagr_score"] = normalize(
        results["cagr"]
    )

    results["profit_factor_score"] = normalize(
        results["profit_factor"]
    )

    # Accuracy around/above 50%.
    results["accuracy_score"] = normalize(
        results["oos_accuracy"]
    )

    # Less negative drawdown = better.
    results["drawdown_score"] = normalize(
        results["max_drawdown"]
    )

    # ---------------------------------------------------------
    # ROBUSTNESS SCORE
    # ---------------------------------------------------------

    results["robustness_score"] = (
        0.30 * results["sharpe_score"]
        + 0.20 * results["cagr_score"]
        + 0.20 * results["profit_factor_score"]
        + 0.15 * results["accuracy_score"]
        + 0.15 * results["drawdown_score"]
    )

    # ---------------------------------------------------------
    # RANK
    # ---------------------------------------------------------

    results = results.sort_values(
        "robustness_score",
        ascending=False,
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        range(1, len(results) + 1),
    )

    return results


def save_ranked_results(
    results: pd.DataFrame,
    path: str = "reports/ml_experiment_rankings.csv",
) -> None:
    """Save ranked experiment results."""

    results.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved rankings to: {path}"
    )