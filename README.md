# AURA — Adaptive AI Quant

> An experimental AI-powered quantitative research platform for market analysis, ML-driven signals, realistic backtesting, walk-forward validation, and out-of-sample evaluation.

AURA is a quantitative research project built around one core principle:

**Do not confuse a good-looking backtest with a reliable trading model.**

The project combines technical indicators, machine learning, walk-forward validation, transaction costs, confidence-based signals, risk metrics, market-regime analysis, and an interactive Streamlit dashboard into a single research platform.

AURA is currently designed as a **research and paper-trading system**, not a production trading bot.

---

## Overview

AURA analyzes historical market data and uses machine learning to generate directional trading signals.

The current system supports:

- Historical market data through Yahoo Finance
- Local market-data caching
- Technical feature engineering
- Random Forest
- Gradient Boosting
- Logistic Regression
- Time-based validation
- Walk-forward out-of-sample validation
- Embargo periods to reduce overlapping-label leakage
- Confidence-based trading signals
- Transaction-cost modeling
- Slippage
- Fixed-fractional position sizing
- Risk and performance metrics
- Market-regime analysis
- Probability-quality analysis
- Final unseen holdout testing
- Experiment tracking and result storage
- Robustness ranking
- Interactive Streamlit dashboard

The goal is not to manufacture impressive numbers.

The goal is to determine whether a strategy survives increasingly realistic evaluation.

---

## Why AURA?

Many beginner ML trading projects make the same methodological mistakes.

| Common Problem | AURA Approach |
|---|---|
| Random train/test splitting | Time-based and walk-forward validation |
| Future information leakage | Historical expanding training windows |
| Overlapping future-return labels | Embargo period between training and testing |
| Zero transaction costs | Commission + slippage |
| Looking only at accuracy | Returns, Sharpe, drawdown, profit factor and exposure |
| Optimizing one backtest | Multiple configurations and robustness analysis |
| Reporting only the best result | Separate final holdout evaluation |
| Assuming confidence means correctness | Probability-quality analysis |
| Ignoring market conditions | Market-regime analysis |
| Treating historical performance as guaranteed | Explicit research limitations and disclaimer |

AURA is intentionally built to make weak results visible rather than hide them.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │    Yahoo Finance    │
                         │   Historical Data   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    PriceLoader      │
                         │   Local Data Cache  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────┐
                  │       Feature Engineering      │
                  │                                │
                  │ Returns / SMA / RSI / MACD     │
                  │ Bollinger Bands / ATR / Volume │
                  │ SPY / QQQ / VIX Context        │
                  └───────────────┬────────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │     ML Pipeline     │
                         │                     │
                         │ Random Forest       │
                         │ Gradient Boosting   │
                         │ Logistic Regression │
                         └──────────┬──────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  Walk-Forward Validation │
                       │                          │
                       │ Train → Embargo → Test  │
                       │          ↓               │
                       │     OOS Predictions      │
                       └────────────┬─────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌────────────────────┐          ┌────────────────────┐
          │  Trading Signals   │          │ Model Evaluation   │
          │                    │          │                    │
          │ Prediction         │          │ Accuracy           │
          │ Probability        │          │ CAGR               │
          │ Confidence         │          │ Sharpe             │
          │ Threshold          │          │ Drawdown           │
          └─────────┬──────────┘          │ Profit Factor      │
                    │                     └────────────────────┘
                    ▼
          ┌────────────────────┐
          │   Backtest Engine  │
          │                    │
          │ Commission         │
          │ Slippage           │
          │ Position Sizing    │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │   Streamlit AURA   │
          │       Dashboard    │
          │                    │
          │ Signals            │
          │ Charts             │
          │ Risk               │
          │ Validation         │
          │ Performance        │
          └────────────────────┘
ML Methodology
Features

AURA currently uses technical and market-context features.

Technical Features
1-day return
5-day return
20-day return
SMA 10
SMA 50
SMA ratio
RSI 14
20-day volatility
ATR 14
MACD
MACD signal
MACD histogram
Bollinger Band position
Market Context
SPY 1-day return
SPY 5-day return
QQQ 1-day return
QQQ 5-day return
VIX level
VIX 5-day change
Relative strength versus the market
Prediction Horizons

AURA supports multiple prediction horizons:

Horizon	Meaning
1D	Approximately next trading day
1W	Approximately next trading week
1M	Approximately next trading month

Internally these correspond to:

1 trading day
5 trading days
21 trading days
Walk-Forward Validation

AURA does not rely on shuffled train/test splits for its primary ML evaluation.

Instead, the model learns from historical data and is evaluated on later unseen periods.

The validation structure is conceptually:

|------ TRAIN ------|-- EMBARGO --|-- TEST --|

|---------- TRAIN ----------|-- EMBARGO --|-- TEST --|

|--------------- TRAIN ---------------|-- EMBARGO --|-- TEST --|

The training window expands over time.

This better represents how a model would operate when deployed sequentially on market data.

The system also supports an embargo period to reduce leakage caused by overlapping future-return labels.

Final Holdout Evaluation

AURA separates model development from a completely unseen final test period.

Current evaluation structure:

Development:
2020-01-01 → 2025-01-01

Final Holdout:
2025-01-01 → 2026-08-22

The holdout period is kept completely separate from model development and is used only for final evaluation.

This provides a stronger test of whether a selected configuration generalizes beyond the periods used during experimentation.

Experimentation

AURA evaluates combinations of:

Tickers
Prediction horizons
Confidence thresholds
Model configurations
Market regimes

Current tested horizons include:

1 trading day
5 trading days
21 trading days

Current confidence thresholds include:

0.50
0.55
0.60
0.65

Experiment results are stored under:

reports/

The project also includes robustness ranking based on multiple performance characteristics rather than simply selecting the highest raw return.

Research Findings

The experiments produced several important findings.

1. Raw Directional Prediction Is Difficult

For AAPL, the 5-day directional model produced walk-forward accuracy close to the 50% region.

This indicates that naive technical features do not provide a consistently strong edge for predicting short-term market direction.

That is an important result rather than a failure.

2. Confidence Is Potentially More Useful Than Raw Accuracy

Probability-quality analysis showed that higher-confidence predictions can contain more useful information than low-confidence predictions.

One AAPL walk-forward experiment produced:

Probability Bucket	Accuracy
≤ 0.50	48.18%
0.50–0.55	60.00%
0.55–0.60	38.89%
0.60–0.65	53.85%
0.65–0.70	72.73%
> 0.70	56.67%

The intermediate buckets contained relatively small sample sizes, so these numbers should not be treated as statistically conclusive.

However, the results support investigating confidence-based signal selection rather than relying exclusively on classification accuracy.

3. Market Regime Matters

AURA evaluates model behavior across simplified market regimes:

Bull
Bear
Mixed

For one AAPL strong-move experiment:

Regime	Samples	Accuracy
Bear	75	56.00%
Bull	187	47.59%
Mixed	108	51.85%

The results suggest that model behavior is not uniform across market environments.

4. Stronger Classification Targets Are Not Automatically Easier

A ±2% five-day target was tested to ignore small and ambiguous price movements.

For AAPL:

Outcome	Frequency
Strong Bearish (< -2%)	24.35%
Strong Bullish (> +2%)	35.36%
Ignored / Ambiguous	40.29%

The resulting classification problem still produced weak out-of-sample performance.

This reinforces the idea that simply changing the target definition does not automatically create predictive power.

5. Regression Did Not Beat the Naive Baseline

Five-day return regression was also tested.

For AAPL, the naive mean-return baseline achieved:

MAE  = 0.029648
RMSE = 0.037874
R²   = -0.0035

The tested ML regressors performed worse:

Random Forest
MAE  = 0.034171
RMSE = 0.043392
R²   = -0.3173
Gradient Boosting
MAE  = 0.037377
RMSE = 0.047933
R²   = -0.6074
Ridge
MAE  = 0.035844
RMSE = 0.045120
R²   = -0.4242

This is another reason AURA treats model experimentation as research rather than assuming ML must outperform a baseline.

Backtesting

AURA uses a backtesting engine with realistic execution assumptions.

The system supports:

Commission
Slippage
Fixed-fractional position sizing
Long/flat strategies
Equity curves
Trade statistics

Performance metrics include:

Total Return
CAGR
Volatility
Sharpe Ratio
Sortino Ratio
Maximum Drawdown
Calmar Ratio
Win Rate
Profit Factor
Number of Trades
Exposure

The backtesting framework is designed to separate signal generation from execution mechanics.

Transaction Costs

The backtest engine incorporates transaction costs rather than assuming frictionless trading.

The current configuration supports:

Commission: 0.05%
Slippage:   2 bps

These assumptions are configurable.

Example Final Holdout Result

One tested configuration produced the following result:

Ticker:              AAPL
Horizon:             5 trading days
Confidence threshold: 0.50

Holdout period:
2025-01-01 → 2026-08-22

Performance:

Total Return : 12.96%
CAGR         : 7.80%
Volatility   : 24.10%
Sharpe       : 0.43
Sortino      : 0.48
Max Drawdown : -21.88%
Calmar       : 0.36
Win Rate     : 62.96%
Profit Factor: 1.65
Trades       : 27
Exposure     : 60.98%

These results are presented as an experimental research result, not as evidence of guaranteed future performance.

AURA intentionally avoids presenting a single backtest result as proof of predictive ability.

Dashboard

AURA includes an interactive Streamlit dashboard.

The dashboard provides:

Ticker selection
Forecast horizon selection
Confidence threshold selection
Current ML prediction
Model confidence
Risk levels
Technical indicators
ML strategy performance
Walk-forward validation
Final holdout performance
Strategy vs buy-and-hold comparison
Latest market data

Run the dashboard with:

streamlit run app.py
Project Structure
AURA/
│
├── app.py
│
├── configs/
│
├── data/
│   └── cache/
│
├── notebooks/
│
├── reports/
│
├── scripts/
│
├── src/
│   └── quant_ml/
│       ├── backtest/
│       ├── data/
│       ├── features/
│       ├── ml/
│       └── strategies/
│
├── tests/
│
├── .github/
│
├── LICENSE
├── pyproject.toml
├── README.md
└── REPORT.md
Important ML Modules
src/quant_ml/ml/

├── experiments.py
├── inference.py
├── performance.py
├── pipeline.py
├── ranking.py
├── risk.py
├── tracking.py
└── walkforward.py
Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AURA
2. Create a virtual environment
python -m venv .venv
3. Activate it on Windows
.venv\Scripts\activate
4. Install the project
pip install -e .
5. Run the dashboard
streamlit run app.py
Testing

Run the test suite with:

pytest

The repository includes tests covering core components such as:

Backtesting
Transaction costs
Technical features
Performance metrics
Strategies
Walk-forward validation
Research Philosophy

AURA is intentionally designed around evaluation before optimization.

The workflow is:

Build
  ↓
Validate
  ↓
Walk-Forward Test
  ↓
Analyze Errors
  ↓
Test Alternative Targets
  ↓
Test Market Context
  ↓
Backtest
  ↓
Final Holdout
  ↓
Assess Robustness

A model is not considered successful simply because it produces a positive backtest.

The more important questions are:

Does the result survive out-of-sample testing?
Does it beat a meaningful baseline?
Does it remain useful across market regimes?
Does confidence correspond to better decisions?
Does performance survive realistic transaction costs?
Does the result remain reasonable on completely unseen data?
Limitations

AURA is intentionally treated as a research platform.

Current limitations include:

Short-term market prediction remains difficult.
Model accuracy is generally close to the base-rate region.
Some experiment configurations have relatively few trades.
Probability buckets can have small sample sizes.
Historical performance does not imply future performance.
Market regimes can change.
Yahoo Finance data is not a production market-data feed.
The current system is not designed for live capital deployment.
Hyperparameter and configuration selection can introduce multiple-comparison bias.
The current research focuses primarily on single-asset experiments.
The current feature set is primarily technical and market-context based.
No claim is made that the current models provide a persistent trading edge.

These limitations are part of the research conclusions rather than being hidden.

Future Work

Potential future extensions include:

Modeling
Probability calibration
Better uncertainty estimation
More robust market-regime models
Additional model architectures
Feature selection and stability analysis
Market Data
Cross-asset features
Sector and macroeconomic features
Higher-quality market-data infrastructure
Additional alternative data sources
Statistical Rigor
Statistical confidence intervals
Bootstrap analysis
Monte Carlo analysis
Deflated Sharpe ratio
Multiple-comparison correction
More formal hypothesis testing
Portfolio Research
Multi-asset portfolio backtesting
Portfolio-level optimization
Position-level risk allocation
Correlation-aware sizing
Deployment
Paper-trading integration
Live market-data infrastructure
FastAPI inference service
Containerized deployment
Monitoring and experiment tracking
Disclaimer

AURA is an experimental quantitative research project.

It is not financial advice and should not be used as the sole basis for investment decisions.

Historical backtests, machine-learning predictions, simulated returns, and statistical analysis do not guarantee future performance.

Trading and investing involve substantial risk, including the potential loss of capital.

The project is intended for educational, experimental, and research purposes.