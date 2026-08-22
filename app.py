import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "src"),
)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from quant_ml.data import PriceLoader
from quant_ml.ml.inference import generate_prediction
from quant_ml.ml.risk import calculate_risk_levels
from quant_ml.features.technical import build_feature_matrix
from quant_ml.ml.performance import (
    run_strategy_backtest,
    run_ml_strategy_backtest,
    run_final_holdout_test,
)
# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AURA — Adaptive AI Quant",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# CACHE ML BACKTEST
# -----------------------------

@st.cache_data(ttl=1800)
def cached_ml_backtest(
    ticker,
    horizon,
    confidence_threshold,
):
    return run_ml_strategy_backtest(
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2026-08-22",
        horizon=horizon,
        confidence_threshold=confidence_threshold,
    )


@st.cache_data(ttl=1800)
def cached_final_holdout_test(
    ticker,
    horizon,
    confidence_threshold,
):
    return run_final_holdout_test(
        ticker=ticker,
        development_start="2020-01-01",
        test_start="2025-01-01",
        test_end="2026-08-22",
        horizon=horizon,
        confidence_threshold=confidence_threshold,
    )


# -----------------------------
# LOAD MARKET DATA
# -----------------------------

@st.cache_data(ttl=300)
def load_market_data(ticker):
    loader = PriceLoader(cache_dir="data/cache")

    return loader.load(
        ticker,
        "2020-01-01",
        "2026-08-22",
    )


# -----------------------------
# HEADER
# -----------------------------

st.title("AURA")
st.caption("Adaptive AI Quant Trading Platform")
st.divider()


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    ticker = st.selectbox(
        "Select Asset",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )

    timeframe = st.selectbox(
        "Chart Period",
        ["1D", "1W", "1M"],
    )

    confidence_threshold = st.slider(
        "ML Confidence Threshold",
        min_value=0.50,
        max_value=0.65,
        value=0.50,
        step=0.05,
    )

    # Map UI timeframe to ML prediction horizon
    horizon_map = {
        "1D": 1,
        "1W": 5,
        "1M": 21,
    }

    prediction_horizon = horizon_map[timeframe]

    run_analysis = st.button(
        "🔄 Run Analysis",
        use_container_width=True,
    )


# -----------------------------
# LOAD DATA
# -----------------------------

try:

    data = load_market_data(ticker)

    latest = data.iloc[-1]

    close_price = latest["Close"]

    previous_close = data["Close"].iloc[-2]

    daily_change = (
        (close_price / previous_close) - 1
    ) * 100

except Exception as e:

    st.error(f"Unable to load market data: {e}")
    st.stop()


# -----------------------------
# AURA ML PREDICTION
# -----------------------------

try:

    prediction_result = generate_prediction(
        data,
        horizon=prediction_horizon,
    )

    prediction = prediction_result["prediction"]
    confidence = prediction_result["confidence"]
    probabilities = prediction_result["probabilities"]
    selected_model = prediction_result["model_name"]
    model_scores = prediction_result["model_scores"]

    # -----------------------------
    # AURA SIGNAL LOGIC
    # -----------------------------

    ACTION_THRESHOLD = 0.60

    if confidence >= ACTION_THRESHOLD:

        if prediction == 1:
            signal = "🟢 BULLISH"
            signal_description = (
                f"AURA detects a bullish signal with "
                f"{confidence * 100:.1f}% confidence."
            )

        else:
            signal = "🔴 BEARISH"
            signal_description = (
                f"AURA detects a bearish signal with "
                f"{confidence * 100:.1f}% confidence."
            )

    else:

        signal = "🟡 WAIT"

        if prediction == 1:
            signal_description = (
                f"The model leans bullish "
                f"({confidence * 100:.1f}%), "
                f"but confidence is below AURA's "
                f"{ACTION_THRESHOLD * 100:.0f}% action threshold."
            )

        else:
            signal_description = (
                f"The model leans bearish "
                f"({confidence * 100:.1f}%), "
                f"but confidence is below AURA's "
                f"{ACTION_THRESHOLD * 100:.0f}% action threshold."
            )

except Exception as e:

    st.error(
        f"Unable to generate AURA prediction: {e}"
    )

    st.stop()


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

try:

    features = build_feature_matrix(
        data,
        close_col="Adj Close",
    )

    latest_features = (
        features
        .dropna()
        .iloc[-1]
    )

except Exception as e:

    st.error(
        f"Unable to calculate technical indicators: {e}"
    )
    st.stop()

# -----------------------------
# AURA RISK MANAGEMENT
# -----------------------------

try:

    # Calculate ATR from the same market data
    # used by the dashboard.
    atr_value = (
        features["atr_14"]
        .dropna()
        .iloc[-1]
    )

    risk_levels = calculate_risk_levels(
        price=float(close_price),
        atr_value=float(atr_value),
        prediction=prediction,
    )

except Exception as e:

    st.error(
        f"Unable to calculate AURA risk levels: {e}"
    )

    st.stop()

# -----------------------------
# MARKET OVERVIEW
# -----------------------------

st.subheader(
    f"Market Overview — {ticker}"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current Price",
        f"${close_price:,.2f}",
        f"{daily_change:+.2f}%",
    )


with col2:

    st.metric(
        "AURA Signal",
        signal,
    )


with col3:

    st.metric(
        "Model Probability",
        f"{confidence * 100:.1f}%",
    )


with col4:
    st.metric(
        "Model",
        selected_model,
    )


st.caption(signal_description)

st.divider()


# -----------------------------
# AURA MARKET CHART
# -----------------------------

st.subheader("📈 AURA Market Chart")

# Select chart range
if timeframe == "1D":
    chart_data = data.tail(250).copy()

elif timeframe == "1W":
    chart_data = data.tail(250 * 5).copy()

else:
    chart_data = data.tail(250 * 21).copy()


# Calculate moving averages
chart_data["SMA 10"] = (
    chart_data["Close"]
    .rolling(10)
    .mean()
)

chart_data["SMA 50"] = (
    chart_data["Close"]
    .rolling(50)
    .mean()
)

# Replace the line with a proper candlestick chart
import plotly.graph_objects as go

fig = go.Figure()

# Candlesticks
fig.add_trace(
    go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="Price",
    )
)


# SMA 10
fig.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["SMA 10"],
        mode="lines",
        name="SMA 10",
        line=dict(width=1.5),
    )
)


# SMA 50
fig.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["SMA 50"],
        mode="lines",
        name="SMA 50",
        line=dict(width=1.5),
    )
)


# Latest AURA signal marker
latest_date = chart_data.index[-1]

fig.add_trace(
    go.Scatter(
        x=[latest_date],
        y=[close_price],
        mode="markers",
        name="AURA Signal",
        marker=dict(
            size=12,
            symbol="diamond",
        ),
    )
)

# Chart layout
fig.update_layout(
    title=f"{ticker} — {timeframe} Market View",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=550,
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


st.caption(
    f"Showing {timeframe} chart view. "
    "Candlesticks show OHLC price action. "
    "SMA 10 and SMA 50 show short- and medium-term trend. "
    "AURA's ML prediction remains based on daily market features."
)

st.divider()


# -----------------------------
# AURA PROBABILITIES
# -----------------------------

st.subheader("🤖 Model Probability")

prob_col1, prob_col2 = st.columns(2)


with prob_col1:

    st.metric(
        "Probability of Down Move",
        f"{probabilities.get(0, 0) * 100:.1f}%",
    )


with prob_col2:

    st.metric(
        "Probability of Up Move",
        f"{probabilities.get(1, 0) * 100:.1f}%",
    )

st.caption(
    "Model probability is the classifier's estimated probability for the "
    "predicted direction. It is not a guaranteed probability of future returns "
    "or a measure of historical accuracy."
)

st.divider()

# -----------------------------
# MODEL COMPARISON
# -----------------------------

st.subheader("🧠 Model Comparison")

comparison_data = pd.DataFrame(
    {
        "Model": list(model_scores.keys()),
        "Validation Accuracy": [
            score * 100
            for score in model_scores.values()
        ],
    }
)

comparison_data = comparison_data.sort_values(
    "Validation Accuracy",
    ascending=True,
)

fig = px.bar(
    comparison_data,
    x="Validation Accuracy",
    y="Model",
    orientation="h",
    text="Validation Accuracy",
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)

fig.update_layout(
    xaxis_title="Validation Accuracy (%)",
    yaxis_title="",
    xaxis_range=[
        0,
        max(comparison_data["Validation Accuracy"]) + 10,
    ],
    height=300,
    margin=dict(l=20, r=40, t=20, b=20),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.caption(
    f"AURA selected {selected_model} "
    f"based on the highest validation accuracy."
)

# -----------------------------
# AURA SIGNAL ANALYSIS
# -----------------------------

st.subheader("🧠 AURA Signal Analysis")

bullish_factors = []
bearish_factors = []
neutral_factors = []


# RSI
rsi = latest_features["rsi_14"]

if rsi < 30:
    bullish_factors.append(
        f"RSI is oversold ({rsi:.1f})"
    )
elif rsi > 70:
    bearish_factors.append(
        f"RSI is overbought ({rsi:.1f})"
    )
else:
    neutral_factors.append(
        f"RSI is neutral ({rsi:.1f})"
    )


# MACD
macd = latest_features["macd"]
macd_signal = latest_features["macd_signal"]

if macd > macd_signal:
    bullish_factors.append(
        "MACD is above its signal line"
    )
else:
    bearish_factors.append(
        "MACD is below its signal line"
    )


# SMA trend
sma_ratio = latest_features["sma_ratio"]

if sma_ratio > 0:
    bullish_factors.append(
        f"Price is above its SMA trend ({sma_ratio * 100:+.2f}%)"
    )
else:
    bearish_factors.append(
        f"Price is below its SMA trend ({sma_ratio * 100:+.2f}%)"
    )


# Recent momentum
ret_5d = latest_features["ret_5d"]

if ret_5d > 0:
    bullish_factors.append(
        f"5-day momentum is positive ({ret_5d * 100:+.2f}%)"
    )
else:
    bearish_factors.append(
        f"5-day momentum is negative ({ret_5d * 100:+.2f}%)"
    )


# Volatility
volatility = latest_features["vol_20d"]

if volatility > 0.30:
    neutral_factors.append(
        f"Elevated volatility ({volatility * 100:.1f}%)"
    )
else:
    neutral_factors.append(
        f"Moderate volatility ({volatility * 100:.1f}%)"
    )


# -----------------------------
# DISPLAY ANALYSIS
# -----------------------------

analysis_col1, analysis_col2, analysis_col3 = st.columns(3)


with analysis_col1:

    st.markdown("### 🟢 Bullish Factors")

    if bullish_factors:

        for factor in bullish_factors:
            st.success(factor)

    else:

        st.caption("No strong bullish factors detected.")


with analysis_col2:

    st.markdown("### 🔴 Bearish Factors")

    if bearish_factors:

        for factor in bearish_factors:
            st.error(factor)

    else:

        st.caption("No strong bearish factors detected.")


with analysis_col3:

    st.markdown("### ⚪ Market Conditions")

    if neutral_factors:

        for factor in neutral_factors:
            st.info(factor)

    else:

        st.caption("No neutral conditions detected.")


# -----------------------------
# MODEL INTERPRETATION
# -----------------------------

st.markdown("### 🤖 AURA Interpretation")

if confidence >= 0.70:
    confidence_text = "high"

elif confidence >= 0.60:
    confidence_text = "moderate"

else:
    confidence_text = "low"


if signal == "🟡 WAIT":

    st.write(
        f"AURA currently recommends **WAIT**. "
        f"The model leans "
        f"{'bullish' if prediction == 1 else 'bearish'} "
        f"with an estimated **{confidence * 100:.1f}% model probability**, "
        f"which is below the **60% action threshold**. "
        "AURA therefore does not consider the current setup "
        "strong enough to generate an actionable signal."
    )

elif prediction == 1:

    st.write(
        f"AURA currently identifies a **bullish setup** "
        f"with **{confidence_text} model probability "
        f"({confidence * 100:.1f}%)**. "
        "The signal is based on the combination of momentum, "
        "trend, volatility and oscillator features."
    )

else:

    st.write(
        f"AURA currently identifies a **bearish setup** "
        f"with **{confidence_text} model probability "
        f"({confidence * 100:.1f}%)**. "
        "The signal is based on the combination of momentum, "
        "trend, volatility and oscillator features."
    )

# -----------------------------
# AURA RISK MANAGEMENT
# -----------------------------

st.divider()

st.subheader("🎯 AURA Risk Management")

if signal == "🟡 WAIT":

    st.info(
        "AURA is currently in WAIT mode. "
        "The levels below are projected risk levels based "
        "on the model's directional bias and current volatility. "
        "They are not an actionable trade recommendation."
    )

else:

    st.caption(
        "Risk levels are dynamically calculated using "
        "ATR-based market volatility."
    )


risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)


with risk_col1:

    st.metric(
        "🎯 Entry",
        f"${risk_levels['entry']:,.2f}",
    )


with risk_col2:

    st.metric(
        "🛑 Stop Loss",
        f"${risk_levels['stop_loss']:,.2f}",
    )


with risk_col3:

    st.metric(
        "💰 Take Profit",
        f"${risk_levels['take_profit']:,.2f}",
    )


with risk_col4:

    st.metric(
        "⚖️ Risk / Reward",
        f"1 : {risk_levels['risk_reward']:.2f}",
    )


st.caption(
    f"ATR (14): ${risk_levels['atr']:,.2f} "
    f"• Stop distance: 1.5 × ATR "
    f"• Target distance: 3 × ATR"
)

# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------

st.subheader("📊 Technical Indicators")


def get_indicator(
    name,
    decimals=2,
    multiplier=1,
    suffix="",
):
    if name not in latest_features.index:
        return "—"

    value = latest_features[name]

    if pd.isna(value):
        return "—"

    value = value * multiplier

    return f"{value:.{decimals}f}{suffix}"


# -----------------------------
# CORE INDICATORS
# -----------------------------

indicator_cols = st.columns(4)

with indicator_cols[0]:
    st.metric(
        "RSI (14)",
        get_indicator("rsi_14"),
    )

with indicator_cols[1]:
    st.metric(
        "Volatility (20D)",
        get_indicator(
            "vol_20d",
            decimals=2,
            multiplier=100,
            suffix="%",
        ),
    )

with indicator_cols[2]:
    st.metric(
        "SMA Ratio",
        get_indicator(
            "sma_ratio",
            decimals=2,
            multiplier=100,
            suffix="%",
        ),
    )

with indicator_cols[3]:
    st.metric(
        "Bollinger Position",
        get_indicator(
            "bb_pct",
            decimals=2,
            multiplier=100,
            suffix="%",
        ),
    )


# -----------------------------
# MOVEMENT INDICATORS
# -----------------------------

indicator_cols_2 = st.columns(4)

with indicator_cols_2[0]:
    st.metric(
        "1D Return",
        get_indicator(
            "ret_1d",
            decimals=2,
            multiplier=100,
            suffix="%",
        ),
    )

with indicator_cols_2[1]:
    st.metric(
        "5D Return",
        get_indicator(
            "ret_5d",
            decimals=2,
            multiplier=100,
            suffix="%",
        ),
    )

with indicator_cols_2[2]:
    st.metric(
        "20D Return",
        get_indicator(
            "ret_20d",
            decimals=2,
            multiplier=100,
            suffix="%",
        ),
    )

with indicator_cols_2[3]:
    st.metric(
        "SMA 10",
        get_indicator("sma_10"),
    )


# -----------------------------
# ADDITIONAL INDICATORS
# -----------------------------

indicator_cols_3 = st.columns(4)

with indicator_cols_3[0]:
    st.metric(
        "SMA 50",
        get_indicator("sma_50"),
    )

with indicator_cols_3[1]:
    st.metric(
        "MACD",
        get_indicator("macd"),
    )

with indicator_cols_3[2]:
    st.metric(
        "MACD Signal",
        get_indicator("macd_signal"),
    )

with indicator_cols_3[3]:
    st.metric(
        "MACD Histogram",
        get_indicator("macd_hist"),
    )


# -----------------------------
# ATR
# -----------------------------

indicator_cols_4 = st.columns(4)

with indicator_cols_4[0]:
    st.metric(
        "ATR (14)",
        get_indicator("atr_14"),
    )

# -----------------------------
# ML STRATEGY PERFORMANCE
# -----------------------------

st.divider()

st.subheader("🤖 AURA ML Strategy Performance")

horizon_map = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
}

horizon = horizon_map[timeframe]

try:

    # =========================================================
    # WALK-FORWARD BACKTEST
    # =========================================================

    st.markdown("### 🔄 Walk-Forward Backtest")

    ml_result, ml_prices, wf_result = cached_ml_backtest(
        ticker=ticker,
        horizon=horizon,
        confidence_threshold=confidence_threshold,
    )

    metrics = ml_result.metrics

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        st.metric(
            "Forecast Horizon",
            timeframe,
        )

    with config_col2:
        st.metric(
            "Probability Threshold",
            f"{confidence_threshold * 100:.0f}%",
        )

    with config_col3:
        st.metric(
            "OOS Accuracy",
            f"{wf_result.mean_accuracy * 100:.2f}%",
        )

    st.caption(
        "Walk-forward results use expanding historical training windows "
        "and strictly out-of-sample test periods."
    )

    # -----------------------------
    # WALK-FORWARD METRICS
    # -----------------------------

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        st.metric(
            "Total Return",
            f"{metrics.total_return * 100:.2f}%",
        )

    with perf_col2:
        st.metric(
            "CAGR",
            f"{metrics.cagr * 100:.2f}%",
        )

    with perf_col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.sharpe:.2f}",
        )

    with perf_col4:
        st.metric(
            "Max Drawdown",
            f"{metrics.max_drawdown * 100:.2f}%",
        )

    perf_col5, perf_col6, perf_col7, perf_col8 = st.columns(4)

    with perf_col5:
        st.metric(
            "Win Rate",
            f"{metrics.win_rate * 100:.2f}%",
        )

    with perf_col6:
        st.metric(
            "Profit Factor",
            f"{metrics.profit_factor:.2f}",
        )

    with perf_col7:
        st.metric(
            "Trades",
            f"{metrics.n_trades}",
        )

    with perf_col8:
        st.metric(
            "Exposure",
            f"{metrics.exposure * 100:.2f}%",
        )

    # =========================================================
    # FINAL HOLDOUT TEST
    # =========================================================

    st.markdown("### 🧪 Final Holdout Test")

    st.caption(
        "The model is trained using data through 2024-12-31 and evaluated "
        "only on the completely unseen 2025-01-01 to 2026-08-22 period."
    )

    holdout_result, holdout_prices = cached_final_holdout_test(
        ticker=ticker,
        horizon=horizon,
        confidence_threshold=confidence_threshold,
    )

    holdout_metrics = holdout_result.metrics

    hold_col1, hold_col2, hold_col3, hold_col4 = st.columns(4)

    with hold_col1:
        st.metric(
            "Holdout Return",
            f"{holdout_metrics.total_return * 100:.2f}%",
        )

    with hold_col2:
        st.metric(
            "Holdout CAGR",
            f"{holdout_metrics.cagr * 100:.2f}%",
        )

    with hold_col3:
        st.metric(
            "Holdout Sharpe",
            f"{holdout_metrics.sharpe:.2f}",
        )

    with hold_col4:
        st.metric(
            "Holdout Drawdown",
            f"{holdout_metrics.max_drawdown * 100:.2f}%",
        )

    hold_col5, hold_col6, hold_col7, hold_col8 = st.columns(4)

    with hold_col5:
        st.metric(
            "Holdout Win Rate",
            f"{holdout_metrics.win_rate * 100:.2f}%",
        )

    with hold_col6:
        st.metric(
            "Holdout Profit Factor",
            f"{holdout_metrics.profit_factor:.2f}",
        )

    with hold_col7:
        st.metric(
            "Holdout Trades",
            f"{holdout_metrics.n_trades}",
        )

    with hold_col8:
        st.metric(
            "Holdout Exposure",
            f"{holdout_metrics.exposure * 100:.2f}%",
        )

    # =========================================================
    # STRATEGY VS BUY & HOLD
    # =========================================================

    st.subheader("📊 AURA ML vs Buy & Hold")

    equity_curve = ml_result.equity_curve

    initial_capital = 100000

    buy_hold = (
        ml_prices["Adj Close"]
        / ml_prices["Adj Close"].iloc[0]
        * initial_capital
    )

    comparison = pd.DataFrame(
        {
            "AURA ML Strategy": equity_curve,
            "Buy & Hold": buy_hold,
        }
    ).dropna()

    st.line_chart(comparison)

    # -----------------------------
    # FINAL RETURNS
    # -----------------------------

    ml_return = (
        comparison["AURA ML Strategy"].iloc[-1]
        / comparison["AURA ML Strategy"].iloc[0]
        - 1
    )

    benchmark_return = (
        comparison["Buy & Hold"].iloc[-1]
        / comparison["Buy & Hold"].iloc[0]
        - 1
    )

    compare_col1, compare_col2 = st.columns(2)

    with compare_col1:
        st.metric(
            "AURA ML Strategy",
            f"{ml_return * 100:.2f}%",
        )

    with compare_col2:
        st.metric(
            "Buy & Hold",
            f"{benchmark_return * 100:.2f}%",
        )

except Exception as e:

    st.error(
        f"Unable to calculate ML strategy performance: {e}"
    )

# -----------------------------
# TREND / MACD
# -----------------------------

indicator_cols_3 = st.columns(4)


with indicator_cols_3[0]:

    st.metric(
        "SMA 50",
        get_indicator("sma_50"),
    )


with indicator_cols_3[1]:

    st.metric(
        "MACD",
        get_indicator("macd"),
    )


with indicator_cols_3[2]:

    st.metric(
        "MACD Signal",
        get_indicator("macd_signal"),
    )


with indicator_cols_3[3]:

    st.metric(
        "MACD Histogram",
        get_indicator("macd_hist"),
    )

# -----------------------------
# RISK INDICATOR
# -----------------------------

indicator_cols_4 = st.columns(4)


with indicator_cols_4[0]:

    st.metric(
        "ATR (14)",
        get_indicator("atr_14"),
    )


st.divider()

# -----------------------------
# AURA MODEL VALIDATION
# -----------------------------

st.divider()

st.subheader("🧠 AURA Model Validation")

st.caption(
    "AURA evaluates model quality using walk-forward out-of-sample "
    "validation and a completely unseen final holdout period."
)

# -----------------------------
# VALIDATION METRICS
# -----------------------------

val_col1, val_col2, val_col3, val_col4 = st.columns(4)

with val_col1:
    st.metric(
        "Walk-Forward OOS Accuracy",
        f"{wf_result.mean_accuracy * 100:.2f}%"
    )

with val_col2:
    st.metric(
        "Holdout Return",
        f"{holdout_metrics.total_return * 100:.2f}%"
    )

with val_col3:
    st.metric(
        "Holdout Sharpe",
        f"{holdout_metrics.sharpe:.2f}"
    )

with val_col4:
    st.metric(
        "Holdout Max Drawdown",
        f"{holdout_metrics.max_drawdown * 100:.2f}%"
    )

# -----------------------------
# VALIDATION INTERPRETATION
# -----------------------------

oos_accuracy = wf_result.mean_accuracy
holdout_return = holdout_metrics.total_return
holdout_sharpe = holdout_metrics.sharpe

if oos_accuracy >= 0.55 and holdout_return > 0 and holdout_sharpe > 0:
    validation_status = "🟢 Strong Validation"
    validation_message = (
        "AURA shows encouraging out-of-sample consistency and "
        "positive performance on the unseen holdout period."
    )

elif oos_accuracy >= 0.50 and holdout_return > 0:
    validation_status = "🟡 Moderate Validation"
    validation_message = (
        "AURA shows some evidence of predictive usefulness, but "
        "the out-of-sample accuracy remains close to the 50% level."
    )

else:
    validation_status = "🔴 Weak Validation"
    validation_message = (
        "AURA does not yet demonstrate strong predictive consistency. "
        "The current results should be treated as experimental."
    )

st.markdown(f"### {validation_status}")

st.info(validation_message)

st.caption(
    f"Walk-forward OOS accuracy: {oos_accuracy * 100:.2f}% | "
    f"Final holdout return: {holdout_return * 100:.2f}% | "
    f"Final holdout Sharpe: {holdout_sharpe:.2f}"
)

st.caption(
    "Walk-forward accuracy measures predictive consistency on unseen "
    "time periods. The final holdout evaluates the frozen configuration "
    "on data that was not used during model development."
)

# -----------------------------
# LATEST MARKET DATA
# -----------------------------

st.subheader("📋 Latest Market Data")

display_data = data.tail(10).copy()

st.dataframe(
    display_data,
    use_container_width=True,
)


st.divider()


# -----------------------------
# FOOTER
# -----------------------------

st.caption(
    "AURA is an experimental quantitative research platform. "
    "Signals are for research purposes only."
)