from __future__ import annotations


def calculate_risk_levels(
    price: float,
    atr_value: float,
    prediction: int,
    stop_atr: float = 1.5,
    target_atr: float = 3.0,
):
    """
    Calculate ATR-based entry, stop-loss and take-profit levels.

    prediction:
        1 = bullish
        0 = bearish
    """

    risk_distance = atr_value * stop_atr
    reward_distance = atr_value * target_atr

    entry = price

    if prediction == 1:
        stop_loss = entry - risk_distance
        take_profit = entry + reward_distance

    else:
        stop_loss = entry + risk_distance
        take_profit = entry - reward_distance

    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)

    risk_reward = reward / risk if risk > 0 else 0

    return {
        "entry": float(entry),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "risk_reward": float(risk_reward),
        "atr": float(atr_value),
    }