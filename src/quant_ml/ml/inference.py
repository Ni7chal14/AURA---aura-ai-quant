from __future__ import annotations

import pandas as pd

from quant_ml.ml.pipeline import (
    build_dataset,
    make_classifier,
)


def generate_prediction(
    prices: pd.DataFrame,
    model_type: str = "random_forest",
    model_params: dict | None = None,
    horizon: int = 1,
):
    """
    Train candidate ML models on historical data and generate
    the latest directional prediction.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical OHLCV price data.

    model_type : str
        Default classifier configuration.

    model_params : dict | None
        Optional model parameters.

    horizon : int
        Forecast horizon in trading days:
            1  = next trading day
            5  = approximately next trading week
            21 = approximately next trading month

    Returns
    -------
    dict
        Prediction, confidence, probabilities, selected model,
        model validation scores, and forecast horizon.
    """

    # ---------------------------------------------------------
    # 1. VALIDATION
    # ---------------------------------------------------------

    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            "prices must be a pandas DataFrame"
        )

    if prices.empty:
        raise ValueError(
            "prices cannot be empty"
        )

    if horizon not in (1, 5, 21):
        raise ValueError(
            "horizon must be 1, 5, or 21 trading days"
        )

    if "Adj Close" not in prices.columns:
        raise ValueError(
            "prices must contain an 'Adj Close' column"
        )

    # ---------------------------------------------------------
    # 2. BUILD DATASET
    # ---------------------------------------------------------

    X, y = build_dataset(
        prices,
        horizon=horizon,
        close_col="Adj Close",
    )

    if len(X) < 100:
        raise ValueError(
            "Not enough historical observations for ML inference"
        )

    # ---------------------------------------------------------
    # 3. TIME-BASED TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------

    split_index = int(len(X) * 0.80)

    if split_index <= 0 or split_index >= len(X):
        raise ValueError(
            "Unable to create a valid time-based validation split"
        )

    X_train = X.iloc[:split_index]
    X_validation = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_validation = y.iloc[split_index:]

    # ---------------------------------------------------------
    # 4. CREATE CANDIDATE MODELS
    # ---------------------------------------------------------

    models = {
        "Random Forest": make_classifier(
            "random_forest",
            params={
                "n_estimators": 200,
                "max_depth": 5,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        "Gradient Boosting": make_classifier(
            "gradient_boosting",
            params={
                "n_estimators": 200,
                "max_depth": 3,
                "random_state": 42,
            },
        ),
        "Logistic Regression": make_classifier(
            "logistic_regression",
            params={
                "max_iter": 1000,
                "random_state": 42,
            },
        ),
    }

    # ---------------------------------------------------------
    # 5. VALIDATE MODELS
    # ---------------------------------------------------------

    model_scores: dict[str, float] = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        score = model.score(
            X_validation,
            y_validation,
        )

        model_scores[name] = float(score)

    # ---------------------------------------------------------
    # 6. SELECT BEST MODEL
    # ---------------------------------------------------------

    best_model_name = max(
        model_scores,
        key=model_scores.get,
    )

    best_model = models[
        best_model_name
    ]

    # ---------------------------------------------------------
    # 7. RETRAIN ON ALL HISTORICAL DATA
    # ---------------------------------------------------------

    best_model.fit(
        X,
        y,
    )

    # ---------------------------------------------------------
    # 8. LATEST FEATURE ROW
    # ---------------------------------------------------------

    latest_features = X.iloc[[-1]]

    # ---------------------------------------------------------
    # 9. GENERATE PREDICTION
    # ---------------------------------------------------------

    prediction = int(
        best_model.predict(
            latest_features
        )[0]
    )

    probabilities = (
        best_model.predict_proba(
            latest_features
        )[0]
    )

    probability_map = {
        int(cls): float(probability)
        for cls, probability in zip(
            best_model.classes_,
            probabilities,
        )
    }

    confidence = probability_map[
        prediction
    ]

    # ---------------------------------------------------------
    # 10. RETURN RESULT
    # ---------------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": probability_map,
        "model": best_model,
        "model_name": best_model_name,
        "model_scores": model_scores,
        "horizon": horizon,
    }