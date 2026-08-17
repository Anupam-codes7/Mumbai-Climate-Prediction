import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# Load feature configuration
# --------------------------------------------------

feature_config = joblib.load(
    os.path.join(MODEL_DIR, "feature_config.joblib")
)

FEATURE_COLUMNS = feature_config["feature_columns"]


# --------------------------------------------------
# Load models
# --------------------------------------------------

MODELS = {
    "temperature": joblib.load(
        os.path.join(
            MODEL_DIR,
            "extra_trees_temperature.joblib"
        )
    ),

    "rain": joblib.load(
        os.path.join(
            MODEL_DIR,
            "extra_trees_rain.joblib"
        )
    ),

    "rain_occurrence": joblib.load(
        os.path.join(
            MODEL_DIR,
            "extra_trees_rain_occurrence.joblib"
        )
    ),

    "wind_speed": joblib.load(
        os.path.join(
            MODEL_DIR,
            "extra_trees_wind_speed.joblib"
        )
    ),

    "pressure": joblib.load(
        os.path.join(
            MODEL_DIR,
            "extra_trees_pressure.joblib"
        )
    ),

    "humidity": joblib.load(
        os.path.join(
            MODEL_DIR,
            "extra_trees_humidity.joblib"
        )
    ),

    "pm2_5": joblib.load(
        os.path.join(
            MODEL_DIR,
            "extra_trees_pm2_5.joblib"
        )
    )
}


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_next_hour(feature_row):

    if isinstance(feature_row, pd.Series):
        feature_row = feature_row.to_frame().T

    # Make sure feature order is identical
    X = feature_row[FEATURE_COLUMNS].copy()

    # Predictions
    temperature = MODELS["temperature"].predict(X)[0]

    rain_occurrence = MODELS[
        "rain_occurrence"
    ].predict(X)[0]

    rain_probability = MODELS[
        "rain_occurrence"
    ].predict_proba(X)[0, 1]

    rainfall = MODELS["rain"].predict(X)[0]

    wind_speed = MODELS[
        "wind_speed"
    ].predict(X)[0]

    pressure = MODELS[
        "pressure"
    ].predict(X)[0]

    humidity = MODELS[
        "humidity"
    ].predict(X)[0]

    pm2_5 = MODELS[
        "pm2_5"
    ].predict(X)[0]

    # Prevent negative rainfall
    rainfall = max(0, rainfall)

    # Approach B:
    # If rain classifier says no rain,
    # rainfall amount becomes zero.
    if rain_occurrence == 0:
        rainfall = 0.0

    return {
        "temperature_C": float(temperature),
        "rain_occurrence": int(rain_occurrence),
        "rain_probability": float(rain_probability),
        "rainfall_mm": float(rainfall),
        "wind_speed_kmh": float(wind_speed),
        "pressure_hPa": float(pressure),
        "humidity_percent": float(humidity),
        "pm2_5": float(pm2_5)
    }