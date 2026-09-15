import os
import sys
import json
import joblib
import pandas as pd

from huggingface_hub import hf_hub_download

# Allow imports from src/
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from weather_api import get_live_mumbai_data
from feature_engineering import prepare_latest_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# HUGGING FACE MODEL REPOSITORY
# ============================================================

HF_REPO_ID = "ezdevz/mumbai-climate-ml-models"


# ============================================================
# LOAD FEATURE CONFIGURATION
# ============================================================

print("=" * 60)
print("MUMBAI CLIMATE ML - LIVE PREDICTION")
print("=" * 60)

print("\nLoading feature configuration...")

config = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_config.joblib"
    )
)

feature_columns = config["feature_columns"]

print(
    "Expected features:",
    len(feature_columns)
)


# ============================================================
# LOAD MODELS
# ============================================================

model_names = [
    "temperature",
    "rain",
    "rain_occurrence",
    "wind_speed",
    "pressure",
    "humidity",
    "pm2_5"
]

models = {}

print("\nLoading ML models...")

for name in model_names:

    filename = f"extra_trees_{name}.joblib"

    local_path = os.path.join(
        MODEL_DIR,
        filename
    )

    # --------------------------------------------------------
    # Use local model if available
    # --------------------------------------------------------

    if os.path.exists(local_path):

        print(
            f"Loading local model: {filename}"
        )

        model_path = local_path

    # --------------------------------------------------------
    # Otherwise download from Hugging Face
    # --------------------------------------------------------

    else:

        print(
            f"Downloading model from Hugging Face: {filename}"
        )

        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename
        )

    models[name] = joblib.load(model_path)

    print(
        f"Loaded: {filename}"
    )


print(
    f"\nAll {len(models)} ML models loaded successfully!"
)


# ============================================================
# FETCH LIVE DATA
# ============================================================

print("\nFetching live Mumbai data...")

live_df = get_live_mumbai_data()


# ============================================================
# SAVE LATEST ACTUAL WEATHER FOR DASHBOARD
# ============================================================

latest_actual = live_df.iloc[-1]

actual_weather_data = {
    "timestamp": str(latest_actual["time"]),

    "actual": {
        "temperature_C": float(
            latest_actual["temperature_2m"]
        ),

        "rainfall_mm": float(
            latest_actual["rain"]
        ),

        "wind_speed_kmh": float(
            latest_actual["wind_speed_10m"]
        ),

        "pressure_hPa": float(
            latest_actual["pressure_msl"]
        ),

        "humidity_percent": float(
            latest_actual["relative_humidity_2m"]
        ),

        "pm2_5": float(
            latest_actual["pm2_5"]
        )
    }
}

with open(
    os.path.join(
        MODEL_DIR,
        "actual_weather.json"
    ),
    "w"
) as f:

    json.dump(
        actual_weather_data,
        f,
        indent=4
    )


print(
    "\nActual weather saved to models/actual_weather.json"
)

print(
    "Live data shape:",
    live_df.shape
)

print(
    "Time range:",
    live_df["time"].min(),
    "to",
    live_df["time"].max()
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating prediction features...")

X = prepare_latest_features(
    historical_df=live_df,
    live_df=pd.DataFrame(
        columns=live_df.columns
    ),
    feature_columns=feature_columns
)

print(
    "Prediction feature shape:",
    X.shape
)

if X.shape != (1, 476):

    raise ValueError(
        f"Unexpected feature shape: {X.shape}. "
        "Expected (1, 476)."
    )


print(
    "476-feature compatibility: OK"
)


# ============================================================
# PREDICTIONS
# ============================================================

predicted_temperature = models[
    "temperature"
].predict(X)[0]

predicted_rainfall = models[
    "rain"
].predict(X)[0]

predicted_rain_occurrence = models[
    "rain_occurrence"
].predict(X)[0]

rain_probability = models[
    "rain_occurrence"
].predict_proba(X)[0][1]

predicted_wind_speed = models[
    "wind_speed"
].predict(X)[0]

predicted_pressure = models[
    "pressure"
].predict(X)[0]

predicted_humidity = models[
    "humidity"
].predict(X)[0]

predicted_pm25 = models[
    "pm2_5"
].predict(X)[0]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("NEXT-HOUR MUMBAI WEATHER PREDICTION")
print("=" * 60)

print(
    f"Temperature      : {predicted_temperature:.2f} C"
)

print(
    f"Rainfall         : {max(0, predicted_rainfall):.2f} mm"
)

print(
    f"Rain occurrence  : {int(predicted_rain_occurrence)}"
)

print(
    f"Rain probability : {rain_probability:.2%}"
)

print(
    f"Wind speed       : {predicted_wind_speed:.2f} km/h"
)

print(
    f"Pressure         : {predicted_pressure:.2f} hPa"
)

print(
    f"Humidity         : {predicted_humidity:.2f} %"
)

print(
    f"PM2.5            : {predicted_pm25:.2f} ug/m3"
)

print("=" * 60)


# ============================================================
# SAVE PREDICTIONS FOR DASHBOARD
# ============================================================

from datetime import datetime

prediction_data = {
    "timestamp": datetime.now().isoformat(),

    "prediction": {
        "temperature_C": float(
            predicted_temperature
        ),

        "rainfall_mm": float(
            max(0, predicted_rainfall)
        ),

        "rain_occurrence": int(
            predicted_rain_occurrence
        ),

        "rain_probability": float(
            rain_probability
        ),

        "wind_speed_kmh": float(
            predicted_wind_speed
        ),

        "pressure_hPa": float(
            predicted_pressure
        ),

        "humidity_percent": float(
            predicted_humidity
        ),

        "pm2_5": float(
            predicted_pm25
        )
    }
}

with open(
    os.path.join(
        MODEL_DIR,
        "prediction.json"
    ),
    "w"
) as f:

    json.dump(
        prediction_data,
        f,
        indent=4
    )


print(
    "\nPrediction saved to models/prediction.json"
)