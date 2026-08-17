import json

with open("models/prediction.json") as f:
    prediction = json.load(f)["prediction"]

with open("models/actual_weather.json") as f:
    actual = json.load(f)["actual"]

comparison = {
    "temperature_C": {
        "predicted": prediction["temperature_C"],
        "actual": actual["temperature_C"],
        "absolute_error": abs(
            prediction["temperature_C"]
            - actual["temperature_C"]
        )
    },

    "rainfall_mm": {
        "predicted": prediction["rainfall_mm"],
        "actual": actual["rainfall_mm"],
        "absolute_error": abs(
            prediction["rainfall_mm"]
            - actual["rainfall_mm"]
        )
    },

    "wind_speed_kmh": {
        "predicted": prediction["wind_speed_kmh"],
        "actual": actual["wind_speed_kmh"],
        "absolute_error": abs(
            prediction["wind_speed_kmh"]
            - actual["wind_speed_kmh"]
        )
    },

    "pressure_hPa": {
        "predicted": prediction["pressure_hPa"],
        "actual": actual["pressure_hPa"],
        "absolute_error": abs(
            prediction["pressure_hPa"]
            - actual["pressure_hPa"]
        )
    },

    "humidity_percent": {
        "predicted": prediction["humidity_percent"],
        "actual": actual["humidity_percent"],
        "absolute_error": abs(
            prediction["humidity_percent"]
            - actual["humidity_percent"]
        )
    },

    "pm2_5": {
        "predicted": prediction["pm2_5"],
        "actual": actual["pm2_5"],
        "absolute_error": abs(
            prediction["pm2_5"]
            - actual["pm2_5"]
        )
    }
}

with open(
    "models/comparison.json",
    "w"
) as f:
    json.dump(
        comparison,
        f,
        indent=4
    )

print("Comparison created successfully!")

for variable, values in comparison.items():
    print(
        f"{variable}: "
        f"Predicted={values['predicted']:.2f}, "
        f"Actual={values['actual']:.2f}, "
        f"Error={values['absolute_error']:.2f}"
    )
