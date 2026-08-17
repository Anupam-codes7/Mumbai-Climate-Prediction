import json

metrics = {
    "regression": {
        "temperature": {
            "MAE": 0.4065,
            "RMSE": 0.5369,
            "R2": 0.9609,
            "training_time": 2.22
        },
        "rainfall": {
            "MAE": 0.3856,
            "RMSE": 0.9364,
            "R2": 0.3962,
            "training_time": 1.15
        },
        "wind_speed": {
            "MAE": 1.6010,
            "RMSE": 2.0584,
            "R2": 0.8630,
            "training_time": 2.12
        },
        "pressure": {
            "MAE": 0.4555,
            "RMSE": 0.5937,
            "R2": 0.9764,
            "training_time": 2.00
        },
        "humidity": {
            "MAE": 2.6430,
            "RMSE": 3.5983,
            "R2": 0.9676,
            "training_time": 2.28
        },
        "pm2_5": {
            "MAE": 3.1220,
            "RMSE": 4.7027,
            "R2": 0.9507,
            "training_time": 1.95
        }
    },

    "rain_occurrence": {
        "accuracy": 0.8417,
        "precision": 0.7850,
        "recall": 0.7683,
        "f1_score": 0.7766,
        "roc_auc": 0.9276
    }
}

with open(
    "models/model_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("Model metrics saved successfully!")
