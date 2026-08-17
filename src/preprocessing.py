import pandas as pd
import numpy as np


def create_features(df):
    """
    Create the same feature-engineering pipeline
    used during model training.
    """

    df = df.copy()

    # Ensure chronological order
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_columns = [
        "temperature",
        "rainfall",
        "wind_speed",
        "pressure",
        "humidity",
        "pm2_5"
    ]

    for column in lag_columns:
        df[f"{column}_lag_1"] = df[column].shift(1)

    # --------------------------------------------------------
    # Rolling averages
    # --------------------------------------------------------

    rolling_columns = [
        "temperature",
        "rainfall",
        "humidity",
        "pm2_5"
    ]

    for column in rolling_columns:

        df[f"{column}_rolling_3"] = (
            df[column].rolling(window=3).mean()
        )

        df[f"{column}_rolling_7"] = (
            df[column].rolling(window=7).mean()
        )

    # --------------------------------------------------------
    # Seasonality
    # --------------------------------------------------------

    df["day_of_year"] = df["date"].dt.dayofyear

    df["sin_day_of_year"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["cos_day_of_year"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    return df

def prepare_prediction_features(
    df,
    features,
    temperature,
    rainfall,
    wind_speed,
    pressure,
    humidity,
    pm2_5,
    prediction_date
):
    """
    Create the 23 features required for tomorrow's prediction.

    Historical data is used to calculate lag and rolling features.
    Today's climate values are appended as the latest observation.
    """

    df = df.copy()

    # Ensure date format
    df["date"] = pd.to_datetime(df["date"])

    # Today's climate observation
    today = pd.DataFrame([{
        "date": pd.to_datetime(prediction_date),
        "temperature": temperature,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "humidity": humidity,
        "pm2_5": pm2_5
    }])

    # Keep only required historical columns
    base_columns = [
        "date",
        "temperature",
        "rainfall",
        "wind_speed",
        "pressure",
        "humidity",
        "pm2_5"
    ]

    df = df[base_columns]

    # Add today's values
    df = pd.concat(
        [df, today],
        ignore_index=True
    )

    # Remove duplicate date if today's date already exists
    df = df.drop_duplicates(
        subset="date",
        keep="last"
    )

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Create engineered features
    engineered_df = create_features(df)

    # Get today's row
    today_features = engineered_df[
        engineered_df["date"] == pd.to_datetime(prediction_date)
    ]

    if today_features.empty:
        raise ValueError(
            "Prediction date was not found in the feature dataset."
        )

    # Select exact model features
    X_latest = today_features[features]

    # Verify no missing values
    if X_latest.isnull().any().any():
        raise ValueError(
            "Missing values found in prediction features."
        )

    return X_latest
    """
    Prepare the latest available row for next-day prediction.
    """

    engineered_df = create_features(df)

    # Remove rows where lag/rolling features are unavailable
    engineered_df = engineered_df.dropna().reset_index(drop=True)

    # Get the latest available observation
    latest_row = engineered_df.iloc[-1]

    # Select features in the exact order used during training
    X_latest = latest_row[features].to_frame().T

    return X_latest