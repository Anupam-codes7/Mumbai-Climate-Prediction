import numpy as np
import pandas as pd


# ============================================================
# REQUIRED RAW WEATHER + AIR QUALITY COLUMNS
# ============================================================

RAW_COLUMNS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "sunshine_duration",
    "wet_bulb_temperature_2m",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust"
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """
    Reproduce the feature-engineering pipeline used during
    Mumbai Climate Prediction model training.

    Input:
        DataFrame containing the 31 raw weather/air-quality columns.

    Output:
        DataFrame containing engineered features.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in RAW_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Ensure datetime
    # --------------------------------------------------------

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    # ========================================================
    # 1. TEMPORAL FEATURES
    # ========================================================

    df["year"] = df["time"].dt.year

    df["month"] = df["time"].dt.month

    df["day"] = df["time"].dt.day

    df["day_of_year"] = df["time"].dt.dayofyear

    df["day_of_week"] = df["time"].dt.dayofweek

    df["hour"] = df["time"].dt.hour

    # ========================================================
    # 2. CYCLICAL TIME FEATURES
    # ========================================================

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # ========================================================
    # 3. WIND-DIRECTION CYCLICAL FEATURES
    # ========================================================

    df["wind_direction_sin"] = np.sin(
        2 * np.pi * df["wind_direction_10m"] / 360
    )

    df["wind_direction_cos"] = np.cos(
        2 * np.pi * df["wind_direction_10m"] / 360
    )

    # ========================================================
    # 4. MUMBAI SEASONAL FEATURES
    # ========================================================

    df["is_monsoon"] = (
        df["month"]
        .isin([6, 7, 8, 9])
        .astype(int)
    )

    df["is_pre_monsoon"] = (
        df["month"]
        .isin([3, 4, 5])
        .astype(int)
    )

    df["is_post_monsoon"] = (
        df["month"]
        .isin([10, 11])
        .astype(int)
    )

    df["is_winter"] = (
        df["month"]
        .isin([12, 1, 2])
        .astype(int)
    )

    # ========================================================
    # 5. MULTI-SCALE LAG FEATURES
    # ========================================================

    lag_variables = [
        "temperature_2m",
        "rain",
        "wind_speed_10m",
        "pressure_msl",
        "relative_humidity_2m",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    lag_periods = [
        1, 3, 6, 12,
        24, 48, 72, 168
    ]

    for variable in lag_variables:

        for lag in lag_periods:

            df[
                f"{variable}_lag_{lag}h"
            ] = df[variable].shift(lag)

    # ========================================================
    # 6. ROLLING-WINDOW FEATURES
    # ========================================================

    rolling_variables = [
        "temperature_2m",
        "rain",
        "wind_speed_10m",
        "pressure_msl",
        "relative_humidity_2m",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    rolling_windows = [
        3, 6, 12,
        24, 72, 168
    ]

    for variable in rolling_variables:

        # IMPORTANT:
        # Original notebook uses shift(1)
        # to ensure only past observations
        # are included.

        past_values = df[variable].shift(1)

        for window in rolling_windows:

            df[
                f"{variable}_rolling_mean_{window}h"
            ] = (
                past_values
                .rolling(window)
                .mean()
            )

            df[
                f"{variable}_rolling_std_{window}h"
            ] = (
                past_values
                .rolling(window)
                .std()
            )

            df[
                f"{variable}_rolling_min_{window}h"
            ] = (
                past_values
                .rolling(window)
                .min()
            )

            df[
                f"{variable}_rolling_max_{window}h"
            ] = (
                past_values
                .rolling(window)
                .max()
            )


       # ========================================================
    # 7. RAINFALL-SPECIFIC FEATURES
    # ========================================================

    rain_past = df["rain"].shift(1)

    # --------------------------------------------------------
    # Rainfall totals over different historical windows
    # --------------------------------------------------------

    df["rain_total_3h"] = (
        rain_past
        .rolling(3)
        .sum()
    )

    df["rain_total_6h"] = (
        rain_past
        .rolling(6)
        .sum()
    )

    df["rain_total_12h"] = (
        rain_past
        .rolling(12)
        .sum()
    )

    df["rain_total_24h"] = (
        rain_past
        .rolling(24)
        .sum()
    )

    df["rain_total_48h"] = (
        rain_past
        .rolling(48)
        .sum()
    )

    df["rain_total_72h"] = (
        rain_past
        .rolling(72)
        .sum()
    )

    df["rain_total_168h"] = (
        rain_past
        .rolling(168)
        .sum()
    )

    # --------------------------------------------------------
    # Number of rainy hours
    # --------------------------------------------------------

    df["rain_hours_24h"] = (
        (rain_past > 0.1)
        .rolling(24)
        .sum()
    )

    df["rain_hours_72h"] = (
        (rain_past > 0.1)
        .rolling(72)
        .sum()
    )

    df["rain_hours_168h"] = (
        (rain_past > 0.1)
        .rolling(168)
        .sum()
    )

    # --------------------------------------------------------
    # Maximum rainfall intensity
    # --------------------------------------------------------

    df["rain_max_24h"] = (
        rain_past
        .rolling(24)
        .max()
    )

    df["rain_max_72h"] = (
        rain_past
        .rolling(72)
        .max()
    )

    df["rain_max_168h"] = (
        rain_past
        .rolling(168)
        .max()
    )

    # --------------------------------------------------------
    # Recent rainfall indicators
    # --------------------------------------------------------

    df["rained_previous_24h"] = (
        df["rain_hours_24h"] > 0
    ).astype(int)

    df["rained_previous_72h"] = (
        df["rain_hours_72h"] > 0
    ).astype(int)

    # ========================================================
    # 8. WEATHER + POLLUTION TREND FEATURES
    # ========================================================

    trend_variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "wind_speed_10m",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    trend_periods = [
        1, 3, 6, 12, 24
    ]

    for variable in trend_variables:

        for period in trend_periods:

            df[
                f"{variable}_change_{period}h"
            ] = (
                df[variable]
                - df[variable].shift(period)
            )

    # ========================================================
    # 9. DERIVED ATMOSPHERIC FEATURES
    # ========================================================

    df["temperature_humidity_index"] = (
        df["temperature_2m"]
        * df["relative_humidity_2m"]
    )

    df["apparent_temperature_difference"] = (
        df["apparent_temperature"]
        - df["temperature_2m"]
    )

    df["dew_point_depression"] = (
        df["temperature_2m"]
        - df["dew_point_2m"]
    )

    df["pressure_difference"] = (
        df["pressure_msl"]
        - df["surface_pressure"]
    )

    df["wind_humidity_interaction"] = (
        df["wind_speed_10m"]
        * df["relative_humidity_2m"]
    )

    df["wind_gust_ratio"] = (
        df["wind_gusts_10m"]
        / (
            df["wind_speed_10m"] + 0.1
        )
    )

    df["pm25_pm10_ratio"] = (
        df["pm2_5"]
        / (
            df["pm10"] + 0.1
        )
    )

    df["pollution_index"] = (
        df["pm2_5"]
        + df["pm10"]
    )

    df["gas_pollution_index"] = (
        df["carbon_monoxide"]
        + df["nitrogen_dioxide"]
        + df["sulphur_dioxide"]
    )

    df["cloud_cover_average"] = (
        (
            df["cloud_cover_low"]
            + df["cloud_cover_mid"]
            + df["cloud_cover_high"]
        ) / 3
    )

    df["total_radiation"] = (
        df["direct_radiation"]
        + df["diffuse_radiation"]
    )

    return df


# ============================================================
# LIVE PREDICTION FEATURE PREPARATION
# ============================================================

# ============================================================
# LIVE PREDICTION FEATURE PREPARATION
# ============================================================

# ============================================================
# LIVE PREDICTION FEATURE PREPARATION
# ============================================================

# ============================================================
# LIVE PREDICTION FEATURE PREPARATION
# ============================================================

def prepare_latest_features(
    historical_df,
    live_df,
    feature_columns
):
    """
    Combine historical/live observations, recreate the
    training features and return the latest valid feature row.
    """

    historical = historical_df.copy()
    live = live_df.copy()

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    historical["time"] = pd.to_datetime(
        historical["time"]
    )

    if not live.empty:
        live["time"] = pd.to_datetime(
            live["time"]
        )

    # --------------------------------------------------------
    # Combine data
    #
    # If live is empty, don't concatenate it.
    # This prevents pandas dtype conversion problems.
    # --------------------------------------------------------

    if live.empty:

        combined = historical.copy()

    else:

        combined = pd.concat(
            [
                historical,
                live
            ],
            ignore_index=True
        )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    combined = (
        combined
        .drop_duplicates(
            subset="time",
            keep="last"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Make sure numeric weather columns are actually numeric
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in combined.columns
        if column != "time"
    ]

    combined[numeric_columns] = (
        combined[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # Create exactly the same engineered features
    # used during model training
    # --------------------------------------------------------

    engineered = create_features(
        combined
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    engineered = engineered.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Find latest row with all required features
    # --------------------------------------------------------

    valid = engineered.dropna(
        subset=feature_columns
    )

    if valid.empty:

        raise ValueError(
            "No valid feature row available. "
            "More historical data is required."
        )

    latest = valid.iloc[[-1]]

    # --------------------------------------------------------
    # Exact feature order used during training
    # --------------------------------------------------------

    latest_features = latest[
        feature_columns
    ].copy()

    return latest_features
    """
    Combine historical/live observations, recreate the
    training features and return the latest valid feature row.
    """

    historical = historical_df.copy()
    live = live_df.copy()

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    historical["time"] = pd.to_datetime(
        historical["time"]
    )

    if not live.empty:
        live["time"] = pd.to_datetime(
            live["time"]
        )

    # --------------------------------------------------------
    # Combine data
    #
    # If live is empty, don't concatenate it.
    # This prevents pandas dtype conversion problems.
    # --------------------------------------------------------

    if live.empty:

        combined = historical.copy()

    else:

        combined = pd.concat(
            [
                historical,
                live
            ],
            ignore_index=True
        )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    combined = (
        combined
        .drop_duplicates(
            subset="time",
            keep="last"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Make sure numeric weather columns are actually numeric
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in combined.columns
        if column != "time"
    ]

    combined[numeric_columns] = (
        combined[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # Create exactly the same engineered features
    # used during model training
    # --------------------------------------------------------

    engineered = create_features(
        combined
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    engineered = engineered.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Find latest row with all required features
    # --------------------------------------------------------

    valid = engineered.dropna(
        subset=feature_columns
    )

    if valid.empty:

        raise ValueError(
            "No valid feature row available. "
            "More historical data is required."
        )

    latest = valid.iloc[[-1]]

    # --------------------------------------------------------
    # Exact feature order used during training
    # --------------------------------------------------------

    latest_features = latest[
        feature_columns
    ].copy()

    return latest_features
    """
    Combine historical/live observations, recreate the
    training features and return the latest valid feature row.
    """

    historical = historical_df.copy()
    live = live_df.copy()

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    historical["time"] = pd.to_datetime(
        historical["time"]
    )

    if not live.empty:
        live["time"] = pd.to_datetime(
            live["time"]
        )

    # --------------------------------------------------------
    # Combine data
    #
    # If live is empty, don't concatenate it.
    # This prevents pandas dtype conversion problems.
    # --------------------------------------------------------

    if live.empty:

        combined = historical.copy()

    else:

        combined = pd.concat(
            [
                historical,
                live
            ],
            ignore_index=True
        )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    combined = (
        combined
        .drop_duplicates(
            subset="time",
            keep="last"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Make sure numeric weather columns are actually numeric
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in combined.columns
        if column != "time"
    ]

    combined[numeric_columns] = (
        combined[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # Create exactly the same engineered features
    # used during model training
    # --------------------------------------------------------

    engineered = create_features(
        combined
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    engineered = engineered.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Find latest row with all required features
    # --------------------------------------------------------

    valid = engineered.dropna(
        subset=feature_columns
    )

    if valid.empty:

        raise ValueError(
            "No valid feature row available. "
            "More historical data is required."
        )

    latest = valid.iloc[[-1]]

    # --------------------------------------------------------
    # Exact feature order used during training
    # --------------------------------------------------------

    latest_features = latest[
        feature_columns
    ].copy()

    return latest_features
    """
    Combine historical data with live API data,
    recreate the training features and return
    the latest valid 476-feature row.

    Parameters
    ----------
    historical_df : DataFrame
        Historical 31-column Mumbai dataset.

    live_df : DataFrame
        Latest API weather + air-quality observations.

    feature_columns : list
        Exact feature order saved during model training.

    Returns
    -------
    DataFrame
        One-row DataFrame containing exactly
        the required model features.
    """

    historical = historical_df.copy()
    live = live_df.copy()

    historical["time"] = pd.to_datetime(
        historical["time"]
    )

    live["time"] = pd.to_datetime(
        live["time"]
    )

    # Combine historical and live observations
    combined = pd.concat(
        [
            historical,
            live
        ],
        ignore_index=True
    )

    # Remove duplicate timestamps
    combined = (
        combined
        .drop_duplicates(
            subset="time",
            keep="last"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Create exactly the same engineered features
    engineered = create_features(combined)

    # Replace infinite values
    engineered = engineered.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Find latest row with all required features
    valid = engineered.dropna(
        subset=feature_columns
    )

    if valid.empty:
        raise ValueError(
            "No valid feature row available. "
            "More historical data is required."
        )

    latest = valid.iloc[[-1]]

    # Ensure exact training feature order
    latest_features = latest[
        feature_columns
    ].copy()

    return latest_features
