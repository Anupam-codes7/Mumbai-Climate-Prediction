import requests
import pandas as pd


# ============================================================
# MUMBAI LOCATION
# ============================================================

MUMBAI_LATITUDE = 19.0760
MUMBAI_LONGITUDE = 72.8777


WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# WEATHER DATA
# ============================================================

def fetch_current_weather():

    weather_params = {
        "latitude": MUMBAI_LATITUDE,
        "longitude": MUMBAI_LONGITUDE,

        "hourly": [
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
            "wet_bulb_temperature_2m"
        ],

        # 7 days of historical hourly data
        # + 1 day of current/future data
        "past_days": 7,
        "forecast_days": 1,

        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        WEATHER_URL,
        params=weather_params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# AIR QUALITY DATA
# ============================================================

def fetch_air_quality():

    air_params = {
        "latitude": MUMBAI_LATITUDE,
        "longitude": MUMBAI_LONGITUDE,

        "hourly": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "aerosol_optical_depth",
            "dust"
        ],

        # Same time range as weather data
        "past_days": 7,
        "forecast_days": 1,

        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=air_params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# COMBINE WEATHER + AIR QUALITY
# ============================================================

def get_live_mumbai_data():

    print("Fetching Mumbai weather history...")

    weather_data = fetch_current_weather()

    print("Fetching Mumbai air-quality history...")

    air_data = fetch_air_quality()

    # --------------------------------------------------------
    # Convert weather data to DataFrame
    # --------------------------------------------------------

    weather_hourly = pd.DataFrame(
        weather_data["hourly"]
    )

    # --------------------------------------------------------
    # Convert air-quality data to DataFrame
    # --------------------------------------------------------

    air_hourly = pd.DataFrame(
        air_data["hourly"]
    )

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    weather_hourly["time"] = pd.to_datetime(
        weather_hourly["time"]
    )

    air_hourly["time"] = pd.to_datetime(
        air_hourly["time"]
    )

    # --------------------------------------------------------
    # Merge on timestamp
    # --------------------------------------------------------

    merged = pd.merge(
        weather_hourly,
        air_hourly,
        on="time",
        how="inner"
    )

    # --------------------------------------------------------
    # Sort and remove duplicate timestamps
    # --------------------------------------------------------

    merged = (
        merged
        .drop_duplicates(
            subset="time"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    return merged


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MUMBAI LIVE WEATHER + AIR QUALITY")
    print("=" * 60)

    df = get_live_mumbai_data()

    print("\nLive data fetched successfully!")

    print("Shape:", df.shape)

    print("\nTime range:")

    print("Start:", df["time"].min())
    print("End  :", df["time"].max())

    print("\nNumber of columns:", len(df.columns))

    print("\nMissing values:", df.isna().sum().sum())

    print(
        "Duplicate timestamps:",
        df["time"].duplicated().sum()
    )

    print("\nLatest observation:")

    print(
        df.tail(1).T
    )