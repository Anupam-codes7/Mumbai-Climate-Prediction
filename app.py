import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mumbai Climate ML Prediction",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌦️ Mumbai Climate ML Prediction")
st.write("Live Weather Intelligence & Next-Hour Prediction")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PREDICTION_FILE = os.path.join(
    MODELS_DIR,
    "prediction.json"
)

ACTUAL_FILE = os.path.join(
    MODELS_DIR,
    "actual_weather.json"
)

METRICS_FILE = os.path.join(
    MODELS_DIR,
    "model_metrics.json"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not os.path.exists(path):
        return None

    try:

        with open(path, "r") as f:
            return json.load(f)

    except Exception:

        return None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

st.sidebar.caption(
    "Fetch live Mumbai weather and generate a new "
    "next-hour prediction."
)


# ============================================================
# REFRESH BUTTON
# ============================================================

if st.sidebar.button(
    "🔄 Refresh Live Prediction",
    use_container_width=True
):

    with st.spinner(
        "Fetching live Mumbai weather and generating prediction..."
    ):

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(
                    BASE_DIR,
                    "src",
                    "live_prediction.py"
                )
            ],
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

    if result.returncode == 0:

        st.sidebar.success(
            "Prediction updated successfully!"
        )

        st.rerun()

    else:

        st.sidebar.error(
            "Prediction update failed."
        )

        if result.stderr:

            st.sidebar.code(
                result.stderr
            )


# ============================================================
# SIDEBAR PROJECT INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("📌 Project")

st.sidebar.write(
    "**Mumbai Climate ML Prediction**"
)

st.sidebar.caption(
    "A machine-learning based system for "
    "next-hour weather and air-quality prediction."
)


# ============================================================
# ML PIPELINE
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("🔧 ML Pipeline")

st.sidebar.write(
    "• 476 engineered features"
)

st.sidebar.write(
    "• 7 trained ML models"
)

st.sidebar.write(
    "• Extra Trees algorithm"
)

st.sidebar.write(
    "• Weather + air-quality data"
)


# ============================================================
# SIDEBAR INSTRUCTION
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Use the refresh button to fetch the latest "
    "available data and generate a new prediction."
)

# ============================================================
# LOAD DATA
# ============================================================

prediction_data = load_json(
    PREDICTION_FILE
)

actual_data = load_json(
    ACTUAL_FILE
)

metrics_data = load_json(
    METRICS_FILE
)


# ============================================================
# DATA VALIDATION
# ============================================================

if prediction_data is None:

    st.warning(
        "Prediction data not found. "
        "Click 'Refresh Live Prediction'."
    )

    st.stop()


if actual_data is None:

    st.warning(
        "Actual weather data not found. "
        "Click 'Refresh Live Prediction'."
    )

    st.stop()


if metrics_data is None:

    st.warning(
        "Model metrics file not found."
    )

    st.stop()


prediction = prediction_data["prediction"]
actual = actual_data["actual"]

prediction_timestamp = prediction_data["timestamp"]
actual_timestamp = actual_data["timestamp"]


# ============================================================
# DATA FRESHNESS
# ============================================================

st.caption(
    f"🕒 Latest weather observation: {actual_timestamp}"
)

st.caption(
    f"🤖 ML prediction generated: {prediction_timestamp}"
)



# ============================================================
# SECTION 1 — CURRENT WEATHER
# ============================================================

st.divider()

st.header("🌍 Current Mumbai Weather")

st.caption(
    f"Latest observation: {actual_timestamp}"
)


# ----------------------------
# Weather Row 1
# ----------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🌡️ Temperature",
        f"{actual['temperature_C']:.1f} °C"
    )


with col2:

    st.metric(
        "💧 Humidity",
        f"{actual['humidity_percent']:.1f}%"
    )


with col3:

    st.metric(
        "🌬️ Wind Speed",
        f"{actual['wind_speed_kmh']:.1f} km/h"
    )


# ----------------------------
# Weather Row 2
# ----------------------------

col4, col5, col6 = st.columns(3)


with col4:

    st.metric(
        "🔵 Pressure",
        f"{actual['pressure_hPa']:.1f} hPa"
    )


with col5:

    st.metric(
        "🌧️ Rainfall",
        f"{actual['rainfall_mm']:.2f} mm"
    )


with col6:

    st.metric(
        "🫁 PM2.5",
        f"{actual['pm2_5']:.1f} µg/m³"
    )

# ============================================================
# CURRENT WEATHER INTERPRETATION
# ============================================================

st.subheader("📝 Current Conditions")


temperature = actual["temperature_C"]
humidity = actual["humidity_percent"]
wind = actual["wind_speed_kmh"]
pm25 = actual["pm2_5"]
rain = actual["rainfall_mm"]


# Temperature description

if temperature < 20:
    temperature_status = "Cool"
elif temperature < 28:
    temperature_status = "Comfortably warm"
elif temperature < 33:
    temperature_status = "Warm"
else:
    temperature_status = "Hot"


# Humidity description

if humidity < 40:
    humidity_status = "Low humidity"
elif humidity < 70:
    humidity_status = "Moderate humidity"
elif humidity < 85:
    humidity_status = "High humidity"
else:
    humidity_status = "Very high humidity"


# Wind description

if wind < 10:
    wind_status = "Light winds"
elif wind < 20:
    wind_status = "Moderate winds"
elif wind < 35:
    wind_status = "Strong winds"
else:
    wind_status = "Very strong winds"


# Rain description

if rain > 0.1:
    rain_status_current = "Rain currently observed"
else:
    rain_status_current = "No significant rainfall currently"


# PM2.5 simple project-level interpretation

if pm25 <= 12:
    pm_status = "Low PM2.5"
elif pm25 <= 35.4:
    pm_status = "Moderate PM2.5"
elif pm25 <= 55.4:
    pm_status = "Elevated PM2.5"
else:
    pm_status = "High PM2.5"


interpretation_col1, interpretation_col2, interpretation_col3 = st.columns(3)


with interpretation_col1:

    st.info(
        f"🌡️ **Temperature:** {temperature_status}\n\n"
        f"Current temperature is **{temperature:.1f} °C**."
    )


with interpretation_col2:

    st.info(
        f"💧 **Atmosphere:** {humidity_status}\n\n"
        f"Humidity is **{humidity:.1f}%** with {wind_status.lower()}."
    )


with interpretation_col3:

    st.info(
        f"🫁 **Air Quality:** {pm_status}\n\n"
        f"PM2.5 concentration is **{pm25:.1f} µg/m³**."
    )


st.caption(
    f"🌧️ Rain status: {rain_status_current}"
)

# ============================================================
# SECTION 2 — ML NEXT-HOUR PREDICTION
# ============================================================

st.divider()

st.header("🤖 ML Next-Hour Prediction")

st.caption(
    "Extra Trees ML ensemble • 476 engineered features • "
    "Next-hour forecasting"
)

st.caption(
    f"Prediction generated: {prediction_timestamp}"
)


# ----------------------------
# Prediction Row 1
# ----------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🌡️ Temperature",
        f"{prediction['temperature_C']:.2f} °C"
    )


with col2:

    st.metric(
        "🌧️ Rainfall",
        f"{prediction['rainfall_mm']:.2f} mm"
    )


with col3:

    rain_status = (
        "Rain Expected"
        if prediction["rain_occurrence"] == 1
        else "No Rain"
    )

    st.metric(
        "☔ Rain Status",
        rain_status
    )


with col4:

    st.metric(
        "🌧️ Rain Probability",
        f"{prediction['rain_probability']:.1%}"
    )


# ----------------------------
# Prediction Row 2
# ----------------------------

col5, col6, col7 = st.columns(3)


with col5:

    st.metric(
        "🌬️ Wind Speed",
        f"{prediction['wind_speed_kmh']:.2f} km/h"
    )


with col6:

    st.metric(
        "💧 Humidity",
        f"{prediction['humidity_percent']:.2f}%"
    )


with col7:

    st.metric(
        "🫁 PM2.5",
        f"{prediction['pm2_5']:.2f} µg/m³"
    )


# ============================================================
# ML PREDICTION INTERPRETATION
# ============================================================

if prediction["rain_probability"] >= 0.70:

    prediction_message = (
        "🌧️ **High chance of rain in the next hour.** "
        "The ML model estimates a "
        f"{prediction['rain_probability']:.1%} probability."
    )

elif prediction["rain_probability"] >= 0.40:

    prediction_message = (
        "🌦️ **Moderate chance of rain in the next hour.** "
        "The ML model estimates a "
        f"{prediction['rain_probability']:.1%} probability."
    )

else:

    prediction_message = (
        "☀️ **Low chance of rain in the next hour.** "
        "The ML model estimates a "
        f"{prediction['rain_probability']:.1%} probability."
    )

st.info(prediction_message)


# ============================================================
# SECTION 3 — RAIN PROBABILITY GAUGE
# ============================================================

st.divider()

st.header("🌧️ Rain Probability")

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=prediction["rain_probability"] * 100,
        number={
            "suffix": "%",
            "font": {
                "size": 40
            }
        },
        title={
            "text": "Next-Hour Rain Probability"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "bar": {
                "color": "#4F8BF9"
            },
            "steps": [
                {
                    "range": [0, 30],
                    "color": "#E8F5E9"
                },
                {
                    "range": [30, 70],
                    "color": "#FFF8E1"
                },
                {
                    "range": [70, 100],
                    "color": "#FFEBEE"
                }
            ]
        }
    )
)

gauge.update_layout(
    height=300,
    margin={
        "l": 30,
        "r": 30,
        "t": 60,
        "b": 20
    }
)

st.plotly_chart(
    gauge,
    use_container_width=True
)


# ============================================================
# SECTION 4 — PREDICTION ANALYSIS
# ============================================================

st.divider()

st.header("📊 Prediction Analysis")

st.caption(
    "ML forecast compared with the latest available observation. "
    "Each variable is shown using its own unit and scale."
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def create_comparison_chart(
    title,
    predicted,
    actual_value,
    unit
):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["ML Prediction", "Latest Observation"],
            y=[predicted, actual_value],
            text=[
                f"{predicted:.2f} {unit}",
                f"{actual_value:.2f} {unit}"
            ],
            textposition="outside",
            name=title
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title=unit,
        showlegend=False,
        height=330,
        margin={
            "l": 40,
            "r": 40,
            "t": 60,
            "b": 40
        }
    )

    return fig


# ============================================================
# ROW 1 — TEMPERATURE / RAINFALL / WIND
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    fig_temperature = create_comparison_chart(
        "🌡️ Temperature",
        prediction["temperature_C"],
        actual["temperature_C"],
        "°C"
    )

    st.plotly_chart(
        fig_temperature,
        use_container_width=True
    )


with col2:

    fig_rainfall = create_comparison_chart(
        "🌧️ Rainfall",
        prediction["rainfall_mm"],
        actual["rainfall_mm"],
        "mm"
    )

    st.plotly_chart(
        fig_rainfall,
        use_container_width=True
    )


with col3:

    fig_wind = create_comparison_chart(
        "🌬️ Wind Speed",
        prediction["wind_speed_kmh"],
        actual["wind_speed_kmh"],
        "km/h"
    )

    st.plotly_chart(
        fig_wind,
        use_container_width=True
    )


# ============================================================
# ROW 2 — PRESSURE / HUMIDITY / PM2.5
# ============================================================

col4, col5, col6 = st.columns(3)


with col4:

    fig_pressure = create_comparison_chart(
        "🔵 Pressure",
        prediction["pressure_hPa"],
        actual["pressure_hPa"],
        "hPa"
    )

    st.plotly_chart(
        fig_pressure,
        use_container_width=True
    )


with col5:

    fig_humidity = create_comparison_chart(
        "💧 Humidity",
        prediction["humidity_percent"],
        actual["humidity_percent"],
        "%"
    )

    st.plotly_chart(
        fig_humidity,
        use_container_width=True
    )


with col6:

    fig_pm25 = create_comparison_chart(
        "🫁 PM2.5",
        prediction["pm2_5"],
        actual["pm2_5"],
        "µg/m³"
    )

    st.plotly_chart(
        fig_pm25,
        use_container_width=True
    )


# ============================================================
# ABSOLUTE DIFFERENCE
# ============================================================

st.subheader("📉 Absolute Difference")


variables = [
    "Temperature",
    "Rainfall",
    "Wind Speed",
    "Pressure",
    "Humidity",
    "PM2.5"
]


errors = [

    abs(
        prediction["temperature_C"]
        - actual["temperature_C"]
    ),

    abs(
        prediction["rainfall_mm"]
        - actual["rainfall_mm"]
    ),

    abs(
        prediction["wind_speed_kmh"]
        - actual["wind_speed_kmh"]
    ),

    abs(
        prediction["pressure_hPa"]
        - actual["pressure_hPa"]
    ),

    abs(
        prediction["humidity_percent"]
        - actual["humidity_percent"]
    ),

    abs(
        prediction["pm2_5"]
        - actual["pm2_5"]
    )

]


units = [
    "°C",
    "mm",
    "km/h",
    "hPa",
    "%",
    "µg/m³"
]


fig_error = go.Figure()


fig_error.add_trace(
    go.Bar(
        x=variables,
        y=errors,
        text=[
            f"{error:.2f}"
            for error in errors
        ],
        textposition="outside",
        name="Absolute Difference"
    )
)


fig_error.update_layout(
    title="Prediction Difference by Variable",
    xaxis_title="Variable",
    yaxis_title="Absolute Difference",
    height=450,
    showlegend=False
)


st.plotly_chart(
    fig_error,
    use_container_width=True
)


# ============================================================
# ERROR SUMMARY
# ============================================================

st.subheader("📋 Prediction Difference Summary")


error_table = pd.DataFrame({

    "Variable": variables,

    "ML Prediction": [
        prediction["temperature_C"],
        prediction["rainfall_mm"],
        prediction["wind_speed_kmh"],
        prediction["pressure_hPa"],
        prediction["humidity_percent"],
        prediction["pm2_5"]
    ],

    "Latest Observation": [
        actual["temperature_C"],
        actual["rainfall_mm"],
        actual["wind_speed_kmh"],
        actual["pressure_hPa"],
        actual["humidity_percent"],
        actual["pm2_5"]
    ],

    "Absolute Difference": errors,

    "Unit": units

})


st.dataframe(
    error_table,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# SECTION 5 — MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("🏆 Model Performance")

st.caption(
    "Extra Trees performance on the held-out test data"
)


regression = metrics_data[
    "regression"
]


# ============================================================
# R² COMPARISON
# ============================================================

st.subheader(
    "Regression Model Performance — R²"
)


r2_targets = [
    "Temperature",
    "Rainfall",
    "Wind Speed",
    "Pressure",
    "Humidity",
    "PM2.5"
]


r2_values = [

    regression[
        "temperature"
    ]["R2"],

    regression[
        "rainfall"
    ]["R2"],

    regression[
        "wind_speed"
    ]["R2"],

    regression[
        "pressure"
    ]["R2"],

    regression[
        "humidity"
    ]["R2"],

    regression[
        "pm2_5"
    ]["R2"]

]


fig_r2 = go.Figure(

    go.Bar(
        x=r2_values,
        y=r2_targets,
        orientation="h",
        text=[
            f"{value:.4f}"
            for value in r2_values
        ],
        textposition="outside"
    )

)


fig_r2.update_layout(
    title="R² Score by Prediction Target",
    xaxis_title="R² Score",
    yaxis_title="Target",
    xaxis={
        "range": [0, 1.05]
    },
    height=450
)


st.plotly_chart(
    fig_r2,
    use_container_width=True
)


# ============================================================
# REGRESSION METRICS TABLE
# ============================================================

st.subheader(
    "Regression Metrics"
)


regression_table = pd.DataFrame({

    "Target": r2_targets,

    "MAE": [

        regression[
            "temperature"
        ]["MAE"],

        regression[
            "rainfall"
        ]["MAE"],

        regression[
            "wind_speed"
        ]["MAE"],

        regression[
            "pressure"
        ]["MAE"],

        regression[
            "humidity"
        ]["MAE"],

        regression[
            "pm2_5"
        ]["MAE"]

    ],

    "RMSE": [

        regression[
            "temperature"
        ]["RMSE"],

        regression[
            "rainfall"
        ]["RMSE"],

        regression[
            "wind_speed"
        ]["RMSE"],

        regression[
            "pressure"
        ]["RMSE"],

        regression[
            "humidity"
        ]["RMSE"],

        regression[
            "pm2_5"
        ]["RMSE"]

    ],

    "R²": r2_values

})


st.dataframe(
    regression_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SECTION 6 — RAINFALL CLASSIFICATION
# ============================================================

st.divider()

st.header(
    "🌧️ Rainfall Occurrence Classification"
)

st.caption(
    "Extra Trees classifier performance on the held-out test data"
)


rain_metrics = metrics_data[
    "rain_occurrence"
]


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Accuracy",
        f"{rain_metrics['accuracy']:.2%}"
    )


with col2:

    st.metric(
        "Precision",
        f"{rain_metrics['precision']:.2%}"
    )


with col3:

    st.metric(
        "Recall",
        f"{rain_metrics['recall']:.2%}"
    )


with col4:

    st.metric(
        "F1 Score",
        f"{rain_metrics['f1_score']:.2%}"
    )


with col5:

    st.metric(
        "ROC-AUC",
        f"{rain_metrics['roc_auc']:.2%}"
    )


# ============================================================
# RAIN CLASSIFICATION CHART
# ============================================================

classification_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]


classification_values = [

    rain_metrics["accuracy"],

    rain_metrics["precision"],

    rain_metrics["recall"],

    rain_metrics["f1_score"],

    rain_metrics["roc_auc"]

]


fig_classification = go.Figure(

    go.Bar(
        x=classification_names,
        y=classification_values,
        text=[
            f"{value:.1%}"
            for value in classification_values
        ],
        textposition="outside"
    )

)


fig_classification.update_layout(
    title="Rainfall Occurrence Classification Performance",
    yaxis_title="Score",
    xaxis_title="Metric",
    yaxis={
        "range": [0, 1.05]
    },
    height=400
)


st.plotly_chart(
    fig_classification,
    use_container_width=True
)


# ============================================================
# SECTION 7 — PROJECT SUMMARY
# ============================================================

st.divider()

st.header("📌 Project Summary")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Input Features",
        "476"
    )


with col2:

    st.metric(
        "ML Models",
        "7"
    )


with col3:

    st.metric(
        "Regression Models",
        "6"
    )


with col4:

    st.metric(
        "Classification Models",
        "1"
    )

# ============================================================
# SECTION 8 — ABOUT THE PROJECT
# ============================================================

st.divider()

st.header("ℹ️ About the Project")

about_col1, about_col2 = st.columns(2)


# ============================================================
# PROJECT DESCRIPTION
# ============================================================

with about_col1:

    st.markdown(
        """
        ### 🌦️ Mumbai Climate ML Prediction

        This project uses machine learning to predict
        next-hour weather and air-quality conditions for Mumbai.

        The system combines historical weather observations,
        air-quality measurements and engineered temporal,
        lag, rolling-window, rainfall and atmospheric features.

        The goal is to provide a simple and useful view of
        current conditions and short-term ML predictions.
        """
    )


# ============================================================
# ML PIPELINE
# ============================================================

with about_col2:

    st.markdown(
        """
        ### ⚙️ ML Pipeline

        **Data:** Mumbai weather + air quality

        **Features:** 476 engineered features

        **Models:** 7 Extra Trees ML models

        **Regression:** Temperature, rainfall, wind speed,
        pressure, humidity and PM2.5

        **Classification:** Rain occurrence

        **Forecast:** Next-hour conditions
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Mumbai Climate ML Prediction • "
    "Extra Trees ML • 476 Engineered Features • "
    "Next-Hour Forecasting"
)

st.caption(
    "Built as a machine-learning project for Mumbai weather "
    "and air-quality prediction."
)