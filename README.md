# 🌦️ Mumbai Climate ML Prediction

### Live Weather Intelligence & Next-Hour Machine Learning Prediction

Mumbai Climate ML Prediction is a machine-learning based weather and air-quality forecasting system designed specifically for Mumbai.

The project combines historical weather observations, live weather and air-quality data, temporal patterns, rainfall statistics, cyclical features, seasonal indicators and multi-scale lag features to generate **next-hour predictions**.

The system uses **7 Extra Trees ML models** trained using **476 engineered features**.

---

## 📌 Project Overview

Weather conditions in Mumbai change rapidly due to seasonal patterns, rainfall, humidity, wind and atmospheric conditions.

This project aims to provide a simple ML-based system that can:

- 🌍 Fetch Mumbai weather and air-quality data
- ⚙️ Generate engineered climate features
- 🤖 Predict next-hour weather conditions
- 🌧️ Estimate rainfall and rain occurrence probability
- 🌬️ Predict wind speed and atmospheric pressure
- 💧 Predict humidity
- 🫁 Predict PM2.5 concentration
- 📊 Compare predictions with the latest available observation
- 📈 Display model performance through an interactive Streamlit dashboard

---

# 🚀 Key Features

## 🌍 Live Mumbai Weather

The application retrieves Mumbai weather and air-quality information and displays:

- Temperature
- Humidity
- Wind speed
- Atmospheric pressure
- Rainfall
- PM2.5
- Other atmospheric and air-quality variables

---

## 🤖 Next-Hour ML Prediction

The system generates next-hour predictions for:

| Target | Output |
|---|---|
| Temperature | °C |
| Rainfall | mm |
| Rain occurrence | Yes / No |
| Rain probability | % |
| Wind speed | km/h |
| Pressure | hPa |
| Humidity | % |
| PM2.5 | µg/m³ |

---

# 🧠 Machine Learning Pipeline

The project follows this pipeline:

```text
Weather + Air Quality Data
            ↓
       Data Validation
            ↓
     Feature Engineering
            ↓
     476 ML Features
            ↓
   Train / Validation / Test
            ↓
     Extra Trees Models
            ↓
     Next-Hour Prediction
            ↓
 Prediction vs Observation
            ↓
     Streamlit Dashboard

     ⚙️ Feature Engineering

A major part of the project is the creation of meaningful features from the raw weather and air-quality data.

The final ML configuration contains:

476 input features

⏰ Temporal Features

The system extracts:

Year
Month
Day
Day of year
Day of week
Hour
🔄 Cyclical Time Features

Cyclical transformations are used to represent repeating time patterns:

hour_sin
hour_cos
day_of_year_sin
day_of_year_cos
day_of_week_sin
day_of_week_cos

This allows the ML models to understand that, for example, 23:00 and 00:00 are close in time.

🌬️ Wind Direction Features

Wind direction is represented using:

wind_direction_sin
wind_direction_cos

This avoids treating 0° and 360° as completely different directions.

🌦️ Mumbai Seasonal Features

The project includes Mumbai-specific seasonal indicators:

Monsoon
Pre-monsoon
Post-monsoon
Winter
⏮️ Multi-Scale Lag Features

Historical values are used to capture short-term and long-term climate patterns.

Lag periods include:

1 hour
3 hours
6 hours
12 hours
24 hours
48 hours
72 hours
168 hours

These features are generated for variables such as:

Temperature
Rainfall
Wind speed
Pressure
Relative humidity
PM2.5
PM10
Carbon monoxide
Nitrogen dioxide
Sulphur dioxide
Ozone
🌧️ Rainfall-Specific Features

The system also creates rainfall history features including:

24-hour rainfall total
72-hour rainfall total
168-hour rainfall total
Rainy hours in previous 24 hours
Rainy hours in previous 72 hours
Rainy hours in previous 168 hours
Maximum rainfall
Previous rainfall indicators

These features help the model understand recent rainfall conditions.

🤖 ML Models

The project uses Extra Trees models for the prediction tasks.

Seven separate models are used:

1. Temperature
2. Rainfall
3. Rain occurrence
4. Wind speed
5. Pressure
6. Humidity
7. PM2.5
Regression Models

Regression models predict continuous values for:

Temperature
Rainfall
Wind speed
Pressure
Humidity
PM2.5
Classification Model

A separate classification model predicts:

Whether rainfall is expected during the next hour.

The project also calculates the probability of rainfall using the classifier's probability output.

🌧️ Rainfall Classification

Rain occurrence is formulated as a binary classification problem.

Rain > 0.1 mm
       ↓
Rain = 1


Rain ≤ 0.1 mm
       ↓
Rain = 0

The dashboard reports:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
📊 Model Evaluation

The project evaluates regression models using metrics such as:

R²
MAE
RMSE

The rainfall occurrence model is evaluated using:

Accuracy
Precision
Recall
F1 Score
ROC-AUC

The Streamlit dashboard presents these metrics visually.

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Dashboard Sections
🌍 Current Mumbai Weather

Displays the latest available weather and air-quality observations.

📝 Current Conditions

Provides a simple interpretation of:

Temperature
Humidity
Wind
PM2.5
Current rainfall
🤖 ML Next-Hour Prediction

Displays predictions generated by the seven Extra Trees models.

🌧️ Rain Probability

A visual gauge displays the predicted probability of rainfall.

📊 Prediction Analysis

The dashboard compares:

ML Prediction vs Latest Observation

Each variable is visualized separately so that variables with different units and scales are not incorrectly combined into a single axis.

📉 Absolute Difference

Displays the absolute difference between the ML prediction and latest observation.

🏆 Model Performance

Displays regression and classification performance metrics.

🛠️ Technology Stack
Technology	Purpose
Python	Core programming language
Pandas	Data processing
NumPy	Numerical operations
Scikit-learn	Machine learning
Extra Trees	ML prediction models
Joblib	Model/configuration serialization
Requests	API requests
Plotly	Interactive visualizations
Streamlit	Dashboard
Jupyter Notebook	Data exploration and experimentation
Git	Version control
GitHub	Project repository
📁 Project Structure
Mumbai-Climate-Prediction/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   ├── mumbai_weather_2015_2025.csv
│   │   └── mumbai_weather_hourly_2022_2025.csv
│   │
│   └── processed/
│       ├── mumbai_climate_ml.csv
│       └── mumbai_weather_air_quality_hourly.csv
│
├── models/
│   ├── actual_weather.json
│   ├── comparison.json
│   ├── feature_config.joblib
│   ├── model_metrics.json
│   └── prediction.json
│
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   └── 01_Mumbai_Data_Collection.ipynb
│
├── src/
│   ├── create_comparison.py
│   ├── create_metrics.py
│   ├── feature_engineering.py
│   ├── live_prediction.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── weather_api.py
│
├── feature_pipeline.txt
├── .gitignore
└── README.md
🔄 Data Flow
             Mumbai Weather Data
                      +
             Mumbai Air Quality
                      ↓
              Data Collection
                      ↓
             Data Preprocessing
                      ↓
             Feature Engineering
                      ↓
              476 Features
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
 Regression Models          Rain Classification
        ↓                           ↓
 Temperature                 Rain / No Rain
 Rainfall                    Probability
 Wind Speed
 Pressure
 Humidity
 PM2.5
        └─────────────┬─────────────┘
                      ↓
             Streamlit Dashboard
