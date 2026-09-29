🌦️ Mumbai Climate ML Prediction

A machine-learning based web application that uses weather and air-quality data for Mumbai to generate next-hour climate predictions through an interactive Streamlit dashboard.

🚀 Live Demo

🌐 Live Dashboard: https://mumbai-climate-prediction-8y5ouw5reoy3ueshg4tsw.streamlit.app/

📂 GitHub Repository: https://github.com/Anupam-codes7/Mumbai-Climate-Prediction

🤗 ML Model Repository: https://huggingface.co/ezdevz/mumbai-climate-ml-models

The dashboard is publicly accessible for demonstration. The source-code repository can remain private while the deployed dashboard is shared publicly.

📌 Project Overview

Mumbai Climate ML Prediction is an end-to-end machine learning project designed to predict important weather and air-quality parameters for the next hour.

The system collects weather and air-quality observations, performs feature engineering, creates 476 model input features, and uses seven trained Extra Trees models to generate predictions.

The project demonstrates the complete ML workflow:

Data Collection → Feature Engineering → Model Training → Prediction → Visualization → Deployment

🎯 Objectives

Collect weather and air-quality data for Mumbai.

Engineer meaningful temporal and historical features.

Predict next-hour weather and PM2.5 values.

Predict the probability of rainfall occurrence.

Build an interactive dashboard for visualization.

Deploy the complete ML application online.

Separate large trained model files from the GitHub source-code repository.

🏗️ System Architecture

                  Open-Meteo APIs
                       │
              ┌────────┴────────┐
              │                 │
        Weather Data       Air Quality Data
              │                 │
              └────────┬────────┘
                       ↓
              Data Processing
                       ↓
              Feature Engineering
                       ↓
              476 Input Features
                       ↓
             ┌───────────────────┐
             │  Extra Trees ML   │
             │      Models       │
             └───────────────────┘
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
      Weather Predictions   Rain Classification
             │                   │
             └─────────┬─────────┘
                       ↓
              Streamlit Dashboard
                       │
                       ↓
                Live Web Demo

🤖 Machine Learning Pipeline

The project uses 7 Extra Trees models:

Regression Models

Temperature

Rainfall

Wind Speed

Pressure

Humidity

PM2.5

Classification Model

Rain Occurrence

The rain-occurrence model uses a binary target derived from rainfall:

Rain > 0.1 mm  → Rain Occurrence = 1
Rain ≤ 0.1 mm  → Rain Occurrence = 0

Why Extra Trees?

Extra Trees was selected because it can handle a large number of features and capture complex non-linear relationships in weather data. Combining many randomized decision trees also makes the model more robust than relying on a single decision tree.

📊 Feature Engineering

The final model input contains 476 engineered features.

Features include:

Raw weather variables

Air-quality variables

Hour/day/month information

Cyclical time features

Wind direction transformations

Mumbai seasonal indicators

Lag features

Historical rainfall features

Rolling rainfall statistics

Lag Features

Historical values are used at multiple time intervals, including:

1 hour
3 hours
6 hours
12 hours
24 hours
48 hours
72 hours
168 hours

This allows the models to use recent and longer-term patterns when generating the next-hour prediction.

Cyclical Features

Time-based variables are transformed using sine/cosine representations so that cyclic relationships such as:

23:00 → 00:00
December → January

are represented more naturally.

🎯 Prediction Targets

The system predicts:

Target

Type

Temperature

Regression

Rainfall

Regression

Rain Occurrence

Classification

Wind Speed

Regression

Pressure

Regression

Humidity

Regression

PM2.5

Regression

The prediction target is generated using the next observation:

Current time (t) → Predict weather at t + 1 hour

📈 Model Evaluation

The rainfall-occurrence classification model was evaluated using:

Metric

Result

Accuracy

84.17%

Precision

78.50%

Recall

76.83%

F1 Score

77.66%

ROC-AUC

92.76%

These values represent the evaluated classification model performance. Dashboard comparisons with the latest available observation should not be interpreted as formal forecast-accuracy metrics because the displayed observation may not always correspond exactly to the future prediction horizon.

🌐 Interactive Dashboard

The Streamlit dashboard provides:

Current Mumbai Weather

Temperature

Humidity

Wind Speed

Pressure

Rainfall

PM2.5

Next-Hour ML Prediction

Predicted temperature

Predicted rainfall

Rain occurrence

Rain probability

Predicted wind speed

Predicted pressure

Predicted humidity

Predicted PM2.5

Dashboard Features

🔄 Live prediction refresh

📊 Interactive Plotly visualizations

🌧️ Rain probability gauge

📈 Prediction analysis

📏 Absolute difference analysis

🧪 Model performance metrics

🌦️ Current weather and air-quality information

📱 Browser-based deployment

☁️ Deployment Architecture

Large trained model files are not stored in the GitHub repository because the seven .joblib files together are approximately 1 GB.

Instead:

GitHub
│
├── Source Code
├── Streamlit Dashboard
├── README
├── Requirements
└── Project Files
        │
        ↓
Streamlit Community Cloud
        │
        ↓
Hugging Face Hub
        │
        └── 7 trained Extra Trees models

The application uses huggingface_hub to download a model automatically when it is not available locally.

Model Repository

https://huggingface.co/ezdevz/mumbai-climate-ml-models

🛠️ Tech Stack

Programming

Python

Data & ML

Pandas

NumPy

Scikit-learn

Joblib

Data Sources

Open-Meteo Weather API

Open-Meteo Air Quality API

Visualization

Plotly

Streamlit

Deployment & Version Control

Git

GitHub

Hugging Face Hub

Streamlit Community Cloud

📁 Project Structure

Mumbai-Climate-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── feature_pipeline.txt
│
├── data/
│
├── images/
│
├── notebooks/
│
├── models/
│   ├── actual_weather.json
│   ├── comparison.json
│   ├── feature_config.joblib
│   ├── model_metrics.json
│   └── prediction.json
│
└── src/
    ├── weather_api.py
    ├── feature_engineering.py
    ├── live_prediction.py
    ├── create_comparison.py
    └── create_metrics.py

The large trained .joblib Extra Trees models are hosted in the Hugging Face model repository rather than GitHub.

💻 Run Locally

1. Clone the repository

git clone https://github.com/Anupam-codes7/Mumbai-Climate-Prediction.git
cd Mumbai-Climate-Prediction

2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run live prediction

python src/live_prediction.py

The script fetches live data, generates the engineered features, loads the ML models, and saves the prediction output.

5. Start the dashboard

streamlit run app.py

The application will open in your browser.

🔄 Live Prediction Workflow

When Refresh Live Prediction is selected:

1. Fetch latest Mumbai weather data
        ↓
2. Fetch latest air-quality data
        ↓
3. Combine and process the data
        ↓
4. Generate the required 476 features
        ↓
5. Load the 7 Extra Trees models
        ↓
6. Generate next-hour predictions
        ↓
7. Save prediction output
        ↓
8. Update the Streamlit dashboard

⚠️ Limitations

The system is designed for short-term next-hour prediction.

Weather conditions can change rapidly and may contain uncertainty.

Model performance depends on the quality and availability of historical data.

Live API availability can affect the refresh process.

Dashboard observation-vs-prediction comparisons are intended for visualization and demonstration, not as a substitute for a formal forecast evaluation protocol.

🔮 Future Improvements

Add longer forecasting horizons.

Compare Extra Trees with XGBoost, Random Forest and other ML approaches.

Add automated model retraining.

Add more historical weather stations and spatial features.

Improve rainfall forecasting using dedicated precipitation models.

Add model explainability using feature importance/SHAP.

Add automated monitoring of prediction drift.

Add scheduled data collection and retraining pipelines.

🎓 Academic Relevance

This project demonstrates practical implementation of:

Data collection

Data preprocessing

Feature engineering

Time-series feature creation

Regression

Binary classification

Model evaluation

API integration

Data visualization

Web application development

Cloud deployment

ML model hosting

It provides an end-to-end example of taking an ML model from data processing and training through to a publicly accessible application.

👨‍💻 Author

Anupam Das

B.Tech – Computer Science Engineering
Specialization: Data Science

GitHub: https://github.com/Anupam-codes7

📜 License

This project is developed for academic, learning and portfolio purposes.
