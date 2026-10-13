# EV_Charging_demand_model
This project is about analyzing EV charging station usage patterns and to predict charging demand based on factors such as charger type, location, charging company, time of charging, and historical usage behavior using regression and classification models.
## Project Overview
An end-to-end data science web application that analyzes electric vehicle charging behavior and provides predictive intelligence for infrastructure planning.

## Project Structure
The project is organized into a clean, production-ready directory structure:
* **`ev_app.py`**: The main interactive Streamlit web application script featuring a multi-tab interface, custom leaf-green styling and integrated models.
* **`data/cleaned/FEChargingRecords.csv`**: The cleaned and feature-engineered dataset containing charging session records, timestamps, durations, and location data.
* **`notebooks/`**: Contains the data cleaning, eda, feature engineering and serialized machine learning models (`.pkl` files):
  * `regressor_ev_demand_model.pkl`: Predicts exact session energy demand in kWh.
  * `classifier_ev_demand_model.pkl`: Classifies sessions into High or Normal demand tiers.
  * `timeseries_holtwinters_model.pkl`: Forecasts aggregate network demand over future horizons.
* **`requirements.txt`**: Lists all exact library dependencies required for local execution and cloud deployment.

[View Live EV Charging Analytics & Demand Platform App](https://evchargingdemandmodel-ygl9pjkscvreuxxf5okfhh.streamlit.app/)

The application is structured into four interactive intelligence tabs:
1. **EDA Insights:** Comprehensive visualization of spatial and temporal usage patterns, peak commuting windows, location-based energy intensity, monthly trends, and hardware preferences.
2. **Regression (Demand kWh):** Individual session energy demand prediction based on duration, location target-encoded values, charger type, company, and start times.
3. **Classification (High/Normal):** Real-time tier classification identifying whether an upcoming session will experience High or Normal demand intensity.
4. **Time-Series Forecast:** Multi-day aggregate network demand forecasting powered by a seasonal Holt-Winters model.
