from datetime import datetime, timedelta
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title='EV Charging Analytics & Demand Platform',
    page_icon='⚡',
    layout='wide',
)

# --- Custom Professional Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Body Font & Color */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1E293B;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
    }

    /* Main Background */
    .main { 
        background-color: #F8FAFC; 
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #1B4332 0%, #0F2F1E 100%);
        padding: 3rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px -5px rgba(27, 67, 50, 0.2);
    }
    .header-container h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        letter-spacing: -0.8px;
    }
    .header-container p {
        color: #D8F3DC !important;
        font-size: 1.15rem;
        font-weight: 400;
        letter-spacing: -0.2px;
    }

    /* Section Headings - Exact Header Background Shade (#1B4332) */
    h1, h2, h3 { 
        color: #1B4332 !important; 
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* Paragraph & Subtext styling */
    p, label, .stMarkdown {
        color: #334155;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    /* Universal Buttons Styling */
   /* Universal Buttons Styling - Targets inner text and spans explicitly */
    .stButton>button, div.stButton>button, .st-key-forecast_btn button { 
        background-color: #2D6A4F !important; 
        border-radius: 8px !important; 
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 12px rgba(45, 106, 79, 0.25);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button *, div.stButton>button *, .st-key-forecast_btn button * { 
        color: #FFFFFF !important; 
        font-weight: 600 !important;
    }
    .stButton>button:hover, div.stButton>button:hover, .st-key-forecast_btn button:hover { 
        background-color: #1B4332 !important; 
        box-shadow: 0 6px 15px rgba(27, 67, 50, 0.35);
        transform: translateY(-1px);
    }

    /* Active Tab Highlight Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E2E8F0;
        border-radius: 8px 8px 0px 0px;
        color: #475569;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2D6A4F !important;
    }
    .stTabs [aria-selected="true"] p, 
    .stTabs [aria-selected="true"] span, 
    .stTabs [aria-selected="true"] div,
    .stTabs [aria-selected="true"] button,
    .stTabs [aria-selected="true"] * {
        color: #FFFFFF !important;
    }

    /* EDA Insights Card */
    .insights-card {
        background-color: #FFFFFF;
        padding: 1.75rem;
        border-radius: 12px;
        border-left: 6px solid #2D6A4F;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        color: #1E293B;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Centered Header Banner ---
st.markdown(
    """
    <div class="header-container">
        <h1>EV Charging Demand Intelligence Platform</h1>
        <p>End-to-end data science decision support tool for electric vehicle charging infrastructure.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# --- Load Data & Models Caching ---
@st.cache_resource
def load_artifacts():
  # Get the directory where ev_app.py is currently located
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Build robust paths that work on both Windows and Linux (Streamlit Cloud)
  data_path = os.path.join(current_dir, 'data', 'cleaned', 'FEChargingRecords.csv')
  reg_model_path = os.path.join(
      current_dir, 'notebooks', 'regressor_ev_demand_model.pkl'
  )
  clf_model_path = os.path.join(
      current_dir, 'notebooks', 'classifier_ev_demand_model.pkl'
  )
  ts_model_path = os.path.join(
      current_dir, 'notebooks', 'timeseries_holtwinters_model.pkl'
  )

  df = pd.read_csv(data_path)
  df.columns = df.columns.str.strip()

  if 'StartDatetime' in df.columns:
    df['StartDatetime'] = pd.to_datetime(df['StartDatetime'], dayfirst=True)

  # Load Regressor Model
  try:
    with open(reg_model_path, 'rb') as f:
      reg_model = pickle.load(f)
  except:
    reg_model = None

  # Load Classifier Model
  try:
    with open(clf_model_path, 'rb') as f:
      clf_model = pickle.load(f)
  except:
    clf_model = None

  # Load Time-Series Model
  try:
    with open(ts_model_path, 'rb') as f:
      ts_model = pickle.load(f)
  except:
    ts_model = None

  return df, reg_model, clf_model, ts_model
df, reg_model, clf_model, ts_model = load_artifacts()

if 'reg_history' not in st.session_state:
  st.session_state.reg_history = []
if 'clf_history' not in st.session_state:
  st.session_state.clf_history = []

location_encoding_map = {}
loc_col = next(
    (
        c
        for c in df.columns
        if c.lower() in ['location', 'location_type', 'locationname']
    ),
    None,
)
enc_col = next(
    (
        c
        for c in df.columns
        if 'location' in c.lower() and 'enc' in c.lower()
    ),
    None,
)

if loc_col and enc_col:
  location_encoding_map = df.groupby(loc_col)[enc_col].first().to_dict()
  available_locations = sorted(list(location_encoding_map.keys()))
elif loc_col:
  available_locations = sorted(df[loc_col].dropna().unique().tolist())
  location_encoding_map = {loc: 50.0 for loc in available_locations}
else:
  available_locations = ['Default Location']
  location_encoding_map = {'Default Location': 50.0}

# --- Create 4 Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    'EDA Insights',
    'Regression (Demand kWh)',
    'Classification (High/Normal)',
    'Time-Series Forecast',
])


# ==========================================
# TAB 1: EDA INSIGHTS
# ==========================================
with tab1:
  st.header('Exploratory Data Analysis Insights')
  st.markdown(
      'Comprehensive exploratory analysis uncovering spatial, temporal, and'
      ' infrastructure charging preferences.'
  )

  col1, col2 = st.columns(2)

  with col1:
    st.subheader('Peak Charging Hours Distribution')
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'Start_Hour' in df.columns:
      sns.histplot(
          df['Start_Hour'], bins=24, kde=True, color='#2D6A4F', ax=ax
      )
    ax.set_title('Charging Sessions Frequency by Hour of Day')
    ax.set_xlabel('Start Hour (0-23)')
    ax.set_ylabel('Session Count')
    st.pyplot(fig)

  with col2:
    st.subheader('Demand by Day of the Week')
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'Start_DayOfWeek' in df.columns:
      sns.barplot(
          x='Start_DayOfWeek',
          y='Demand',
          data=df,
          estimator=np.sum,
          palette='Greens_d',
          ax=ax,
      )
      ax.set_title('Total Energy Demand (kWh) across Days')
      ax.set_xlabel('Day of Week (0=Mon, 6=Sun)')
      ax.set_ylabel('Total Demand (kWh)')
    st.pyplot(fig)

  st.markdown('---')

  col3, col4 = st.columns(2)

  with col3:
    st.subheader('Average Energy Demand by Location Type')
    fig, ax = plt.subplots(figsize=(8, 4))
    if loc_col and 'Demand' in df.columns:
      loc_demand = (
          df.groupby(loc_col)['Demand']
          .mean()
          .sort_values(ascending=False)
          .reset_index()
      )
      sns.barplot(
          x='Demand', y=loc_col, data=loc_demand, palette='crest', ax=ax
      )
      ax.set_title('Average Demand (kWh) per Session by Location')
      ax.set_xlabel('Average Demand (kWh)')
      ax.set_ylabel('Location Type')
    st.pyplot(fig)

  with col4:
    st.subheader('Average Demand by Time of Day (Hour)')
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'Start_Hour' in df.columns:
      hourly_avg = (
          df.groupby('Start_Hour')['Demand'].mean().reset_index()
      )
      sns.lineplot(
          x='Start_Hour',
          y='Demand',
          data=hourly_avg,
          color='#2D6A4F',
          marker='o',
          linewidth=2.5,
          ax=ax,
      )
      ax.set_title('Mean Energy Consumption per Session across Hours')
      ax.set_xlabel('Start Hour (0-23)')
      ax.set_ylabel('Average Demand (kWh)')
      ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

  st.markdown('---')

  col5, col6 = st.columns(2)

  with col5:
    st.subheader('Month-wise Total Demand')
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'Start_Month' in df.columns:
      monthly_demand = df.groupby('Start_Month')['Demand'].sum().reset_index()
      sns.barplot(
          x='Start_Month',
          y='Demand',
          data=monthly_demand,
          palette='viridis',
          ax=ax,
      )
      ax.set_title('Total Energy Demand (kWh) by Month')
      ax.set_xlabel('Month')
      ax.set_ylabel('Total Demand (kWh)')
    st.pyplot(fig)

  with col6:
    st.subheader('Preferable Charger Type & Company')
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'ChargerType' in df.columns and 'ChargerCompany' in df.columns:
      charger_pref = (
          df.groupby(['ChargerType', 'ChargerCompany'])['Demand']
          .count()
          .reset_index(name='Session_Count')
      )
      sns.barplot(
          x='ChargerType',
          y='Session_Count',
          hue='ChargerCompany',
          data=charger_pref,
          palette='Set2',
          ax=ax,
      )
      ax.set_title('Session Distribution by Charger Type & Company')
      ax.set_xlabel('Charger Type')
      ax.set_ylabel('Total Session Count')
      ax.legend(
          title='Charger Company', bbox_to_anchor=(1.05, 1), loc='upper left'
      )
    st.pyplot(fig)

  st.divider()
  st.subheader('Key Takeaways & Infrastructure Insights')
  st.markdown(
      """
        <div class="insights-card">
            <ul>
                <li><strong>Temporal Load Dynamics (Peak Hours):</strong> The distribution histogram reveals a severe bi-modal concentration of sessions. Charging events peak sharply during morning commuter windows (7:00 AM – 9:00 AM) and evening return windows (5:00 PM – 7:00 PM).</li>
                <br>
                <li><strong>Spatial Utilization vs. Energy Intensity:</strong> High-frequency session counts concentrate predominantly in public spaces and residential apartments due to frequent top-ups. However, <em>accommodations and bus garages</em> yield significantly higher average energy demand (kWh) per individual session, pointing to prolonged dwell times or heavy-duty fleet recharging.</li>
                <br>
                <li><strong>Weekly Load Profiles:</strong> Daily aggregated bar charts confirm that mid-week periods (Tuesday through Thursday) record the highest overall cumulative energy consumption, tapering noticeably over the weekend as commercial commuter traffic subsides.</li>
                <br>
                <li><strong>Hourly Consumption Efficiency:</strong> Line analysis of mean energy consumption per session across hours indicates that overnight and mid-day sessions draw distinct average loads, helping distinguish between casual topping off and full-cycle charging.</li>
                <br>
                <li><strong>Seasonal Demand Fluctuations:</strong> Month-wise analysis indicates variance in total network demand, driven by seasonal travel and temperature changes impacting EV battery efficiency.</li>
                <br>
                <li><strong>Hardware & Provider Preferences:</strong> Cross-tabulation of charger types and operating companies shows dominant market shares concentrated in specific hardware configurations, offering a blueprint for future infrastructure scaling and hardware investments.</li>
            </ul>
        </div>
        """,
      unsafe_allow_html=True,
  )


# ==========================================
# TAB 2: REGRESSION MODEL
# ==========================================
with tab2:
  st.header('Individual Session Demand Prediction (Regression)')
  st.write(
      'Input session characteristics below to predict exact energy consumption'
      ' in kWh.'
  )

  with st.form('reg_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
      charger_id = st.number_input('Charger ID', min_value=1, value=1)
      charger_company = st.selectbox('Charger Company', [0, 1])
      selected_location = st.selectbox('Location Name', available_locations)
    with c2:
      charger_type = st.selectbox('Charger Type', [0,1])
      duration = st.number_input(
          'Duration (Minutes)', min_value=1, max_value=720, value=60
      )
      start_hour = st.slider('Start Hour', 0, 23, 12)
    with c3:
      day_of_week = st.selectbox(
          'Day of Week (0=Mon, 6=Sun)', range(7), index=2
      )
      month = st.selectbox('Start Month', range(1, 13), index=9)
      is_weekend = 1 if day_of_week >= 5 else 0

    submit_reg = st.form_submit_button('Predict Energy Demand (kWh)')

  if submit_reg:
    if reg_model is not None:
      location_enc = location_encoding_map.get(selected_location, 50.0)

      input_vector = pd.DataFrame([{
          'ChargerID': charger_id,
          'ChargerCompany': charger_company,
          'Location_TargetEncoded': location_enc,
          'ChargerType': charger_type,
          'Duration': duration,
          'Start_Hour': start_hour,
          'Start_DayOfWeek': day_of_week,
          'Start_Month': month,
          'Is_Weekend': is_weekend,
          'Is_Peak_Hour': 1
          if (7 <= start_hour <= 9) or (17 <= start_hour <= 19)
          else 0,
      }])
      pred_kwh = reg_model.predict(input_vector)[0]
      st.success(
          f'Predicted Session Energy Demand: **{pred_kwh:.2f} kWh**'
      )

      st.session_state.reg_history.insert(
          0,
          {
              'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'Location': selected_location,
              'Duration': duration,
              'Hour': start_hour,
              'Predicted kWh': round(pred_kwh, 2),
          },
      )
    else:
      st.error('Regression model `.pkl` file not found.')

  st.subheader('Last 5 Regressor Predictions')
  if st.session_state.reg_history:
    st.dataframe(
        pd.DataFrame(st.session_state.reg_history[:5]),
        use_container_width=True,
    )
  else:
    st.write('No predictions made yet in this session.')


# ==========================================
# TAB 3: CLASSIFICATION MODEL
# ==========================================
with tab3:
  st.header('Demand Intensity Classification (High vs Normal)')
  st.write(
      'Determine whether an upcoming charging session will experience High or'
      ' Normal demand levels.'
  )

  with st.form('clf_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
      c_id = st.number_input('Charger ID', min_value=1, value=1)
      c_comp = st.selectbox('Charger Company', [0, 1])
      c_selected_loc = st.selectbox(
          'Location Name', available_locations
      )
    with c2:
      c_type = st.selectbox('Charger Type', [0,1])
      c_dur = st.number_input(
          'Duration (Minutes)', min_value=1, max_value=720, value=90
      )
      c_hour = st.slider('Start Hour', 0, 23, 18)
    with c3:
      c_dow = st.selectbox(
          'Day of Week (0=Mon, 6=Sun)', range(7), index=4
      )
      c_mon = st.selectbox('Start Month', range(1, 13), index=9)
      c_wkend = 1 if c_dow >= 5 else 0

    submit_clf = st.form_submit_button('Classify Demand Tier')

  if submit_clf:
    if clf_model is not None:
      c_loc_enc = location_encoding_map.get(c_selected_loc, 50.0)

      clf_vector = pd.DataFrame([{
          'ChargerID': c_id,
          'ChargerCompany': c_comp,
          'Location_TargetEncoded': c_loc_enc,
          'ChargerType': c_type,
          'Duration': c_dur,
          'Start_Hour': c_hour,
          'Start_DayOfWeek': c_dow,
          'Start_Month': c_mon,
          'Is_Weekend': c_wkend,
          'Is_Peak_Hour': 1
          if (7 <= c_hour <= 9) or (17 <= c_hour <= 19)
          else 0,
      }])
      pred_class = clf_model.predict(clf_vector)[0]
      pred_prob = (
          clf_model.predict_proba(clf_vector)[0][1]
          if hasattr(clf_model, 'predict_proba')
          else 0.0
      )

      tier_label = (
          'High Demand Tier' if pred_class == 1 else 'Normal Demand Tier'
      )
      st.success(
          f'Classification Result: **{tier_label}** (Probability of High'
          f' Demand: {pred_prob:.2%})'
      )

      st.session_state.clf_history.insert(
          0,
          {
              'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'Location': c_selected_loc,
              'Duration': c_dur,
              'Hour': c_hour,
              'Predicted Class': 'High' if pred_class == 1 else 'Normal',
              'High Demand Prob': round(pred_prob, 4),
          },
      )
    else:
      st.error('Classification model `.pkl` file not found.')

  st.subheader('Last 5 Classifier Predictions')
  if st.session_state.clf_history:
    st.dataframe(
        pd.DataFrame(st.session_state.clf_history[:5]),
        use_container_width=True,
    )
  else:
    st.write('No classification queries made yet in this session.')


# ==========================================
# TAB 4: TIME-SERIES FORECAST MODEL
# ==========================================
with tab4:
  st.header('Aggregate Network Demand Forecasting (Time-Series)')
  st.write(
      'Generate dynamic multi-day network demand forecasts using the saved'
      ' Holt-Winters seasonal model.'
  )

  forecast_steps = st.slider(
      'Select Forecast Horizon (Days)', min_value=1, max_value=100, value=7
  )

  # Using Markdown HTML formatting to ensure button text is natively white
  st.markdown(
      '<style>div.stButton > button:first-child {{ color: #FFFFFF'
      ' !important; }}</style>',
      unsafe_allow_html=True,
  )

  if st.button('Generate Future Forecast', key='forecast_btn'):
    if ts_model is not None:
      forecast_values = ts_model.forecast(steps=forecast_steps)

      last_data_date = pd.to_datetime(df['StartDatetime']).max().date()
      forecast_start_date = last_data_date + timedelta(days=1)

      forecast_df = pd.DataFrame({
          'Date': pd.date_range(
              start=forecast_start_date, periods=forecast_steps, freq='D'
          ),
          'Forecasted Total Demand (kWh)': forecast_values.values,
      })
      forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')

      st.success(
          f'Successfully forecasted next {forecast_steps} days of network'
          ' demand!'
      )
      st.dataframe(forecast_df, use_container_width=True)

      fig, ax = plt.subplots(figsize=(10, 4))
      ax.plot(
          forecast_df['Date'],
          forecast_df['Forecasted Total Demand (kWh)'],
          marker='o',
          color='#2D6A4F',
          linewidth=2,
      )
      ax.set_title(
          f'Holt-Winters Seasonal Forecast for Next {forecast_steps} Days'
      )
      ax.set_xlabel('Date')
      ax.set_ylabel('Total Demand (kWh)')
      plt.xticks(rotation=45)
      st.pyplot(fig)
    else:
      st.error('Time-series model `.pkl` file not found.')