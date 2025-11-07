import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Title
st.title("🌫️ Air Quality Index (AQI) Prediction Dashboard")
st.markdown("### Karachi Air Quality Monitoring & Forecasting")

# Load data
@st.cache_data
def load_data():
    try:
        historical = pd.read_csv('data/2years_features.csv', parse_dates=['datetime_utc'])
        predictions = pd.read_csv('data/future_aqi_predictions.csv', parse_dates=['datetime_utc'])
        training_results = pd.read_csv('data/training_results.csv')
        comparison = pd.read_csv('data/future_prediction_comparison.csv')
        return historical, predictions, training_results, comparison
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        return None, None, None, None

historical, predictions, training_results, comparison = load_data()
# Ensure datetime type
historical['datetime_utc'] = pd.to_datetime(historical['datetime_utc'], errors='coerce')

if historical is None:
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Configuration")
view_mode = st.sidebar.radio("Select View", ["Historical Data", "Future Predictions", "Model Comparison"])

# Define AQI categories
def get_aqi_color(aqi):
    if aqi <= 50:
        return '#00E400', 'Good'
    elif aqi <= 100:
        return '#FFFF00', 'Moderate'
    elif aqi <= 150:
        return '#FF7E00', 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return '#FF0000', 'Unhealthy'
    elif aqi <= 300:
        return '#8F3F97', 'Very Unhealthy'
    else:
        return '#7E0023', 'Hazardous'

# ============================================================================
# HISTORICAL DATA VIEW
# ============================================================================
if view_mode == "Historical Data":
    st.header("📊 Historical Air Quality Data")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_aqi = historical['us_aqi'].iloc[-1]
        color, category = get_aqi_color(current_aqi)
        st.metric("Current AQI", f"{current_aqi:.0f}", category)
    with col2:
        avg_aqi = historical['us_aqi'].mean()
        st.metric("Average AQI (2 years)", f"{avg_aqi:.1f}")
    with col3:
        max_aqi = historical['us_aqi'].max()
        st.metric("Maximum AQI", f"{max_aqi:.0f}")
    with col4:
        st.metric("Total Records", f"{len(historical):,}")
    
    # Time series plot
    st.subheader("🕒 AQI Over Time")
    
    # Date range selector
    date_range = st.date_input(
        "Select Date Range",
        value=(historical['datetime_utc'].min(), historical['datetime_utc'].max()),
        min_value=historical['datetime_utc'].min().date(),
        max_value=historical['datetime_utc'].max().date()
    )
    
    if len(date_range) == 2:
        mask = (historical['datetime_utc'].dt.date >= date_range[0]) & (historical['datetime_utc'].dt.date <= date_range[1])
        filtered_data = historical[mask]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_data['datetime_utc'],
            y=filtered_data['us_aqi'],
            mode='lines',
            name='US AQI',
            line=dict(color='#FF6B6B', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        
        # Add AQI category bands
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
        fig.add_hrect(y0=100,
