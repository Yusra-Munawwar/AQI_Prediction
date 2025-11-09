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
st.markdown("### Karachi Air Quality Monitoring & Forecasting (ML Pipeline Results)")

# Define AQI categories and colors
AQI_CATEGORIES = {
    (0, 50): ('Good', '#00E400'),
    (51, 100): ('Moderate', '#FFFF00'),
    (101, 150): ('Unhealthy for Sensitive Groups', '#FF7E00'),
    (151, 200): ('Unhealthy', '#FF0000'),
    (201, 300): ('Very Unhealthy', '#8F3F97'),
    (301, 500): ('Hazardous', '#7E0023')
}

def get_aqi_color(aqi):
    """Returns the color and category for a given AQI value."""
    for (low, high), (category, color) in AQI_CATEGORIES.items():
        if low <= aqi <= high:
            return color, category
    # Handle values > 500
    return '#7E0023', 'Hazardous'

# Load data
@st.cache_data
def load_data():
    historical_df = None
    predictions_df = None
    comparison_df = None
    
    # 1. Historical Data
    try:
        historical_df = pd.read_csv('data/2years_features_clean.csv', parse_dates=['datetime_utc'])
        historical_df['datetime_utc'] = pd.to_datetime(historical_df['datetime_utc'], errors='coerce')
        historical_df = historical_df.sort_values('datetime_utc').reset_index(drop=True)
    except FileNotFoundError:
        st.error("Historical file 'data/2years_features_clean.csv' not found. Check training script paths.")
    except Exception as e:
        st.error(f"Error loading historical data: {e}")

    # 2. Prediction Results
    try:
        predictions_df = pd.read_csv('data/future_aqi_predictions.csv')
        predictions_df.rename(columns={'datetime': 'datetime_utc'}, inplace=True) 
        predictions_df['datetime_utc'] = pd.to_datetime(predictions_df['datetime_utc'], errors='coerce', utc=True)
    except FileNotFoundError:
        st.warning("Prediction file 'data/future_aqi_predictions.csv' not found. Run training.py first.")
    except Exception as e:
        st.error(f"Error loading prediction data: {e}")

    # 3. Future Comparison Metrics
    try:
        comparison_df = pd.read_csv('data/future_prediction_comparison.csv')
    except FileNotFoundError:
        st.warning("Comparison file 'data/future_prediction_comparison.csv' not found. Run training.py first.")
    except Exception as e:
        st.error(f"Error loading comparison data: {e}")

    return historical_df, predictions_df, comparison_df

historical, predictions, comparison = load_data()

# Check for essential data availability
if historical is None or historical.empty:
    st.info("Waiting for historical data load to proceed...")
    st.stop() 

# --- KEY PERFORMANCE INDICATORS (ALWAYS DISPLAYED) ---
if not historical.empty:
    current_aqi = historical['us_aqi'].iloc[-1]
    color, category = get_aqi_color(current_aqi)
    last_reading_time = historical['datetime_utc'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')
    
    col_aqi, col_time, col_forecast = st.columns([1, 1, 1])

    with col_aqi:
        st.markdown(f"""
        <div style="
            background-color: {color}; 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            color: #1a1a1a; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h5 style="margin: 0; font-size: 0.8em;">CURRENT US AQI</h5>
            <h1 style="margin: 5px 0 0 0; font-weight: bold; font-size: 2.5em;">{current_aqi:.0f}</h1>
            <p style="margin: 0; font-size: 1.1em; font-weight: 600;">{category}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_time:
        st.metric("Last Data Reading Time", last_reading_time)
        
    with col_forecast:
        forecast_hours = len(predictions) if predictions is not None and not predictions.empty else "N/A"
        st.metric("Forecast Horizon", f"{forecast_hours} Hours")
    
    st.markdown("---") # Separator after KPIs

# Sidebar
st.sidebar.header("⚙️ Configuration")
view_mode = st.sidebar.radio("Select View", ["Future Predictions", "Historical Data", "Forecast Comparison"])
st.sidebar.markdown("---")
st.sidebar.info("The prediction models run hourly to provide the latest 96-hour forecast.")

# ============================================================================
# HISTORICAL DATA VIEW
# ============================================================================
if view_mode == "Historical Data":
    st.header("📊 Historical Air Quality Data")
    
    # Historical Key metrics (excluding current AQI, which is now always visible)
    col2, col3, col4 = st.columns(3)
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
    
    # Date range selector (using latest 30 days as default)
    default_start = historical['datetime_utc'].max() - pd.Timedelta(days=30)
    
    date_range = st.date_input(
        "Select Date Range",
        value=(default_start.date(), historical['datetime_utc'].max().date()),
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
            line=dict(color='#3498DB', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        
        # Add AQI category bands
        for (low, high), (category, color) in AQI_CATEGORIES.items():
            fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.1, line_width=0, annotation_text=category, annotation_position="top left")

        
        fig.update_layout(
            title="AQI Time Series (Last 30 Days Default)",
            xaxis_title="Date",
            yaxis_title="US AQI",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pollutant breakdown - Simplified
    st.subheader("🧪 Pollutant Concentrations (Last 30 Days)")
    
    # --- CRITICAL FIX: Dynamically determine which pollutants exist ---
    EXPECTED_POLLUTANTS = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
    pollutant_cols = [p for p in EXPECTED_POLLUTANTS if p in filtered_data.columns]
    
    if not pollutant_cols:
        st.warning("⚠️ Could not find any of the standard raw pollutant columns (pm2_5, o3, etc.) in the historical data file.")
    else:
        fig_pollutant = go.Figure()
        colors = px.colors.qualitative.Bold
        
        for i, p in enumerate(pollutant_cols):
            fig_pollutant.add_trace(go.Scatter(
                x=filtered_data['datetime_utc'],
                y=filtered_data[p], 
                mode='lines',
                name=p.upper(),
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))

        fig_pollutant.update_layout(
            title="Key Pollutant Concentrations",
            xaxis_title="Date", 
            yaxis_title="Concentration (units vary)", 
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_pollutant, use_container_width=True)

# ============================================================================
# FUTURE PREDICTIONS VIEW
# ============================================================================
elif view_mode == "Future Predictions":
    st.header("🔮 96-Hour AQI Forecast")
    
    if predictions is None or predictions.empty:
        st.error("Forecast data is not available. Please run the training script to generate predictions.")
        st.stop()

    # Determine available models dynamically
    EXCLUDED_COLS = ['datetime_utc', 'Actual_AQI', 'Closest_Model']
    available_models = [col for col in predictions.columns if col not in EXCLUDED_COLS]
    
    # Prioritize the best checkpoint model if available
    default_model_index = 0
    if 'best_checkpoint' in available_models:
        default_model_index = available_models.index('best_checkpoint')
        
    selected_model = st.sidebar.selectbox("Select Model for Detailed View", available_models, index=default_model_index)
    
    # Key metrics for prediction
    col1, col2 = st.columns(2)
    with col1:
        avg_pred = predictions[selected_model].mean()
        st.metric(f"Average Predicted AQI ({selected_model.upper()})", f"{avg_pred:.1f}")
    with col2:
        max_pred = predictions[selected_model].max()
        st.metric("Maximum Predicted AQI", f"{max_pred:.0f}")
    
    # Prediction comparison plot
    st.subheader(f"📊 {selected_model.upper()} Forecast vs OWM Actual AQI")
    
    fig_pred = go.Figure()
    
    # OWM Actual AQI (used as the current 'truth' for the next 96 hours)
    fig_pred.add_trace(go.Scatter(
        x=predictions['datetime_utc'],
        y=predictions['Actual_AQI'],
        mode='lines',
        name='OWM Actual AQI (Reference)',
        line=dict(color='#2ECC71', width=3, dash='dot')
    ))
    
    # Predicted AQI
    fig_pred.add_trace(go.Scatter(
        x=predictions['datetime_utc'],
        y=predictions[selected_model],
        mode='lines+markers',
        name=f'{selected_model.upper()} Prediction',
        line=dict(color='#E74C3C', width=2),
        marker=dict(size=4)
    ))
    
    # Add AQI category bands
    for (low, high), (category, color) in AQI_CATEGORIES.items():
        fig_pred.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.1, line_width=0)
        
    fig_pred.update_layout(
        title=f"AQI Forecast - {selected_model.upper()}",
        xaxis_title="Date & Time (UTC)",
        yaxis_title="US AQI",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
    
    # Prediction table
    st.subheader("📋 Detailed Hourly Predictions")
    
    display_cols = ['datetime_utc', 'Actual_AQI', selected_model, 'Closest_Model']
    display_df = predictions[display_cols].copy()
    display_df['datetime_utc'] = display_df['datetime_utc'].dt.strftime('%Y-%m-%d %H:%M')
    display_df[f'{selected_model} Error'] = abs(display_df['Actual_AQI'] - display_df[selected_model]).round(1)
    
    st.dataframe(
        display_df.style
        .highlight_max(subset=[f'{selected_model} Error'], color='#FFDBDB')
        .format({'Actual_AQI': '{:.0f}', selected_model: '{:.0f}'}),
        use_container_width=True, 
        height=400
    )


# ============================================================================
# FORECAST COMPARISON VIEW
# ============================================================================
else:
    st.header("🤖 Forecast Performance Comparison")
    
    if comparison is None or comparison.empty:
        st.error("Comparison metrics are not available. Please run the training script to generate metrics.")
        st.stop()
        
    # Extract best models based on MAE (lower is better)
    best_forecast_model = comparison.loc[comparison['MAE'].idxmin(), 'Model']
    best_mae = comparison['MAE'].min()
    
    st.info(f"The best performing model for the **96-Hour Forecast** (lowest MAE) is **{best_forecast_model.upper()}** (MAE: {best_mae:.3f}).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast MAE
        fig_mae = px.bar(comparison.sort_values('MAE'), x='Model', y='MAE', 
                         title='Mean Absolute Error (Lower is Better)', color='MAE',
                         color_continuous_scale='Reds',
                         text='MAE')
        fig_mae.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_mae.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        # Forecast R²
        fig_r2_fut = px.bar(comparison.sort_values('R²', ascending=False), x='Model', y='R²', 
                            title='R² Score (Higher is Better)', color='R²',
                            color_continuous_scale='RdYlGn',
                            text='R²')
        fig_r2_fut.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_r2_fut.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_r2_fut, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Forecast Metrics")
    
    st.dataframe(comparison.style
                 .highlight_min(subset=['MAE', 'RMSE'], color='#D5F4E6')
                 .highlight_max(subset=['R²'], color='#E52B50')
                 .format({'MAE': '{:.3f}', 'RMSE': '{:.3f}', 'R²': '{:.3f}', 'MAPE': '{:.2f}%'}),
                 use_container_width=True)


# Footer
st.markdown("---")
st.markdown("**Data Source:** OpenWeatherMap API | **Location:** Karachi, Pakistan")
st.markdown(f"**Last Dashboard View:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("Note: The 'Actual AQI' in the prediction view is the OWM API's own 96-hour forecast, used as the current target for model comparison.")
