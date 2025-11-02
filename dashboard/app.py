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
        fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
        fig.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="AQI Time Series",
            xaxis_title="Date",
            yaxis_title="US AQI",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pollutant breakdown
    st.subheader("🧪 Pollutant Concentrations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PM2.5 and PM10
        fig_pm = go.Figure()
        fig_pm.add_trace(go.Scatter(x=filtered_data['datetime_utc'], y=filtered_data['pm2_5'], 
                                    mode='lines', name='PM2.5', line=dict(color='#E74C3C')))
        fig_pm.add_trace(go.Scatter(x=filtered_data['datetime_utc'], y=filtered_data['pm10'], 
                                    mode='lines', name='PM10', line=dict(color='#9B59B6')))
        fig_pm.update_layout(title="Particulate Matter (PM2.5 & PM10)", 
                            xaxis_title="Date", yaxis_title="µg/m³", height=400)
        st.plotly_chart(fig_pm, use_container_width=True)
    
    with col2:
        # Gases
        fig_gas = go.Figure()
        fig_gas.add_trace(go.Scatter(x=filtered_data['datetime_utc'], y=filtered_data['o3'], 
                                     mode='lines', name='O3', line=dict(color='#3498DB')))
        fig_gas.add_trace(go.Scatter(x=filtered_data['datetime_utc'], y=filtered_data['no2'], 
                                     mode='lines', name='NO2', line=dict(color='#E67E22')))
        fig_gas.add_trace(go.Scatter(x=filtered_data['datetime_utc'], y=filtered_data['so2'], 
                                     mode='lines', name='SO2', line=dict(color='#F39C12')))
        fig_gas.add_trace(go.Scatter(x=filtered_data['datetime_utc'], y=filtered_data['co'], 
                                     mode='lines', name='CO', line=dict(color='#1ABC9C')))
        fig_gas.update_layout(title="Gaseous Pollutants", 
                             xaxis_title="Date", yaxis_title="µg/m³", height=400)
        st.plotly_chart(fig_gas, use_container_width=True)
    
    # AQI distribution
    st.subheader("📈 AQI Distribution by Category")
    
    aqi_cats = pd.cut(filtered_data['us_aqi'], 
                      bins=[0, 50, 100, 150, 200, 300, 500],
                      labels=['Good', 'Moderate', 'Unhealthy-SG', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
    
    cat_counts = aqi_cats.value_counts().sort_index()
    colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']
    
    fig_dist = go.Figure(data=[go.Bar(
        x=cat_counts.index,
        y=cat_counts.values,
        marker_color=colors[:len(cat_counts)]
    )])
    fig_dist.update_layout(title="AQI Category Distribution", 
                          xaxis_title="Category", yaxis_title="Count", height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================================
# FUTURE PREDICTIONS VIEW
# ============================================================================
elif view_mode == "Future Predictions":
    st.header("🔮 Future AQI Predictions")
    
    # Model selector
    model_options = ['Ensemble', 'Linear', 'Polynomial', 'PyTorch_Linear', 'PyTorch_MLP', 'GB', 'XGB', 'RF']
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_pred = predictions[selected_model].mean()
        st.metric(f"Average Predicted AQI ({selected_model})", f"{avg_pred:.1f}")
    with col2:
        max_pred = predictions[selected_model].max()
        st.metric("Maximum Predicted", f"{max_pred:.0f}")
    with col3:
        hours_forecast = len(predictions)
        st.metric("Forecast Hours", f"{hours_forecast}")
    
    # Prediction comparison plot
    st.subheader(f"📊 Actual vs Predicted AQI - {selected_model}")
    
    fig_pred = go.Figure()
    
    # Actual AQI
    fig_pred.add_trace(go.Scatter(
        x=predictions['datetime_utc'],
        y=predictions['Actual_AQI'],
        mode='lines+markers',
        name='Actual AQI',
        line=dict(color='#2ECC71', width=2),
        marker=dict(size=6)
    ))
    
    # Predicted AQI
    fig_pred.add_trace(go.Scatter(
        x=predictions['datetime_utc'],
        y=predictions[selected_model],
        mode='lines+markers',
        name=f'{selected_model} Prediction',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig_pred.update_layout(
        title=f"AQI Forecast - {selected_model}",
        xaxis_title="Date & Time",
        yaxis_title="US AQI",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Bar chart comparison
    st.subheader("📊 Hourly Comparison")
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=predictions['datetime_utc'],
        y=predictions['Actual_AQI'],
        name='Actual',
        marker_color='#2ECC71'
    ))
    fig_bar.add_trace(go.Bar(
        x=predictions['datetime_utc'],
        y=predictions[selected_model],
        name=selected_model,
        marker_color='#E74C3C'
    ))
    
    fig_bar.update_layout(
        title="Actual vs Predicted (Bar Chart)",
        xaxis_title="Date & Time",
        yaxis_title="US AQI",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Prediction table
    st.subheader("📋 Detailed Predictions")
    
    display_cols = ['datetime_utc', 'Actual_AQI', selected_model, 'Closest_Model']
    display_df = predictions[display_cols].copy()
    display_df['datetime_utc'] = display_df['datetime_utc'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['Error'] = abs(display_df['Actual_AQI'] - display_df[selected_model])
    
    st.dataframe(display_df.style.highlight_max(subset=['Error'], color='#FFE5E5'), 
                 use_container_width=True, height=400)

# ============================================================================
# MODEL COMPARISON VIEW
# ============================================================================
else:
    st.header("🤖 Model Performance Comparison")
    
    # Training results
    st.subheader("📊 Test Set Performance")
    
    fig_train = go.Figure()
    
    fig_train.add_trace(go.Bar(
        name='MAE',
        x=training_results['Model'],
        y=training_results['Test MAE'],
        marker_color='#3498DB'
    ))
    fig_train.add_trace(go.Bar(
        name='RMSE',
        x=training_results['Model'],
        y=training_results['Test RMSE'],
        marker_color='#E74C3C'
    ))
    
    fig_train.update_layout(
        title="Model Error Metrics (Lower is Better)",
        xaxis_title="Model",
        yaxis_title="Error Value",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_train, use_container_width=True)
    
    # R² scores
    fig_r2 = go.Figure(data=[
        go.Bar(
            x=training_results['Model'],
            y=training_results['Test R²'],
            marker_color=training_results['Test R²'],
            marker_colorscale='RdYlGn',
            text=training_results['Test R²'].round(3),
            textposition='outside'
        )
    ])
    fig_r2.update_layout(
        title="R² Scores (Higher is Better)",
        xaxis_title="Model",
        yaxis_title="R² Score",
        height=400
    )
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # Future prediction performance
    st.subheader("🔮 Forecast Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mae = px.bar(comparison.sort_values('MAE'), x='Model', y='MAE', 
                        title='Mean Absolute Error', color='MAE',
                        color_continuous_scale='Reds')
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        fig_r2_fut = px.bar(comparison.sort_values('R²', ascending=False), x='Model', y='R²', 
                           title='R² Score', color='R²',
                           color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_r2_fut, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Performance**")
        st.dataframe(training_results.style.highlight_max(subset=['Test R²'], color='#D5F4E6')
                                          .highlight_min(subset=['Test MAE', 'Test RMSE'], color='#D5F4E6'),
                    use_container_width=True)
    
    with col2:
        st.markdown("**Forecast Performance**")
        st.dataframe(comparison.style.highlight_max(subset=['R²'], color='#D5F4E6')
                                     .highlight_min(subset=['MAE', 'RMSE'], color='#D5F4E6'),
                    use_container_width=True)
    
    # Best model recommendation
    best_model_train = training_results.loc[training_results['Test R²'].idxmax(), 'Model']
    best_model_forecast = comparison.loc[comparison['R²'].idxmax(), 'Model']
    
    st.success(f"🏆 **Best Training Model:** {best_model_train}")
    st.success(f"🏆 **Best Forecast Model:** {best_model_forecast}")

# Footer
st.markdown("---")
st.markdown("**Data Source:** OpenWeatherMap API | **Location:** Karachi, Pakistan")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
