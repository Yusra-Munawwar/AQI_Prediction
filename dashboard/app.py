import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import os
import json
import numpy as np
import requests
import statsmodels.api as sm
import torch
import torch.nn as nn
import pickle
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Constants for Karachi
LAT = 24.8607
LON = 67.0011

# UTC
UTC = timezone.utc

# AQI Breakpoints (from EPA Technical Assistance Document)
breakpoints = {
    'pm2_5': [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
              (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
              (350.5, 500.4, 401, 500)],
    'pm10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400),
             (505, 604, 401, 500)],
    'o3': [(0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
           (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)],
    'no2': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400),
            (1650, 2049, 401, 500)],
    'so2': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
            (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400),
            (805, 1004, 401, 500)],
    'co': [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
           (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400),
           (40.5, 50.4, 401, 500)]
}

def linear_aqi(c, breaks):
    """Linear interpolation for AQI sub-index."""
    for cl, ch, il, ih in breaks:
        if cl <= c <= ch:
            if ch == cl:
                return il
            return il + (ih - il) * (c - cl) / (ch - cl)
    return min(500, max(0, c))  # Clamp to 0-500

def calc_us_aqi(row):
    """Calculate US AQI as max of sub-indices for major pollutants."""
    subs = []
    pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    # PM2.5 (μg/m³)
    if 'pm2_5' in row:
        subs.append(linear_aqi(row['pm2_5'], breakpoints['pm2_5']))
    
    # PM10 (μg/m³)
    if 'pm10' in row:
        subs.append(linear_aqi(row['pm10'], breakpoints['pm10']))
    
    # O3 (μg/m³ to ppm)
    if 'o3' in row:
        c_ppm = row['o3'] / 1960.0
        subs.append(linear_aqi(c_ppm, breakpoints['o3']))
    
    # NO2 (μg/m³ to ppb)
    if 'no2' in row:
        c_ppb = row['no2'] * (24.45 / 46.0)  # ≈0.532 * μg/m³
        subs.append(linear_aqi(c_ppb, breakpoints['no2']))
    
    # SO2 (μg/m³ to ppb)
    if 'so2' in row:
        c_ppb = row['so2'] * (24.45 / 64.0)  # ≈0.382 * μg/m³
        subs.append(linear_aqi(c_ppb, breakpoints['so2']))
    
    # CO (mg/m³ to ppm)
    if 'co' in row:
        c_ppm = row['co'] / 1.145
        subs.append(linear_aqi(c_ppm, breakpoints['co']))
    
    return max(subs) if subs else 0

def engineer_features(df):
    """Add temporal features."""
    df = df.copy()
    df['hour'] = df['datetime_utc'].dt.hour
    df['month'] = df['datetime_utc'].dt.month
    df['day_of_week'] = df['datetime_utc'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def calc_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, R², and MAPE"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
   
    # Safe MAPE calculation
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
   
    return mae, rmse, r2, mape

# Generate future predictions (cached for 1 hour)
@st.cache_data(ttl=3600)
def generate_future_predictions():
    """Generate future AQI predictions and save CSVs."""
    os.makedirs('data', exist_ok=True)
    
    # Load historical for cleaning and features
    historical = pd.read_csv('data/2years_features.csv', parse_dates=['datetime_utc'])
    df_clean = historical
    
    # Load selected features
    with open('selected_features.json', 'r') as f:
        selected_features = json.load(f)
    
    # Fetch forecast
    def fetch_forecast_pollutants(lat, lon, api_key):
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"
        r = requests.get(url).json()
        records = []
        for item in r["list"]:
            dt = datetime.fromtimestamp(item["dt"], tz=UTC)
            comp = item["components"]
            aqi_1_5 = item["main"]["aqi"]
            temp_row = {"datetime_utc": dt, **comp}
            usaqi = calc_us_aqi(pd.Series(temp_row))
            usaqi = np.clip(usaqi, 0, 500)
            records.append({
                "datetime_utc": dt,
                "aqi_api": aqi_1_5,
                "Actual_AQI": usaqi,
                **comp
            })
        return pd.DataFrame(records)
    
    api_key = os.getenv("OWM_API_KEY")
    if not api_key:
        st.warning("OWM_API_KEY not set. Skipping forecast fetch.")
        return
    
    forecast_df = fetch_forecast_pollutants(LAT, LON, api_key)
    print(f"✅ Forecast pollutant data received: {forecast_df.shape[0]} hours")
    
    # Clean & Feature Engineering
    pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    for col in pollutants:
        if col in forecast_df.columns:
            forecast_df[col] = forecast_df[col].clip(lower=0)
            forecast_df[col] = forecast_df[col].fillna(df_clean[col].median())
            forecast_df[col] = forecast_df[col].clip(upper=df_clean[col].quantile(0.995))
    
    forecast_feat = engineer_features(forecast_df)
    X_future = forecast_feat[selected_features].fillna(0).values
    
    # Scale
    scaler = joblib.load('scaler.pkl')
    X_future_scaled = scaler.transform(X_future)
    
    # Predictions
    pred = pd.DataFrame({
        "datetime_utc": forecast_df["datetime_utc"],
        "Actual_AQI": forecast_df["Actual_AQI"].astype(float)
    })
    
    # 1. Linear Regression
    model1 = pickle.load(open('linear_model.pkl', 'rb'))
    X_sm = sm.add_constant(X_future_scaled)
    pred["Linear"] = model1.predict(X_sm)
    
    # 2. Polynomial
    coeffs = np.load('poly_coeffs.npy')
    pm2_idx = selected_features.index('pm2_5') if 'pm2_5' in selected_features else 0
    pred["Polynomial"] = np.polyval(coeffs, X_future_scaled[:, pm2_idx])
    
    # 3. PyTorch Linear
    class LinearModel(nn.Module):
        def __init__(self, n): super().__init__(); self.linear = nn.Linear(n, 1)
        def forward(self, x): return self.linear(x)
    
    m3 = LinearModel(len(selected_features))
    m3.load_state_dict(torch.load("linear_model.pth"))
    m3.eval()
    pred["PyTorch_Linear"] = m3(torch.tensor(X_future_scaled, dtype=torch.float32)).detach().numpy().flatten()
    
    # 4. PyTorch MLP
    class MLP(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fc1 = nn.Linear(n, 64); self.fc2 = nn.Linear(64, 32); self.fc3 = nn.Linear(32, 1)
            self.r = nn.ReLU()
        def forward(self, x): return self.fc3(self.r(self.fc2(self.r(self.fc1(x)))))
    
    m4 = MLP(len(selected_features))
    m4.load_state_dict(torch.load("mlp_model.pth"))
    m4.eval()
    pred["PyTorch_MLP"] = m4(torch.tensor(X_future_scaled, dtype=torch.float32)).detach().numpy().flatten()
    
    # 5. GB
    gb = joblib.load("gb_model.pkl")
    pred["GB"] = gb.predict(X_future)
    
    # 6. XGB
    xg = xgb.XGBRegressor()
    xg.load_model("xgb_model.json")
    pred["XGB"] = xg.predict(X_future)
    
    # 7. RF
    rf = joblib.load("rf_model.pkl")
    pred["RF"] = rf.predict(X_future)
    
    # Ensemble
    pred["Ensemble"] = pred[['Linear', 'Polynomial', 'PyTorch_Linear', 'PyTorch_MLP', 'GB', 'XGB', 'RF']].mean(axis=1)
    
    # Clip
    for col in pred.columns[2:]:
        pred[col] = np.clip(pred[col], 0, 500)
    
    # Closest Model
    model_cols = pred.columns.drop(["datetime_utc", "Actual_AQI"])
    pred["Closest_Model"] = (pred[model_cols].sub(pred["Actual_AQI"], axis=0).abs().idxmin(axis=1))
    
    # Summary Metrics
    model_map = {
        'Linear': 'Linear', 'Polynomial': 'Polynomial', 'PyTorch_Linear': 'PyTorch Linear',
        'PyTorch_MLP': 'PyTorch MLP', 'GB': 'Gradient Boosting', 'XGB': 'XGBoost',
        'RF': 'Random Forest', 'Ensemble': 'Ensemble'
    }
    model_cols = [col for col in pred.columns if col not in ['datetime_utc', 'Actual_AQI', 'Closest_Model']]
    summary_data = []
    y_true = pred['Actual_AQI']
    for model_col in model_cols:
        if model_col in model_map:
            y_pred = pred[model_col]
            mae, rmse, r2, mape = calc_metrics(y_true, y_pred)
            summary_data.append({
                'Model': model_map[model_col], 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape
            })
    comparison_summary = pd.DataFrame(summary_data).sort_values('R²', ascending=False)
    
    # Save
    pred.to_csv("data/future_aqi_predictions.csv", index=False)
    comparison_summary.to_csv("data/future_prediction_comparison.csv", index=False)
    
    print("💾 Saved future predictions and comparison CSVs.")

# Page config
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Generate predictions
generate_future_predictions()

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
