# train.py
import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import requests
import hopsworks
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# CONFIGURATION
# ==========================================
UTC = timezone.utc
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Coordinates & API (set via env vars)
LAT = float(os.getenv("LAT", "40.7128"))  # Default: NYC
LON = float(os.getenv("LON", "-74.0060"))
OWM_API_KEY = os.getenv("OWM_API_KEY")

# Feature group settings
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# ==========================================
# HELPER: US AQI Calculation
# ==========================================
def calc_us_aqi(row):
    """Convert pollutant concentrations to US AQI (simplified version)"""
    pm25 = row['pm2_5']
    pm10 = row['pm10']
    o3 = row['o3']
    no2 = row['no2']
    co = row['co']
    so2 = row['so2']

    # Simplified AQI breakpoints (EPA-based)
    def aqi_from_breakpoints(c, breakpoints):
        for (low_c, high_c), (low_i, high_i) in breakpoints:
            if low_c <= c <= high_c:
                return ((high_i - low_i) / (high_c - low_c)) * (c - low_c) + low_i
        return 0

    pm25_break = [((0, 12), (0, 50)), ((12.1, 35.4), (51, 100)), ((35.5, 55.4), (101, 150)),
                  ((55.5, 150.4), (151, 200)), ((150.5, 250.4), (201, 300)), ((250.5, 500.4), (301, 500))]
    pm10_break = [((0, 54), (0, 50)), ((55, 154), (51, 100)), ((155, 254), (101, 150)),
                  ((255, 354), (151, 200)), ((355, 424), (201, 300)), ((425, 604), (301, 500))]
    o3_break = [((0, 54), (0, 50)), ((55, 70), (51, 100)), ((71, 85), (101, 150)),
                ((86, 105), (151, 200)), ((106, 200), (201, 300))]
    no2_break = [((0, 53), (0, 50)), ((54, 100), (51, 100)), ((101, 360), (101, 150)),
                 ((361, 649), (151, 200)), ((650, 1249), (201, 300)), ((1250, 2049), (301, 500))]
    co_break = [((0, 4.4), (0, 50)), ((4.5, 9.4), (51, 100)), ((9.5, 12.4), (101, 150)),
                ((12.5, 15.4), (151, 200)), ((15.5, 30.4), (201, 300)), ((30.5, 50.4), (301, 500))]
    so2_break = [((0, 35), (0, 50)), ((36, 75), (51, 100)), ((76, 185), (101, 150)),
                 ((186, 304), (151, 200)), ((305, 604), (201, 300)), ((605, 1004), (301, 500))]

    aqi_pm25 = aqi_from_breakpoints(pm25, pm25_break)
    aqi_pm10 = aqi_from_breakpoints(pm10, pm10_break)
    aqi_o3 = aqi_from_breakpoints(o3, pm10_break)
    aqi_no2 = aqi_from_breakpoints(no2, no2_break)
    aqi_co = aqi_from_breakpoints(co, co_break)
    aqi_so2 = aqi_from_breakpoints(so2, so2_break)

    return max(aqi_pm25, aqi_pm10, aqi_o3, aqi_no2, aqi_co, aqi_so2, 0)

# ==========================================
# SECTION 1: FETCH FROM HOPSWORKS
# ==========================================
def fetch_from_hopsworks(feature_group_name="aqi_features", version=1, for_training=False):
    print("\n" + "="*70)
    print(f"{'FETCHING FROM HOPSWORKS FOR VERIFICATION' if not for_training else 'SECTION 1: FETCHING FEATURES FROM HOPSWORKS'}")
    print("="*70)
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        df = feature_group.read()

        if 'datetime_utc' in df.columns and df['datetime_utc'].dtype == 'int64':
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', errors='coerce')
            invalid_mask = df['datetime_utc'].isna() | (df['datetime_utc'] < pd.Timestamp('1678-01-01')) | (df['datetime_utc'] > pd.Timestamp('2262-04-11'))
            if invalid_mask.sum() > 0:
                print(f"Warning: Dropped {invalid_mask.sum()} rows with invalid timestamps.")
                df = df[~invalid_mask].reset_index(drop=True)
            int_cols = ['us_aqi', 'hour', 'month', 'day_of_week', 'is_weekend']
            for col in int_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        print(f"Success: Fetched from Hopsworks! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error: Hopsworks fetch failed: {e}. Falling back to local CSV.")
        try:
            df = pd.read_csv('2years_features.csv', parse_dates=['datetime_utc'])
            print(f"Success: Fallback to local CSV! Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise ValueError("No local CSV found—ensure upload happened first.")

# ==========================================
# DATASET COMPARISON
# ==========================================
def compare_datasets(local_df, hops_df, tolerance=1e-6):
    print("\n" + "="*70)
    print("DATASET COMPARISON")
    print("="*70)

    local_sorted = local_df.sort_values('datetime_utc').reset_index(drop=True)
    hops_sorted = hops_df.sort_values('datetime_utc').reset_index(drop=True)

    if len(local_sorted) != len(hops_sorted):
        print(f"Warning: Row count mismatch: Local={len(local_sorted)}, Hops={len(hops_sorted)}")
        min_len = min(len(local_sorted), len(hops_sorted))
        local_sorted = local_sorted.iloc[:min_len]
        hops_sorted = hops_sorted.iloc[:min_len]
        print(f"   → Truncated to {min_len} rows.")

    print(f"Local shape: {local_sorted.shape}, Hops shape: {hops_sorted.shape}")

    local_cols = set(local_sorted.columns)
    hops_cols = set(hops_sorted.columns)
    expected_extras = {'id'}
    core_hops_cols = hops_cols - expected_extras

    if local_cols == core_hops_cols:
        print("Success: Core columns match")
    else:
        print("Warning: Column mismatch")

    numeric_cols = local_sorted.select_dtypes(include=[np.number]).columns
    if 'id' in numeric_cols:
        numeric_cols = numeric_cols.drop('id')

    sample_local = local_sorted[numeric_cols].head(5).round(4)
    sample_hops = hops_sorted[numeric_cols.intersection(hops_sorted.columns)].head(5).round(4)

    if sample_local.equals(sample_hops):
        print("Success: Sample values match")
    else:
        close = np.allclose(sample_local.values, sample_hops.values, atol=tolerance)
        print("Success: Values close" if close else "Error: Values differ")

    if 'us_aqi' in local_sorted.columns:
        local_mean = local_sorted['us_aqi'].mean()
        hops_mean = hops_sorted['us_aqi'].mean()
        if abs(local_mean - hops_mean) < tolerance:
            print("Success: Target stats match")

    return (local_cols == core_hops_cols) and (abs(local_mean - hops_mean) < tolerance if 'us_aqi' in local_sorted.columns else True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()
    df['hour'] = df['datetime_utc'].dt.hour
    df['month'] = df['datetime_utc'].dt.month
    df['day_of_week'] = df['datetime_utc'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

# ==========================================
# METRICS
# ==========================================
def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    return mae, rmse, r2, mape

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Fetch & Verify ---
    df_feat = fetch_from_hopsworks(for_training=True)
    df_feat = engineer_features(df_feat)

    local_df = pd.read_csv('2years_features.csv', parse_dates=['datetime_utc'])
    local_df = engineer_features(local_df)
    hops_df = fetch_from_hopsworks(for_training=False)
    is_match = compare_datasets(local_df, hops_df)
    print("\nSuccess: CORE MATCH! Ready for training!" if is_match else "\nWarning: Minor differences detected.")

    # --- Data Prep ---
    print("\n" + "="*70)
    print("SECTION 2: DATA PREPARATION & SPLITTING")
    print("="*70)

    exclude = ['us_aqi', 'aqi_category', 'id', 'datetime_utc', 'aqi']
    feature_cols = [col for col in df_feat.columns if col not in exclude]
    corr = df_feat[feature_cols + ['us_aqi']].corr()['us_aqi'].sort_values(ascending=False)
    selected_features = [f for f in corr.index[1:] if abs(corr[f]) > 0.3]
    print(f"Success: Using {len(selected_features)} features: {selected_features}")

    X = df_feat[selected_features].fillna(0).values
    y = df_feat['us_aqi'].values

    split_train = int(0.7 * len(X))
    split_val = int(0.8 * len(X))
    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler.pkl'))

    tscv = TimeSeriesSplit(n_splits=5)

    # --- Model Training ---
    print("\n" + "="*70)
    print("SECTION 3: MODEL TRAINING (7 MODELS)")
    print("="*70)
    results = {}

    # 1. Linear
    X_train_sm = sm.add_constant(X_train_scaled)
    model1 = sm.OLS(y_train, X_train_sm).fit()
    y_test_pred1 = model1.predict(sm.add_constant(X_test_scaled))
    mae1, rmse1, r21, mape1 = calc_metrics(y_test, y_test_pred1)
    cv_scores1 = [r2_score(y_train[val_idx], sm.OLS(y_train[train_idx], sm.add_constant(X_train_scaled[train_idx])).fit().predict(sm.add_constant(X_train_scaled[val_idx]))) for train_idx, val_idx in tscv.split(X_train_scaled)]
    results['linear'] = {'name': 'Linear Regression', 'test_r2': r21, 'test_rmse': rmse1, 'test_mae': mae1, 'test_mape': mape1, 'cv_r2': np.mean(cv_scores1)}
    with open(os.path.join(DATA_DIR, 'linear_model.pkl'), 'wb') as f:
        pickle.dump(model1, f)

    # 2. Polynomial
    pm2_idx = selected_features.index('pm2_5') if 'pm2_5' in selected_features else 0
    p_coeffs = np.polyfit(X_train_scaled[:, pm2_idx], y_train, 2)
    y_test_pred2 = np.polyval(p_coeffs, X_test_scaled[:, pm2_idx])
    mae2, rmse2, r22, mape2 = calc_metrics(y_test, y_test_pred2)
    results['poly'] = {'name': 'Polynomial Regression', 'test_r2': r22, 'test_rmse': rmse2, 'test_mae': mae2, 'test_mape': mape2}
    np.save(os.path.join(DATA_DIR, 'poly_coeffs.npy'), p_coeffs)

    # 3. PyTorch Linear
    class LinearModel(nn.Module):
        def __init__(self, n): super().__init__(); self.linear = nn.Linear(n, 1)
        def forward(self, x): return self.linear(x)
    device = torch.device('cpu')
    model3 = LinearModel(len(selected_features)).to(device)
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    opt = optim.Adam(model3.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for _ in range(100):
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model3(bx), by)
            loss.backward()
            opt.step()
    model3.eval()
    y_test_pred3 = model3(torch.tensor(X_test_scaled, dtype=torch.float32)).detach().cpu().numpy().flatten()
    mae3, rmse3, r23, mape3 = calc_metrics(y_test, y_test_pred3)
    results['pytorch_linear'] = {'name': 'PyTorch Linear', 'test_r2': r23, 'test_rmse': rmse3, 'test_mae': mae3, 'test_mape': mape3}
    torch.save(model3.state_dict(), os.path.join(DATA_DIR, 'linear_model.pth'))

    # 4. MLP
    class MLP(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fc1 = nn.Linear(n, 64); self.fc2 = nn.Linear(64, 32); self.fc3 = nn.Linear(32, 1)
            self.r = nn.ReLU(); self.d = nn.Dropout(0.3)
        def forward(self, x): x = self.r(self.fc1(x)); x = self.d(x); x = self.r(self.fc2(x)); x = self.d(x); return self.fc3(x)
    model4 = MLP(len(selected_features)).to(device)
    opt = optim.Adam(model4.parameters(), lr=0.001)
    for _ in range(100):
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model4(bx), by)
            loss.backward()
            opt.step()
    model4.eval()
    y_test_pred4 = model4(torch.tensor(X_test_scaled, dtype=torch.float32)).detach().cpu().numpy().flatten()
    mae4, rmse4, r24, mape4 = calc_metrics(y_test, y_test_pred4)
    results['mlp'] = {'name': 'PyTorch MLP', 'test_r2': r24, 'test_rmse': rmse4, 'test_mae': mae4, 'test_mape': mape4}
    torch.save(model4.state_dict(), os.path.join(DATA_DIR, 'mlp_model.pth'))

    # 5. GB
    model5 = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.7, min_samples_split=50, min_samples_leaf=20, random_state=42)
    model5.fit(X_train, y_train)
    y_test_pred5 = model5.predict(X_test)
    mae5, rmse5, r25, mape5 = calc_metrics(y_test, y_test_pred5)
    cv_scores5 = [r2_score(y_train[val_idx], GradientBoostingRegressor(**model5.get_params()).fit(X_train[train_idx], y_train[train_idx]).predict(X_train[val_idx])) for train_idx, val_idx in tscv.split(X_train)]
    results['gb'] = {'name': 'Gradient Boosting', 'test_r2': r25, 'test_rmse': rmse5, 'test_mae': mae5, 'test_mape': mape5, 'cv_r2': np.mean(cv_scores5)}
    joblib.dump(model5, os.path.join(DATA_DIR, 'gb_model.pkl'))

    # 6. XGBoost
    model6 = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbosity=0)
    model6.fit(X_train, y_train)
    y_test_pred6 = model6.predict(X_test)
    mae6, rmse6, r26, mape6 = calc_metrics(y_test, y_test_pred6)
    cv_scores6 = [r2_score(y_train[val_idx], xgb.XGBRegressor(**model6.get_params()).fit(X_train[train_idx], y_train[train_idx]).predict(X_train[val_idx])) for train_idx, val_idx in tscv.split(X_train)]
    results['xgb'] = {'name': 'XGBoost', 'test_r2': r26, 'test_rmse': rmse6, 'test_mae': mae6, 'test_mape': mape6, 'cv_r2': np.mean(cv_scores6)}
    model6.save_model(os.path.join(DATA_DIR, 'xgb_model.json'))

    # 7. RF
    model7 = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_split=50, min_samples_leaf=20, max_features=0.5, random_state=42)
    model7.fit(X_train, y_train)
    y_test_pred7 = model7.predict(X_test)
    mae7, rmse7, r27, mape7 = calc_metrics(y_test, y_test_pred7)
    cv_scores7 = [r2_score(y_train[val_idx], RandomForestRegressor(**model7.get_params()).fit(X_train[train_idx], y_train[train_idx]).predict(X_train[val_idx])) for train_idx, val_idx in tscv.split(X_train)]
    results['rf'] = {'name': 'Random Forest', 'test_r2': r27, 'test_rmse': rmse7, 'test_mae': mae7, 'test_mape': mape7, 'cv_r2': np.mean(cv_scores7)}
    joblib.dump(model7, os.path.join(DATA_DIR, 'rf_model.pkl'))

    # Save training summary
    summary_df = pd.DataFrame([{
        'Model': v['name'],
        'Test MAE': v['test_mae'],
        'Test RMSE': v['test_rmse'],
        'Test R²': v['test_r2'],
        'Test MAPE (%)': v['test_mape'],
        'CV R²': v.get('cv_r2', np.nan)
    } for v in results.values()]).sort_values('Test R²', ascending=False)
    summary_df.to_csv(os.path.join(DATA_DIR, 'training_results.csv'), index=False)
    print(summary_df)

    # ==========================================
    # SECTION 6: FUTURE PREDICTION
    # ==========================================
    print("\n" + "="*70)
    print("SECTION 6: FUTURE AQI PREDICTION")
    print("="*70)

    def fetch_forecast_pollutants(lat, lon, api_key):
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"
        r = requests.get(url).json()
        records = []
        for item in r["list"]:
            dt = datetime.fromtimestamp(item["dt"], tz=UTC)
            comp = item["components"]
            row = {"datetime_utc": dt, **comp}
            usaqi = calc_us_aqi(pd.Series(row))
            records.append({
                "datetime_utc": dt,
                "aqi_api": item["main"]["aqi"],
                "Actual_AQI": np.clip(usaqi, 0, 500),
                **comp
            })
        return pd.DataFrame(records)

    if not OWM_API_KEY:
        raise ValueError("OWM_API_KEY not set!")

    forecast_df = fetch_forecast_pollutants(LAT, LON, OWM_API_KEY)
    print(f"Success: Forecast data: {forecast_df.shape[0]} hours")

    # Clean & engineer
    pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    df_clean = pd.read_csv('2years_features.csv', parse_dates=['datetime_utc'])
    for col in pollutants:
        forecast_df[col] = forecast_df[col].clip(lower=0)
        forecast_df[col] = forecast_df[col].fillna(df_clean[col].median())
        forecast_df[col] = forecast_df[col].clip(upper=df_clean[col].quantile(0.995))

    forecast_feat = engineer_features(forecast_df)
    X_future = forecast_feat[selected_features].fillna(0)
    X_future_scaled = scaler.transform(X_future)

    pred = pd.DataFrame({"datetime_utc": forecast_df["datetime_utc"], "Actual_AQI": forecast_df["Actual_AQI"].astype(float)})

    # Load & predict
    model1 = pickle.load(open(os.path.join(DATA_DIR, 'linear_model.pkl'), 'rb'))
    pred["Linear"] = model1.predict(sm.add_constant(X_future_scaled, has_constant='add'))

    coeffs = np.load(os.path.join(DATA_DIR, 'poly_coeffs.npy'))
    pm2_idx = selected_features.index('pm2_5') if 'pm2_5' in selected_features else 0
    pred["Polynomial"] = np.polyval(coeffs, X_future_scaled[:, pm2_idx])

    m3 = LinearModel(len(selected_features))
    m3.load_state_dict(torch.load(os.path.join(DATA_DIR, 'linear_model.pth')))
    m3.eval()
    pred["PyTorch_Linear"] = m3(torch.tensor(X_future_scaled, dtype=torch.float32)).detach().numpy().flatten()

    m4 = MLP(len(selected_features))
    m4.load_state_dict(torch.load(os.path.join(DATA_DIR, 'mlp_model.pth')))
    m4.eval()
    pred["PyTorch_MLP"] = m4(torch.tensor(X_future_scaled, dtype=torch.float32)).detach().numpy().flatten()

    pred["GB"] = joblib.load(os.path.join(DATA_DIR, 'gb_model.pkl')).predict(X_future)
    xg = xgb.XGBRegressor(); xg.load_model(os.path.join(DATA_DIR, 'xgb_model.json'))
    pred["XGB"] = xg.predict(X_future)
    pred["RF"] = joblib.load(os.path.join(DATA_DIR, 'rf_model.pkl')).predict(X_future)

    pred["Ensemble"] = pred[['Linear','Polynomial','PyTorch_Linear','PyTorch_MLP','GB','XGB','RF']].mean(axis=1)
    for col in pred.columns[2:]:
        pred[col] = np.clip(pred[col], 0, 500)

    pred["Closest_Model"] = pred.iloc[:, 2:-1].sub(pred["Actual_AQI"], axis=0).abs().idxmin(axis=1)

    # Summary
    model_map = {'Linear': 'Linear', 'Polynomial': 'Polynomial', 'PyTorch_Linear': 'PyTorch Linear', 'PyTorch_MLP': 'PyTorch MLP', 'GB': 'Gradient Boosting', 'XGB': 'XGBoost', 'RF': 'Random Forest', 'Ensemble': 'Ensemble'}
    summary_data = []
    for col in pred.columns[2:-1]:
        mae, rmse, r2, mape = calc_metrics(pred['Actual_AQI'], pred[col])
        summary_data.append({'Model': model_map.get(col, col), 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape})
    comparison_summary = pd.DataFrame(summary_data).sort_values('R²', ascending=False)

    # Save
    pred.to_csv(os.path.join(DATA_DIR, "future_aqi_predictions.csv"), index=False)
    comparison_summary.to_csv(os.path.join(DATA_DIR, "future_prediction_comparison.csv"), index=False)
    print("Success: Saved: future_aqi_predictions.csv, future_prediction_comparison.csv, training_results.csv")

    print("\nSample Predictions:")
    print(pred.head(10))
    print("\nForecast Performance:")
    print(comparison_summary)
