import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import pickle
import joblib
import hopsworks
import os
import requests
from datetime import datetime, timezone

# Custom metrics
def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, r2, mape

# Fetch from Hopsworks
def fetch_from_hopsworks(feature_group_name="aqi_features", version=1):
    """Fetch data from Hopsworks with fallback to local"""
    print("\n" + "="*70)
    print("FETCHING FEATURES FROM HOPSWORKS")
    print("="*70)
    
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        df = feature_group.read()
        
        # Convert timestamp back to datetime
        if 'datetime_utc' in df.columns and df['datetime_utc'].dtype == 'int64':
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', errors='coerce')
            
            # Drop invalid timestamps
            invalid_mask = df['datetime_utc'].isna() | (df['datetime_utc'] < pd.Timestamp('1678-01-01')) | (df['datetime_utc'] > pd.Timestamp('2262-04-11'))
            if invalid_mask.sum() > 0:
                print(f"⚠️ Dropped {invalid_mask.sum()} rows with invalid timestamps.")
                df = df[~invalid_mask].reset_index(drop=True)
            
            # Fix int dtypes
            int_cols = ['us_aqi', 'hour', 'month', 'day_of_week', 'is_weekend']
            for col in int_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
        
        # Drop 'id' if present
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        print(f"✅ Fetched from Hopsworks! Shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"❌ Hopsworks fetch failed: {e}. Falling back to local CSV.")
        try:
            df = pd.read_csv('data/2years_features.csv', parse_dates=['datetime_utc'])
            print(f"✅ Fallback to local CSV! Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise ValueError("No local CSV found—ensure upload happened first.")

# PyTorch Models
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def train_all_models(df_feat):
    """Train all 7 models"""
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load selected features
    with open('data/selected_features.txt', 'r') as f:
        selected_features = f.read().strip().split(',')
    
    print(f"✅ Using {len(selected_features)} features")
    
    # Prepare X/y
    exclude = ['us_aqi', 'aqi_category', 'id', 'datetime_utc', 'aqi']
    feature_cols = [col for col in df_feat.columns if col not in exclude]
    
    X = df_feat[selected_features].fillna(0).values
    y = df_feat['us_aqi'].values
    
    # Chronological split: 70% train, 10% val, 20% test
    split_train = int(0.7 * len(X))
    split_val = int(0.8 * len(X))
    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]
    
    print(f"📊 Split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {}
    
    # MODEL 1: Linear Regression
    print("\n1️⃣ Linear Regression...")
    X_train_sm = sm.add_constant(X_train_scaled)
    model1 = sm.OLS(y_train, X_train_sm).fit()
    X_test_sm = sm.add_constant(X_test_scaled)
    y_test_pred1 = model1.predict(X_test_sm)
    mae1, rmse1, r21, mape1 = calc_metrics(y_test, y_test_pred1)
    print(f" Test: MAE={mae1:.2f}, RMSE={rmse1:.2f}, R²={r21:.3f}, MAPE={mape1:.2f}%")
    results['linear'] = {'model': model1, 'name': 'Linear Regression', 'test_r2': r21, 'test_rmse': rmse1, 'test_mae': mae1, 'test_mape': mape1}
    with open('models/linear_model.pkl', 'wb') as f:
        pickle.dump(model1, f)
    
    # MODEL 2: Polynomial Regression
    print("\n2️⃣ Polynomial Regression...")
    pm2_idx = selected_features.index('pm2_5') if 'pm2_5' in selected_features else 0
    p_coeffs = np.polyfit(X_train_scaled[:, pm2_idx], y_train, 2)
    y_test_pred2 = np.polyval(p_coeffs, X_test_scaled[:, pm2_idx])
    mae2, rmse2, r22, mape2 = calc_metrics(y_test, y_test_pred2)
    print(f" Test: MAE={mae2:.2f}, RMSE={rmse2:.2f}, R²={r22:.3f}, MAPE={mape2:.2f}%")
    results['poly'] = {'model': p_coeffs, 'name': 'Polynomial Regression', 'test_r2': r22, 'test_rmse': rmse2, 'test_mae': mae2, 'test_mape': mape2}
    np.save('models/poly_coeffs.npy', p_coeffs)
    
    # MODEL 3: PyTorch Linear
    print("\n3️⃣ PyTorch Linear...")
    device = torch.device('cpu')
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    model3 = LinearModel(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model3.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(100):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model3(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model3.eval()
    with torch.no_grad():
        y_test_pred3 = model3(X_test_t).cpu().numpy().flatten()
    mae3, rmse3, r23, mape3 = calc_metrics(y_test, y_test_pred3)
    print(f" Test: MAE={mae3:.2f}, RMSE={rmse3:.2f}, R²={r23:.3f}, MAPE={mape3:.2f}%")
    results['pytorch_linear'] = {'model': model3, 'name': 'PyTorch Linear', 'test_r2': r23, 'test_rmse': rmse3, 'test_mae': mae3, 'test_mape': mape3}
    torch.save(model3.state_dict(), 'models/linear_model.pth')
    
    # MODEL 4: PyTorch MLP
    print("\n4️⃣ PyTorch MLP...")
    model4 = MLP(X_train_scaled.shape[1]).to(device)
    optimizer = optim.Adam(model4.parameters(), lr=0.001)
    
    for epoch in range(100):
        model4.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model4(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model4.eval()
    with torch.no_grad():
        y_test_pred4 = model4(X_test_t).cpu().numpy().flatten()
    mae4, rmse4, r24, mape4 = calc_metrics(y_test, y_test_pred4)
    print(f" Test: MAE={mae4:.2f}, RMSE={rmse4:.2f}, R²={r24:.3f}, MAPE={mape4:.2f}%")
    results['mlp'] = {'model': model4, 'name': 'PyTorch MLP', 'test_r2': r24, 'test_rmse': rmse4, 'test_mae': mae4, 'test_mape': mape4}
    torch.save(model4.state_dict(), 'models/mlp_model.pth')
    
    # MODEL 5: Gradient Boosting
    print("\n5️⃣ Gradient Boosting...")
    model5 = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.7,
        min_samples_split=50, min_samples_leaf=20, random_state=42
    )
    model5.fit(X_train, y_train)
    y_test_pred5 = model5.predict(X_test)
    mae5, rmse5, r25, mape5 = calc_metrics(y_test, y_test_pred5)
    print(f" Test: MAE={mae5:.2f}, RMSE={rmse5:.2f}, R²={r25:.3f}, MAPE={mape5:.2f}%")
    results['gb'] = {'model': model5, 'name': 'Gradient Boosting', 'test_r2': r25, 'test_rmse': rmse5, 'test_mae': mae5, 'test_mape': mape5}
    joblib.dump(model5, 'models/gb_model.pkl')
    
    # MODEL 6: XGBoost
    print("\n6️⃣ XGBoost...")
    model6 = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbosity=0
    )
    model6.fit(X_train, y_train)
    y_test_pred6 = model6.predict(X_test)
    mae6, rmse6, r26, mape6 = calc_metrics(y_test, y_test_pred6)
    print(f" Test: MAE={mae6:.2f}, RMSE={rmse6:.2f}, R²={r26:.3f}, MAPE={mape6:.2f}%")
    results['xgb'] = {'model': model6, 'name': 'XGBoost', 'test_r2': r26, 'test_rmse': rmse6, 'test_mae': mae6, 'test_mape': mape6}
    model6.save_model('models/xgb_model.json')
    
    # MODEL 7: Random Forest
    print("\n7️⃣ Random Forest...")
    model7 = RandomForestRegressor(
        n_estimators=50, max_depth=8, min_samples_split=50, min_samples_leaf=20,
        max_features=0.5, random_state=42
    )
    model7.fit(X_train, y_train)
    y_test_pred7 = model7.predict(X_test)
    mae7, rmse7, r27, mape7 = calc_metrics(y_test, y_test_pred7)
    print(f" Test: MAE={mae7:.2f}, RMSE={rmse7:.2f}, R²={r27:.3f}, MAPE={mape7:.2f}%")
    results['rf'] = {'model': model7, 'name': 'Random Forest', 'test_r2': r27, 'test_rmse': rmse7, 'test_mae': mae7, 'test_mape': mape7}
    joblib.dump(model7, 'models/rf_model.pkl')
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    results_summary = pd.DataFrame([
        {'Model': res['name'], 'Test MAE': res['test_mae'], 'Test RMSE': res['test_rmse'],
         'Test R²': res['test_r2'], 'Test MAPE (%)': res['test_mape']}
        for res in results.values()
    ]).sort_values('Test R²', ascending=False)
    print(results_summary.to_string(index=False, float_format='%.3f'))
    results_summary.to_csv('data/training_results.csv', index=False)
    
    return results, selected_features

def predict_future_aqi(selected_features):
    """Predict future AQI using forecast pollutants"""
    print("\n" + "="*70)
    print("FUTURE AQI PREDICTION")
    print("="*70)
    
    API_KEY = os.getenv('OWM_API_KEY', '29e4f8ef9151633260fb36745ed19012')
    LAT = 24.8607
    LON = 67.0011
    
    # Fetch forecast data
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={API_KEY}"
    response = requests.get(url).json()
    
    # Import calc_us_aqi from eda
    from eda import engineer_features
    
    # Load cleaned data for median/percentiles
    df_clean = pd.read_csv('data/2years_cleaned.csv', parse_dates=['datetime_utc'])
    
    # Process forecast data (simplified US AQI calculation)
    records = []
    for item in response["list"]:
        dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        comp = item["components"]
        aqi_api = item["main"]["aqi"]
        
        # Simple US AQI approximation (use API AQI * 50 as proxy)
        usaqi = np.clip(aqi_api * 50, 0, 500)
        
        records.append({
            "datetime_utc": dt,
            "Actual_AQI": usaqi,
            **comp
        })
    
    forecast_df = pd.DataFrame(records)
    print(f"✅ Forecast data: {forecast_df.shape[0]} hours")
    
    # Clean and engineer features
    pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    for col in pollutants:
        forecast_df[col] = forecast_df[col].clip(lower=0)
        forecast_df[col] = forecast_df[col].fillna(df_clean[col].median())
    
    forecast_feat = engineer_features(forecast_df)
    X_future = forecast_feat[selected_features].fillna(0)
    
    # Scale
    scaler = joblib.load('models/scaler.pkl')
    X_future_scaled = scaler.transform(X_future)
    
    # Predictions
    pred = pd.DataFrame({
        "datetime_utc": forecast_df["datetime_utc"],
        "Actual_AQI": forecast_df["Actual_AQI"].astype(float)
    })
    
    # Load and predict with all models
    # 1. Linear
    model1 = pickle.load(open('models/linear_model.pkl', 'rb'))
    X_sm = sm.add_constant(X_future_scaled, has_constant='add')
    pred["Linear"] = model1.predict(X_sm)
    
    # 2. Polynomial
    coeffs = np.load('models/poly_coeffs.npy')
    pm2_idx = selected_features.index('pm2_5')
    pred["Polynomial"] = np.polyval(coeffs, X_future_scaled[:, pm2_idx])
    
    # 3. PyTorch Linear
    m3 = LinearModel(len(selected_features))
    m3.load_state_dict(torch.load("models/linear_model.pth"))
    m3.eval()
    pred["PyTorch_Linear"] = m3(torch.tensor(X_future_scaled, dtype=torch.float32)).detach().numpy().flatten()
    
    # 4. PyTorch MLP
    m4 = MLP(len(selected_features))
    m4.load_state_dict(torch.load("models/mlp_model.pth"))
    m4.eval()
    pred["PyTorch_MLP"] = m4(torch.tensor(X_future_scaled, dtype=torch.float32)).detach().numpy().flatten()
    
    # 5. Gradient Boosting
    gb = joblib.load("models/gb_model.pkl")
    pred["GB"] = gb.predict(X_future)
    
    # 6. XGBoost
    xg = xgb.XGBRegressor()
    xg.load_model("models/xgb_model.json")
    pred["XGB"] = xg.predict(X_future)
    
    # 7. Random Forest
    rf = joblib.load("models/rf_model.pkl")
    pred["RF"] = rf.predict(X_future)
    
    # Ensemble
    pred["Ensemble"] = pred[['Linear', 'Polynomial', 'PyTorch_Linear', 'PyTorch_MLP', 'GB', 'XGB', 'RF']].mean(axis=1)
    
    # Clip to valid range
    for col in pred.columns[2:]:
        pred[col] = np.clip(pred[col], 0, 500)
    
    # Identify best model per timestamp
    model_cols = pred.columns.drop(["datetime_utc", "Actual_AQI"])
    pred["Closest_Model"] = pred[model_cols].sub(pred["Actual_AQI"], axis=0).abs().idxmin(axis=1)
    
    # Save
    pred.to_csv("data/future_aqi_predictions.csv", index=False)
    print("💾 Saved: data/future_aqi_predictions.csv")
    
    # Model comparison
    results = []
    for model in model_cols:
        mae = mean_absolute_error(pred["Actual_AQI"], pred[model])
        rmse = np.sqrt(mean_squared_error(pred["Actual_AQI"], pred[model]))
        r2 = r2_score(pred["Actual_AQI"], pred[model])
        results.append([model, round(mae, 2), round(rmse, 2), round(r2, 4)])
    
    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²"])
    results_df = results_df.sort_values(by="RMSE")
    print("\n📊 Model Performance on Future Predictions:")
    print(results_df.to_string(index=False))
    results_df.to_csv('data/future_prediction_comparison.csv', index=False)

def main():
    """Main execution function"""
    # Fetch data from Hopsworks
    df_feat = fetch_from_hopsworks()
    
    # Train all models
    results, selected_features = train_all_models(df_feat)
    
    # Predict future AQI
    predict_future_aqi(selected_features)
    
    print("\n✅ Training and prediction complete!")

if __name__ == "__main__":
    main()