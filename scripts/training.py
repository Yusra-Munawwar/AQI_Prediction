"""
Complete AQI Prediction Model Training Pipeline
Includes data fetching, preprocessing, model training, evaluation, and future forecasting
FIXED: Consistent feature engineering for training and forecast data
"""
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
import warnings
import json
from datetime import datetime, timezone
import requests

warnings.filterwarnings('ignore')

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

def engineer_features(df, is_forecast=False):
    """Add temporal and derived features consistently for training and forecast."""
    df = df.copy()
    
    # Temporal features
    df['hour'] = df['datetime_utc'].dt.hour
    df['month'] = df['datetime_utc'].dt.month
    df['day_of_week'] = df['datetime_utc'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encodings
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Aggregations
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['total_pm'] = df['pm2_5'] + df['pm10']
    if all(col in df.columns for col in ['co', 'no2', 'o3', 'so2']):
        df['total_gases'] = df['co'] + df['no2'] + df['o3'] + df['so2']
    
    # Interactions
    if 'pm2_5' in df.columns and 'co' in df.columns:
        df['pm2_5_co_interaction'] = df['pm2_5'] * df['co']
    if 'no2' in df.columns and 'o3' in df.columns:
        # Avoid division by zero
        df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
    
    # Rolling averages (only for historical data with sufficient history)
    if not is_forecast and len(df) > 3:
        for col in ['pm2_5', 'pm10', 'co']:
            if col in df.columns:
                df[f'{col}_rolling_3h'] = df[col].rolling(window=3, min_periods=1).mean()
    else:
        # For forecast, use current values as proxy (no historical data available)
        for col in ['pm2_5', 'pm10', 'co']:
            if col in df.columns:
                df[f'{col}_rolling_3h'] = df[col]
    
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

# =============================================================================
# SECTION 1: FETCH DATA FROM HOPSWORKS
# =============================================================================
def fetch_from_hopsworks(feature_group_name="aqi_features", version=1, for_training=False):
    """
    Fetch data from Hopsworks (fallback to local). Handles invalid timestamps and nullable dtypes.
    """
    print("\n" + "="*70)
    print(f"{'FETCHING FROM HOPSWORKS FOR VERIFICATION' if not for_training else 'SECTION 1: FETCHING FEATURES FROM HOPSWORKS'}")
    print("="*70)
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        df = feature_group.read()
        # Convert timestamp back to datetime
        if 'datetime_utc' in df.columns and df['datetime_utc'].dtype == 'int64':
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', errors='coerce')
            # Drop rows with invalid timestamps
            invalid_mask = df['datetime_utc'].isna() | (df['datetime_utc'] < pd.Timestamp('1678-01-01')) | (df['datetime_utc'] > pd.Timestamp('2262-04-11'))
            if invalid_mask.sum() > 0:
                print(f"⚠️ Dropped {invalid_mask.sum()} rows with invalid timestamps")
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
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Hopsworks fetch failed: {e}. Falling back to local CSV.")
        try:
            df = pd.read_csv('2years_features.csv', parse_dates=['datetime_utc'])
            print(f"✅ Fallback to local CSV! Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise ValueError("No local CSV found—ensure upload happened first.")

# =============================================================================
# SECTION 2: DATA PREPARATION & SPLITTING
# =============================================================================
def prepare_data(df_feat):
    """
    Prepare features and split data chronologically
    """
    print("\n" + "="*70)
    print("SECTION 2: DATA PREPARATION & SPLITTING")
    print("="*70)
    # Select features based on correlation (exclude leakage columns)
    exclude = ['us_aqi', 'aqi_category', 'id', 'datetime_utc', 'aqi']
    feature_cols = [col for col in df_feat.columns if col not in exclude]
   
    # Calculate correlation with target
    corr = df_feat[feature_cols + ['us_aqi']].corr()['us_aqi'].sort_values(ascending=False)
    selected_features = [f for f in corr.index[1:] if abs(corr[f]) > 0.3]
   
    print(f"✅ Using {len(selected_features)} features: {selected_features}")
    # Prepare X/y
    X = df_feat[selected_features].fillna(0).values
    y = df_feat['us_aqi'].values
    # Chronological split: 70% train, 10% val, 20% test
    split_train = int(0.7 * len(X))
    split_val = int(0.8 * len(X))
    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]
    print(f"📊 Split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    # Scaling (for non-tree models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
   
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Scaling complete!")
    # Save selected features
    with open('selected_features.json', 'w') as f:
        json.dump(selected_features, f)
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            X_train_scaled, X_val_scaled, X_test_scaled, selected_features)

# =============================================================================
# CHECKPOINT MANAGEMENT FOR CONTINUOUS TRAINING
# =============================================================================
def save_checkpoint(model, model_name, epoch, val_loss, is_best=False):
    """
    Save model checkpoint with timestamp for continuous training resilience.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/{model_name}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save current checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, f"{checkpoint_dir}/checkpoint.pth")
    
    if is_best:
        # Update best checkpoint symlink or copy
        best_path = f"checkpoints/{model_name}_best.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, best_path)
        print(f"💾 Saved best checkpoint for {model_name} at epoch {epoch} (val_loss: {val_loss:.4f})")
    
    print(f"💾 Saved checkpoint for {model_name} at epoch {epoch} (val_loss: {val_loss:.4f})")

def load_best_checkpoint(model_class, model_name, input_dim):
    """
    Load the best checkpoint for a PyTorch model.
    """
    best_path = f"checkpoints/{model_name}_best.pth"
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu')
        model = model_class(input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best checkpoint for {model_name} (epoch: {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")
        return model
    else:
        print(f"⚠️ No best checkpoint found for {model_name}, using fresh model.")
        return None

def compare_and_save_best_results(results_summary, prev_path='previous_best_results.csv', current_path='data/training_results.csv'):
    """
    Compare new training results with previous best to prevent degradation in continuous training.
    Save the better one (higher max Test R²) to the current path.
    """
    new_best_r2 = results_summary['Test R²'].max()
    
    if os.path.exists(prev_path):
        prev_results = pd.read_csv(prev_path)
        old_best_r2 = prev_results['Test R²'].max()
        print(f"📊 Comparing: New best R²={new_best_r2:.3f} vs Previous best R²={old_best_r2:.3f}")
        
        if new_best_r2 > old_best_r2:
            print("✅ New results are better! Saving new results and models.")
            results_summary.to_csv(current_path, index=False)
            # Optionally, copy current model files to a 'best_models' dir
            best_models_dir = 'best_models'
            os.makedirs(best_models_dir, exist_ok=True)
            model_files = ['linear_model.pkl', 'poly_coeffs.npy', 'linear_model.pth', 'mlp_model.pth',
                           'gb_model.pkl', 'xgb_model.json', 'rf_model.pkl', 'scaler.pkl']
            for file in model_files:
                if os.path.exists(file):
                    import shutil
                    shutil.copy(file, best_models_dir)
            # Update previous
            results_summary.to_csv(prev_path, index=False)
            return True  # New is best
        else:
            print("🔄 Keeping previous best results to prevent degradation.")
            prev_results.to_csv(current_path, index=False)
            # Optionally, restore model files from best_models if needed, but assume they are already there
            return False  # Old is best
    else:
        print("✅ No previous results found. Saving current as best.")
        results_summary.to_csv(current_path, index=False)
        results_summary.to_csv(prev_path, index=False)
        return True

# =============================================================================
# SECTION 3: MODEL TRAINING
# =============================================================================
def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test,
                     X_train_scaled, X_val_scaled, X_test_scaled, selected_features):
    """
    Train all 7 models with regularization and checkpointing for PyTorch models.
    """
    print("\n" + "="*70)
    print("SECTION 3: MODEL TRAINING (7 MODELS WITH REGULARIZATION & CHECKPOINTING)")
    print("="*70)
   
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)
   
    # ==================== MODEL 1: Linear Regression ====================
    print("\n1️⃣ Linear Regression (statsmodels)...")
    X_train_sm = sm.add_constant(X_train_scaled)
    model1 = sm.OLS(y_train, X_train_sm).fit()
    X_val_sm = sm.add_constant(X_val_scaled)
    X_test_sm = sm.add_constant(X_test_scaled)
   
    y_train_pred1 = model1.predict(X_train_sm)
    y_val_pred1 = model1.predict(X_val_sm)
    y_test_pred1 = model1.predict(X_test_sm)
   
    mae1_t, rmse1_t, r21_t, mape1_t = calc_metrics(y_train, y_train_pred1)
    mae1_v, rmse1_v, r21_v, mape1_v = calc_metrics(y_val, y_val_pred1)
    mae1, rmse1, r21, mape1 = calc_metrics(y_test, y_test_pred1)
   
    print(f" Train: MAE={mae1_t:.2f}, RMSE={rmse1_t:.2f}, R²={r21_t:.3f}, MAPE={mape1_t:.2f}%")
    print(f" Val: MAE={mae1_v:.2f}, RMSE={rmse1_v:.2f}, R²={r21_v:.3f}, MAPE={mape1_v:.2f}%")
    print(f" Test: MAE={mae1:.2f}, RMSE={rmse1:.2f}, R²={r21:.3f}, MAPE={mape1:.2f}% | Gap: {r21_t - r21:.3f}")
   
    # CV R²
    cv_scores1 = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_fold_train = sm.add_constant(X_train_scaled[train_idx])
        fold_model = sm.OLS(y_train[train_idx], X_fold_train).fit()
        X_fold_val = sm.add_constant(X_train_scaled[val_idx])
        fold_pred = fold_model.predict(X_fold_val)
        cv_scores1.append(r2_score(y_train[val_idx], fold_pred))
    cv_r2_1 = np.mean(cv_scores1)
    print(f" CV R²: {cv_r2_1:.3f}")
   
    results['linear'] = {
        'model': model1, 'name': 'Linear Regression',
        'test_r2': r21, 'test_rmse': rmse1, 'test_mae': mae1, 'test_mape': mape1, 'cv_r2': cv_r2_1
    }
    with open('linear_model.pkl', 'wb') as f:
        pickle.dump(model1, f)
   
    # ==================== MODEL 2: Polynomial Regression ====================
    print("\n2️⃣ Polynomial Regression (deg 2 on PM2.5)...")
    pm2_idx = selected_features.index('pm2_5') if 'pm2_5' in selected_features else 0
    p_coeffs = np.polyfit(X_train_scaled[:, pm2_idx], y_train, 2)
   
    y_train_pred2 = np.polyval(p_coeffs, X_train_scaled[:, pm2_idx])
    y_val_pred2 = np.polyval(p_coeffs, X_val_scaled[:, pm2_idx])
    y_test_pred2 = np.polyval(p_coeffs, X_test_scaled[:, pm2_idx])
   
    mae2_t, rmse2_t, r22_t, mape2_t = calc_metrics(y_train, y_train_pred2)
    mae2_v, rmse2_v, r22_v, mape2_v = calc_metrics(y_val, y_val_pred2)
    mae2, rmse2, r22, mape2 = calc_metrics(y_test, y_test_pred2)
   
    print(f" Train: MAE={mae2_t:.2f}, RMSE={rmse2_t:.2f}, R²={r22_t:.3f}, MAPE={mape2_t:.2f}%")
    print(f" Val: MAE={mae2_v:.2f}, RMSE={rmse2_v:.2f}, R²={r22_v:.3f}, MAPE={mape2_v:.2f}%")
    print(f" Test: MAE={mae2:.2f}, RMSE={rmse2:.2f}, R²={r22:.3f}, MAPE={mape2:.2f}% | Gap: {r22_t - r22:.3f}")
   
    results['poly'] = {
        'model': p_coeffs, 'name': 'Polynomial Regression',
        'test_r2': r22, 'test_rmse': rmse2, 'test_mae': mae2, 'test_mape': mape2
    }
    np.save('poly_coeffs.npy', p_coeffs)
   
    # ==================== MODEL 3: PyTorch Linear ====================
    print("\n3️⃣ PyTorch Linear Regressor (with checkpointing)...")
   
    class LinearModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        def forward(self, x):
            return self.linear(x)
   
    device = torch.device('cpu')
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
   
    # Try to load best checkpoint
    model3 = load_best_checkpoint(LinearModel, 'linear', X_train_scaled.shape[1])
    if model3 is None:
        model3 = LinearModel(X_train_scaled.shape[1]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model3.parameters(), lr=0.01)
   
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
   
    best_val_loss = float('inf')
    patience = 10  # For early stopping
    patience_counter = 0
   
    for epoch in range(100):
        model3.train()
        train_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model3(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loss
        model3.eval()
        with torch.no_grad():
            val_outputs = model3(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        avg_train_loss = train_loss / len(loader)
        print(f"Epoch {epoch+1}/100 - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Checkpointing
        save_checkpoint(model3, 'linear', epoch+1, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model3, 'linear', epoch+1, val_loss, is_best=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
   
    # Load best checkpoint for final evaluation
    best_model3 = load_best_checkpoint(LinearModel, 'linear', X_train_scaled.shape[1])
    if best_model3:
        model3 = best_model3
   
    model3.eval()
    with torch.no_grad():
        y_train_pred3 = model3(X_train_t).cpu().numpy().flatten()
        y_val_pred3 = model3(X_val_t).cpu().numpy().flatten()
        y_test_pred3 = model3(X_test_t).cpu().numpy().flatten()
   
    mae3_t, rmse3_t, r23_t, mape3_t = calc_metrics(y_train, y_train_pred3)
    mae3_v, rmse3_v, r23_v, mape3_v = calc_metrics(y_val, y_val_pred3)
    mae3, rmse3, r23, mape3 = calc_metrics(y_test, y_test_pred3)
   
    print(f" Train: MAE={mae3_t:.2f}, RMSE={rmse3_t:.2f}, R²={r23_t:.3f}, MAPE={mape3_t:.2f}%")
    print(f" Val: MAE={mae3_v:.2f}, RMSE={rmse3_v:.2f}, R²={r23_v:.3f}, MAPE={mape3_v:.2f}%")
    print(f" Test: MAE={mae3:.2f}, RMSE={rmse3:.2f}, R²={r23:.3f}, MAPE={mape3:.2f}% | Gap: {r23_t - r23:.3f}")
   
    results['pytorch_linear'] = {
        'model': model3, 'name': 'PyTorch Linear',
        'test_r2': r23, 'test_rmse': rmse3, 'test_mae': mae3, 'test_mape': mape3
    }
    torch.save(model3.state_dict(), 'linear_model.pth')
   
    # ==================== MODEL 4: PyTorch MLP ====================
    print("\n4️⃣ PyTorch MLP (increased dropout, with checkpointing)...")
   
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
   
    # Try to load best checkpoint
    model4 = load_best_checkpoint(MLP, 'mlp', X_train_scaled.shape[1])
    if model4 is None:
        model4 = MLP(X_train_scaled.shape[1]).to(device)
    
    optimizer = optim.Adam(model4.parameters(), lr=0.001)
   
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
   
    for epoch in range(100):
        model4.train()
        train_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model4(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loss
        model4.eval()
        with torch.no_grad():
            val_outputs = model4(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        avg_train_loss = train_loss / len(loader)
        print(f"Epoch {epoch+1}/100 - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Checkpointing
        save_checkpoint(model4, 'mlp', epoch+1, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model4, 'mlp', epoch+1, val_loss, is_best=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
   
    # Load best checkpoint for final evaluation
    best_model4 = load_best_checkpoint(MLP, 'mlp', X_train_scaled.shape[1])
    if best_model4:
        model4 = best_model4
   
    model4.eval()
    with torch.no_grad():
        y_train_pred4 = model4(X_train_t).cpu().numpy().flatten()
        y_val_pred4 = model4(X_val_t).cpu().numpy().flatten()
        y_test_pred4 = model4(X_test_t).cpu().numpy().flatten()
   
    mae4_t, rmse4_t, r24_t, mape4_t = calc_metrics(y_train, y_train_pred4)
    mae4_v, rmse4_v, r24_v, mape4_v = calc_metrics(y_val, y_val_pred4)
    mae4, rmse4, r24, mape4 = calc_metrics(y_test, y_test_pred4)
   
    print(f" Train: MAE={mae4_t:.2f}, RMSE={rmse4_t:.2f}, R²={r24_t:.3f}, MAPE={mape4_t:.2f}%")
    print(f" Val: MAE={mae4_v:.2f}, RMSE={rmse4_v:.2f}, R²={r24_v:.3f}, MAPE={mape4_v:.2f}%")
    print(f" Test: MAE={mae4:.2f}, RMSE={rmse4:.2f}, R²={r24:.3f}, MAPE={mape4:.2f}% | Gap: {r24_t - r24:.3f}")
   
    results['mlp'] = {
        'model': model4, 'name': 'PyTorch MLP',
        'test_r2': r24, 'test_rmse': rmse4, 'test_mae': mae4, 'test_mape': mape4
    }
    torch.save(model4.state_dict(), 'mlp_model.pth')
   
    # ==================== MODEL 5: Gradient Boosting ====================
    print("\n5️⃣ Gradient Boosting (stronger reg)...")
    model5 = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.7,
        min_samples_split=50, min_samples_leaf=20, random_state=42
    )
    model5.fit(X_train, y_train)
   
    y_train_pred5 = model5.predict(X_train)
    y_val_pred5 = model5.predict(X_val)
    y_test_pred5 = model5.predict(X_test)
   
    mae5_t, rmse5_t, r25_t, mape5_t = calc_metrics(y_train, y_train_pred5)
    mae5_v, rmse5_v, r25_v, mape5_v = calc_metrics(y_val, y_val_pred5)
    mae5, rmse5, r25, mape5 = calc_metrics(y_test, y_test_pred5)
   
    print(f" Train: MAE={mae5_t:.2f}, RMSE={rmse5_t:.2f}, R²={r25_t:.3f}, MAPE={mape5_t:.2f}%")
    print(f" Val: MAE={mae5_v:.2f}, RMSE={rmse5_v:.2f}, R²={r25_v:.3f}, MAPE={mape5_v:.2f}%")
    print(f" Test: MAE={mae5:.2f}, RMSE={rmse5:.2f}, R²={r25:.3f}, MAPE={mape5:.2f}% | Gap: {r25_t - r25:.3f}")
   
    # CV
    cv_scores5 = []
    for train_idx, val_idx in tscv.split(X_train):
        fold_model = GradientBoostingRegressor(**model5.get_params())
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        fold_pred = fold_model.predict(X_train[val_idx])
        cv_scores5.append(r2_score(y_train[val_idx], fold_pred))
    cv_r2_5 = np.mean(cv_scores5)
    print(f" CV R²: {cv_r2_5:.3f}")
   
    results['gb'] = {
        'model': model5, 'name': 'Gradient Boosting',
        'test_r2': r25, 'test_rmse': rmse5, 'test_mae': mae5, 'test_mape': mape5, 'cv_r2': cv_r2_5
    }
    joblib.dump(model5, 'gb_model.pkl')
   
    # ==================== MODEL 6: XGBoost ====================
    print("\n6️⃣ XGBoost (higher reg)...")
    model6 = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbosity=0
    )
    model6.fit(X_train, y_train)
   
    y_train_pred6 = model6.predict(X_train)
    y_val_pred6 = model6.predict(X_val)
    y_test_pred6 = model6.predict(X_test)
   
    mae6_t, rmse6_t, r26_t, mape6_t = calc_metrics(y_train, y_train_pred6)
    mae6_v, rmse6_v, r26_v, mape6_v = calc_metrics(y_val, y_val_pred6)
    mae6, rmse6, r26, mape6 = calc_metrics(y_test, y_test_pred6)
   
    print(f" Train: MAE={mae6_t:.2f}, RMSE={rmse6_t:.2f}, R²={r26_t:.3f}, MAPE={mape6_t:.2f}%")
    print(f" Val: MAE={mae6_v:.2f}, RMSE={rmse6_v:.2f}, R²={r26_v:.3f}, MAPE={mape6_v:.2f}%")
    print(f" Test: MAE={mae6:.2f}, RMSE={rmse6:.2f}, R²={r26:.3f}, MAPE={mape6:.2f}% | Gap: {r26_t - r26:.3f}")
   
    cv_scores6 = []
    for train_idx, val_idx in tscv.split(X_train):
        fold_model = xgb.XGBRegressor(**model6.get_params())
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        fold_pred = fold_model.predict(X_train[val_idx])
        cv_scores6.append(r2_score(y_train[val_idx], fold_pred))
    cv_r2_6 = np.mean(cv_scores6)
    print(f" CV R²: {cv_r2_6:.3f}")
   
    results['xgb'] = {
        'model': model6, 'name': 'XGBoost',
        'test_r2': r26, 'test_rmse': rmse6, 'test_mae': mae6, 'test_mape': mape6, 'cv_r2': cv_r2_6
    }
    model6.save_model('xgb_model.json')
   
    # ==================== MODEL 7: Random Forest ====================
    print("\n7️⃣ Random Forest (higher mins)...")
    model7 = RandomForestRegressor(
        n_estimators=50, max_depth=8, min_samples_split=50, min_samples_leaf=20,
        max_features=0.5, random_state=42
    )
    model7.fit(X_train, y_train)
   
    y_train_pred7 = model7.predict(X_train)
    y_val_pred7 = model7.predict(X_val)
    y_test_pred7 = model7.predict(X_test)
   
    mae7_t, rmse7_t, r27_t, mape7_t = calc_metrics(y_train, y_train_pred7)
    mae7_v, rmse7_v, r27_v, mape7_v = calc_metrics(y_val, y_val_pred7)
    mae7, rmse7, r27, mape7 = calc_metrics(y_test, y_test_pred7)
   
    print(f" Train: MAE={mae7_t:.2f}, RMSE={rmse7_t:.2f}, R²={r27_t:.3f}, MAPE={mape7_t:.2f}%")
    print(f" Val: MAE={mae7_v:.2f}, RMSE={rmse7_v:.2f}, R²={r27_v:.3f}, MAPE={mape7_v:.2f}%")
    print(f" Test: MAE={mae7:.2f}, RMSE={rmse7:.2f}, R²={r27:.3f}, MAPE={mape7:.2f}% | Gap: {r27_t - r27:.3f}")
   
    cv_scores7 = []
    for train_idx, val_idx in tscv.split(X_train):
        fold_model = RandomForestRegressor(**model7.get_params())
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        fold_pred = fold_model.predict(X_train[val_idx])
        cv_scores7.append(r2_score(y_train[val_idx], fold_pred))
    cv_r2_7 = np.mean(cv_scores7)
    print(f" CV R²: {cv_r2_7:.3f}")
   
    results['rf'] = {
        'model': model7, 'name': 'Random Forest',
        'test_r2': r27, 'test_rmse': rmse7, 'test_mae': mae7, 'test_mape': mape7, 'cv_r2': cv_r2_7
    }
    joblib.dump(model7, 'rf_model.pkl')
   
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY (Test Metrics)")
    print("="*70)
    results_summary = pd.DataFrame([
        {'Model': res['name'], 'Test MAE': res['test_mae'], 'Test RMSE': res['test_rmse'],
         'Test R²': res['test_r2'], 'Test MAPE (%)': res['test_mape'], 'CV R²': res.get('cv_r2', np.nan)}
        for res in results.values()
    ]).sort_values('Test R²', ascending=False)
   
    print(results_summary.to_string(index=False, float_format='%.3f'))
    
    # Compare and save best results for continuous training
    is_new_best = compare_and_save_best_results(results_summary)
    if not is_new_best:
        print("📝 Results summary not updated in training_results.csv (keeping previous best).")
    else:
        print("\n✅ Training complete! Checkpoints saved in 'checkpoints/' directory.")
   
    return results

# =============================================================================
# SECTION 4: FUTURE AQI PREDICTION
# =============================================================================
def generate_future_predictions(df_feat, selected_features):
    """
    Generate future AQI predictions using trained models and save CSVs.
    """
    print("\n" + "="*70)
    print("SECTION 4: FUTURE AQI PREDICTION USING FORECAST POLLUTANTS")
    print("="*70)
    
    try:
        # Verify inputs
        print(f"📊 Historical data shape: {df_feat.shape}")
        print(f"📊 Selected features ({len(selected_features)}): {selected_features}")
        
        os.makedirs('data', exist_ok=True)
        df_clean = df_feat.copy()  # Historical for cleaning (make a copy to be safe)
        
        # Verify required columns exist in historical data
        required_pollutants = ['co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10']
        missing_from_history = [col for col in required_pollutants if col not in df_clean.columns]
        if missing_from_history:
            print(f"⚠️ Warning: Historical data missing columns: {missing_from_history}")
        
        # Fetch forecast pollutants
        def fetch_forecast_pollutants(lat, lon, api_key):
            try:
                url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"
                print(f"🌐 Fetching from: {url}")
                r = requests.get(url, timeout=30)
                r.raise_for_status()  # Raise error for bad status codes
                data = r.json()
                
                if "list" not in data:
                    raise ValueError(f"Unexpected API response format: {data}")
                
                records = []
                for item in data["list"]:
                    dt = datetime.fromtimestamp(item["dt"], tz=UTC)
                    comp = item["components"]
                    aqi_1_5 = item["main"]["aqi"]
                    temp_row = {"datetime_utc": dt, **comp}
                    usaqi = calc_us_aqi(pd.Series(temp_row))
                    usaqi = np.clip(usaqi, 0, 500)
                    records.append({
                        "datetime_utc": dt,
                        "aqi_api": aqi_1_5,
                        "Actual_AQI": usaqi,  # For comparison (forecast-based "actual")
                        **comp
                    })
                return pd.DataFrame(records)
            except requests.exceptions.RequestException as e:
                print(f"❌ API request failed: {e}")
                raise
            except Exception as e:
                print(f"❌ Error parsing API response: {e}")
                raise
        
        api_key = os.getenv("OWM_API_KEY")
        if not api_key:
            print("⚠️ OWM_API_KEY not set. Skipping forecast generation.")
            return
        
        print("🌐 Fetching forecast data from OpenWeatherMap API...")
        forecast_df = fetch_forecast_pollutants(LAT, LON, api_key)
        print(f"✅ Forecast pollutant data received: {forecast_df.shape[0]} hours")
        print(f"📊 Forecast columns: {list(forecast_df.columns)}")
        
        # Check for NaN/inf values in forecast
        if forecast_df.isnull().any().any():
            print(f"⚠️ Found NaN values in forecast data:")
            print(forecast_df.isnull().sum()[forecast_df.isnull().sum() > 0])
        
        # Clean & Feature Engineering with is_forecast=True flag
        print("🔧 Cleaning forecast data...")
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        for col in pollutants:
            if col in forecast_df.columns:
                if col in df_clean.columns:
                    median_val = df_clean[col].median()
                    quantile_val = df_clean[col].quantile(0.995)
                    print(f"  {col}: median={median_val:.2f}, 99.5%={quantile_val:.2f}")
                    forecast_df[col] = forecast_df[col].clip(lower=0)
                    forecast_df[col] = forecast_df[col].fillna(median_val)
                    forecast_df[col] = forecast_df[col].clip(upper=quantile_val)
                else:
                    print(f"  ⚠️ {col}: not in historical data, using fallback")
                    forecast_df[col] = forecast_df[col].clip(lower=0).fillna(0)
        
        print("🔧 Engineering features for forecast data...")
        forecast_feat = engineer_features(forecast_df, is_forecast=True)
        print(f"✅ Feature engineering complete. Shape: {forecast_feat.shape}")
        print(f"📊 Engineered columns: {list(forecast_feat.columns)}")
        
        # Ensure all selected features exist, fill missing with 0
        print(f"🔍 Checking for {len(selected_features)} selected features...")
        missing_features = []
        for feature in selected_features:
            if feature not in forecast_feat.columns:
                print(f"⚠️ Missing feature '{feature}' in forecast data, filling with 0")
                forecast_feat[feature] = 0
                missing_features.append(feature)
        
        if missing_features:
            print(f"⚠️ Total missing features: {len(missing_features)}")
        else:
            print("✅ All selected features present in forecast data")
        
        print("🔄 Preparing feature matrix...")
        X_future = forecast_feat[selected_features].fillna(0).values
        print(f"✅ X_future shape: {X_future.shape}")
        
        # Check for NaN/inf in feature matrix
        if np.isnan(X_future).any():
            nan_count = np.isnan(X_future).sum()
            print(f"⚠️ Warning: {nan_count} NaN values in X_future, replacing with 0")
            X_future = np.nan_to_num(X_future, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("📏 Loading scaler and transforming data...")
        if not os.path.exists('scaler.pkl'):
            raise FileNotFoundError("scaler.pkl not found! Training may have failed.")
        
        scaler = joblib.load('scaler.pkl')
        X_future_scaled = scaler.transform(X_future)
        print(f"✅ X_future_scaled shape: {X_future_scaled.shape}")
        
        # Predictions
        print("🤖 Generating predictions from all models...")
        pred = pd.DataFrame({
            "datetime_utc": forecast_df["datetime_utc"],
            "Actual_AQI": forecast_df["Actual_AQI"].astype(float)
        })
        
        # 1. Linear Regression
        print("  1️⃣ Linear Regression...")
        if not os.path.exists('linear_model.pkl'):
            raise FileNotFoundError("linear_model.pkl not found!")
        model1 = pickle.load(open('linear_model.pkl', 'rb'))
        X_sm = sm.add_constant(X_future_scaled)
        pred["Linear"] = model1.predict(X_sm)
        
        # 2. Polynomial
        print("  2️⃣ Polynomial Regression...")
        if not os.path.exists('poly_coeffs.npy'):
            raise FileNotFoundError("poly_coeffs.npy not found!")
        coeffs = np.load('poly_coeffs.npy')
        pm2_idx = selected_features.index('pm2_5') if 'pm2_5' in selected_features else 0
        pred["Polynomial"] = np.polyval(coeffs, X_future_scaled[:, pm2_idx])
        
        # 3. PyTorch Linear
        print("  3️⃣ PyTorch Linear...")
        if not os.path.exists('linear_model.pth'):
            raise FileNotFoundError("linear_model.pth not found!")
            
        class LinearModel(nn.Module):
            def __init__(self, n): 
                super().__init__()
                self.linear = nn.Linear(n, 1)
            def forward(self, x): 
                return self.linear(x)
        
        m3 = LinearModel(len(selected_features))
        m3.load_state_dict(torch.load("linear_model.pth", map_location='cpu'))
        m3.eval()
        with torch.no_grad():
            pred["PyTorch_Linear"] = m3(torch.tensor(X_future_scaled, dtype=torch.float32)).numpy().flatten()
        
        # 4. PyTorch MLP
        print("  4️⃣ PyTorch MLP...")
        if not os.path.exists('mlp_model.pth'):
            raise FileNotFoundError("mlp_model.pth not found!")
            
        class MLP(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.fc1 = nn.Linear(n, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.r = nn.ReLU()
            def forward(self, x): 
                return self.fc3(self.r(self.fc2(self.r(self.fc1(x)))))
        
        m4 = MLP(len(selected_features))
        m4.load_state_dict(torch.load("mlp_model.pth", map_location='cpu'))
        m4.eval()
        with torch.no_grad():
            pred["PyTorch_MLP"] = m4(torch.tensor(X_future_scaled, dtype=torch.float32)).numpy().flatten()
        
        # 5. GB
        print("  5️⃣ Gradient Boosting...")
        if not os.path.exists('gb_model.pkl'):
            raise FileNotFoundError("gb_model.pkl not found!")
        gb = joblib.load("gb_model.pkl")
        pred["GB"] = gb.predict(X_future)
        
        # 6. XGB
        print("  6️⃣ XGBoost...")
        if not os.path.exists('xgb_model.json'):
            raise FileNotFoundError("xgb_model.json not found!")
        xg = xgb.XGBRegressor()
        xg.load_model("xgb_model.json")
        pred["XGB"] = xg.predict(X_future)
        
        # 7. RF
        print("  7️⃣ Random Forest...")
        if not os.path.exists('rf_model.pkl'):
            raise FileNotFoundError("rf_model.pkl not found!")
        rf = joblib.load("rf_model.pkl")
        pred["RF"] = rf.predict(X_future)
        
        # Ensemble
        print("🎯 Computing ensemble predictions...")
        pred["Ensemble"] = pred[['Linear', 'Polynomial', 'PyTorch_Linear', 'PyTorch_MLP', 'GB', 'XGB', 'RF']].mean(axis=1)
        
        # Clip
        print("✂️ Clipping predictions to valid AQI range [0, 500]...")
        for col in pred.columns[2:]:
            pred[col] = np.clip(pred[col], 0, 500)
        
        # Closest Model
        print("🎯 Finding closest model for each prediction...")
        model_cols = pred.columns.drop(["datetime_utc", "Actual_AQI"])
        pred["Closest_Model"] = (pred[model_cols].sub(pred["Actual_AQI"], axis=0).abs().idxmin(axis=1))
        
        # Summary Metrics
        print("📊 Computing summary metrics...")
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
        
        # Save to data/
        print("💾 Saving results to CSV files...")
        pred.to_csv("data/future_aqi_predictions.csv", index=False)
        comparison_summary.to_csv("data/future_prediction_comparison.csv", index=False)
        
        print("✅ Saved: data/future_aqi_predictions.csv")
        print("✅ Saved: data/future_prediction_comparison.csv")
        
        print("\n📊 Sample Output:")
        print(pred.head(10).to_string())
        print("\n📊 Forecast Performance Summary:")
        print(comparison_summary.to_string(index=False))
        print("\n✅ Future predictions generated successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR in generate_future_predictions: {type(e).__name__}: {str(e)}")
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        raise  # Re-raise to ensure the workflow fails properly

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Fetch data
    df_feat = fetch_from_hopsworks(feature_group_name="aqi_features", version=1, for_training=True)
    
    # Apply feature engineering to historical data FIRST (before prepare_data uses it)
    print("\n🔧 Applying feature engineering to historical data...")
    df_feat = engineer_features(df_feat, is_forecast=False)
    print(f"✅ Historical data with features: {df_feat.shape}")
   
    # Prepare data
    (X_train, X_val, X_test, y_train, y_val, y_test,
     X_train_scaled, X_val_scaled, X_test_scaled, selected_features) = prepare_data(df_feat)
   
    # Train all models
    results = train_all_models(
        X_train, X_val, X_test, y_train, y_val, y_test,
        X_train_scaled, X_val_scaled, X_test_scaled, selected_features
    )
   
    # Generate future predictions
    generate_future_predictions(df_feat, selected_features)
   
    print("\n" + "="*70)
    print("✅ FULL PIPELINE COMPLETE! (Training + Forecasting)")
    print("="*70)
    print("\nSaved Models (using best checkpoints where applicable):")
    print(" - linear_model.pkl")
    print(" - poly_coeffs.npy")
    print(" - linear_model.pth (best checkpoint)")
    print(" - mlp_model.pth (best checkpoint)")
    print(" - gb_model.pkl")
    print(" - xgb_model.json")
    print(" - rf_model.pkl")
    print(" - scaler.pkl")
    print(" - selected_features.json")
    print(" - data/training_results.csv (best across runs)")
    print(" - data/future_aqi_predictions.csv")
    print(" - data/future_prediction_comparison.csv")
    print(" - Checkpoints in 'checkpoints/' for resuming/inspection")
