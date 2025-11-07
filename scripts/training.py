"""
train.py - Complete AQI Prediction Model Training Pipeline
Combines data fetching, preparation, model training, and future predictions
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import pickle
import joblib
import hopsworks
import os
import requests
from datetime import datetime, timezone

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
            invalid_mask = (df['datetime_utc'].isna() | 
                          (df['datetime_utc'] < pd.Timestamp('1678-01-01')) | 
                          (df['datetime_utc'] > pd.Timestamp('2262-04-11')))
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


def compare_datasets(local_df, hops_df, tolerance=1e-6):
    """
    Compare local vs. Hopsworks datasets
    """
    print("\n" + "="*70)
    print("DATASET COMPARISON")
    print("="*70)

    # Sort both by datetime_utc for fair comparison
    local_sorted = local_df.sort_values('datetime_utc').reset_index(drop=True)
    hops_sorted = hops_df.sort_values('datetime_utc').reset_index(drop=True)
    print("🔄 Both datasets sorted by 'datetime_utc' for comparison")

    # Handle row count mismatch
    if len(local_sorted) != len(hops_sorted):
        print(f"⚠️ Row count mismatch: Local={len(local_sorted)}, Hops={len(hops_sorted)}")
        min_len = min(len(local_sorted), len(hops_sorted))
        local_sorted = local_sorted.iloc[:min_len].reset_index(drop=True)
        hops_sorted = hops_sorted.iloc[:min_len].reset_index(drop=True)
        print(f"   → Truncated both to {min_len} rows for comparison.")

    # Shape comparison
    print(f"Local shape (aligned): {local_sorted.shape}")
    print(f"Hopsworks shape (aligned): {hops_sorted.shape}")
    if local_sorted.shape[0] == hops_sorted.shape[0]:
        print("✅ Row counts match (after alignment)")

    # Columns comparison
    local_cols = set(local_sorted.columns)
    hops_cols = set(hops_sorted.columns)
    expected_extras = {'id'}
    core_hops_cols = hops_cols - expected_extras
    
    if local_cols == core_hops_cols:
        print("✅ Core columns match exactly")
    else:
        print("⚠️ MISMATCH: Core columns differ!")
        print("Missing in Hopsworks:", local_cols - core_hops_cols)
        print("Extra in Hopsworks:", hops_cols - expected_extras - local_cols)

    # Summary stats comparison
    if 'us_aqi' in local_sorted.columns:
        local_mean = local_sorted['us_aqi'].mean()
        hops_mean = hops_sorted['us_aqi'].mean()
        print(f"\nTarget 'us_aqi' mean: Local={local_mean:.4f} | Hops={hops_mean:.4f}")
        
        overall_match = (local_cols == core_hops_cols) and (abs(local_mean - hops_mean) < tolerance)
        return overall_match
    
    return False


# =============================================================================
# PYTORCH MODEL DEFINITIONS
# =============================================================================

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


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    # SECTION 1: Fetch data from Hopsworks
    df_feat = fetch_from_hopsworks(feature_group_name="aqi_features", version=1, for_training=True)
    print("\n✅ Fetch complete!")
    
    # =============================================================================
    # SECTION 2: DATA PREPARATION & SPLITTING
    # =============================================================================
    print("\n" + "="*70)
    print("SECTION 2: DATA PREPARATION & SPLITTING")
    print("="*70)

    # Regenerate selected features (correlation > 0.3, exclude non-features)
    exclude = ['us_aqi', 'aqi_category', 'id', 'datetime_utc', 'aqi']
    feature_cols = [col for col in df_feat.columns if col not in exclude]
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
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Scaling complete!")

    # TimeSeries CV setup
    tscv = TimeSeriesSplit(n_splits=5)
    print("\n✅ Preparation complete!")

    # =============================================================================
    # SECTION 3: MODEL TRAINING (7 MODELS WITH REGULARIZATION)
    # =============================================================================
    print("\n" + "="*70)
    print("SECTION 3: MODEL TRAINING (7 MODELS WITH REGULARIZATION)")
    print("="*70)
    
    results = {}

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
    print(f" Val:   MAE={mae1_v:.2f}, RMSE={rmse1_v:.2f}, R²={r21_v:.3f}, MAPE={mape1_v:.2f}%")
    print(f" Test:  MAE={mae1:.2f}, RMSE={rmse1:.2f}, R²={r21:.3f}, MAPE={mape1:.2f}% | Gap: {r21_t - r21:.3f}")
    
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
        'test_r2': r21, 'test_rmse': rmse1, 'test_mae': mae1, 
        'test_mape': mape1, 'cv_r2': cv_r2_1
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
    print(f" Val:   MAE={mae2_v:.2f}, RMSE={rmse2_v:.2f}, R²={r22_v:.3f}, MAPE={mape2_v:.2f}%")
    print(f" Test:  MAE={mae2:.2f}, RMSE={rmse2:.2f}, R²={r22:.3f}, MAPE={mape2:.2f}% | Gap: {r22_t - r22:.3f}")
    
    results['poly'] = {
        'model': p_coeffs, 'name': 'Polynomial Regression',
        'test_r2': r22, 'test_rmse': rmse2, 'test_mae': mae2, 'test_mape': mape2
    }
    np.save('poly_coeffs.npy', p_coeffs)

    # ==================== MODEL 3: PyTorch Linear ====================
    print("\n3️⃣ PyTorch Linear Regressor...")
    device = torch.device('cpu')
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
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
        y_train_pred3 = model3(X_train_t).cpu().numpy().flatten()
        y_val_pred3 = model3(X_val_t).cpu().numpy().flatten()
        y_test_pred3 = model3(X_test_t).cpu().numpy().flatten()
    
    mae3_t, rmse3_t, r23_t, mape3_t = calc_metrics(y_train, y_train_pred3)
    mae3_v, rmse3_v, r23_v, mape3_v = calc_metrics(y_val, y_val_pred3)
    mae3, rmse3, r23, mape3 = calc_metrics(y_test, y_test_pred3)
    
    print(f" Train: MAE={mae3_t:.2f}, RMSE={rmse3_t:.2f}, R²={r23_t:.3f}, MAPE={mape3_t:.2f}%")
    print(f" Val:   MAE={mae3_v:.2f}, RMSE={rmse3_v:.2f}, R²={r23_v:.3f}, MAPE={mape3_v:.2f}%")
    print(f" Test:  MAE={mae3:.2f}, RMSE={rmse3:.2f}, R²={r23:.3f}, MAPE={mape3:.2f}% | Gap: {r23_t - r23:.3f}")
    
    results['pytorch_linear'] = {
        'model': model3, 'name': 'PyTorch Linear',
        'test_r2': r23, 'test_rmse': rmse3, 'test_mae': mae3, 'test_mape': mape3
    }
    torch.save(model3.state_dict(), 'linear_model.pth')

    # ==================== MODEL 4: PyTorch MLP ====================
    print("\n4️⃣ PyTorch MLP (increased dropout)...")
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
        y_train_pred4 = model4(X_train_t).cpu().numpy().flatten()
        y_val_pred4 = model4(X_val_t).cpu().numpy().flatten()
        y_test_pred4 = model4(X_test_t).cpu().numpy().flatten()
    
    mae4_t, rmse4_t, r24_t, mape4_t = calc_metrics(y_train, y_train_pred4)
    mae4_v, rmse4_v, r24_v, mape4_v = calc_metrics(y_val, y_val_pred4)
    mae4, rmse4, r24, mape4 = calc_metrics(y_test, y_test_pred4)
    
    print(f" Train: MAE={mae4_t:.2f}, RMSE={rmse4_t:.2f}, R²={r24_t:.3f}, MAPE={mape4_t:.2f}%")
    print(f" Val:   MAE={mae4_v:.2f}, RMSE={rmse4_v:.2f}, R²={r24_v:.3f}, MAPE={mape4_v:.2f}%")
    print(f" Test:  MAE={mae4:.2f}, RMSE={rmse4:.2f}, R²={r24:.3f}, MAPE={mape4:.2f}% | Gap: {r24_t - r24:.3f}")
    
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
    print(f" Val:   MAE={mae5_v:.2f}, RMSE={rmse5_v:.2f}, R²={r25_v:.3f}, MAPE={mape5_v:.2f}%")
    print(f" Test:  MAE={mae5:.2f}, RMSE={rmse5:.2f}, R²={r25:.3f}, MAPE={mape5:.2f}% | Gap: {r25_t - r25:.3f}")
    
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
        'test_r2': r25, 'test_rmse': rmse5, 'test_mae': mae5, 
        'test_mape': mape5, 'cv_r2': cv_r2_5
    }
    joblib.dump(model5, 'gb_model.pkl')

    # ==================== MODEL 6: XGBoost ====================
    print("\n6️⃣ XGBoost (higher reg)...")
    model6 = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.7, 
        colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=2.0, 
        random_state=42, verbosity=0
    )
    model6.fit(X_train, y_train)
    
    y_train_pred6 = model6.predict(X_train)
    y_val_pred6 = model6.predict(X_val)
    y_test_pred6 = model6.predict(X_test)
    
    mae6_t, rmse6_t, r26_t, mape6_t = calc_metrics(y_train, y_train_pred6)
    mae6_v, rmse6_v, r26_v, mape6_v = calc_metrics(y_val, y_val_pred6)
    mae6, rmse6, r26, mape6 = calc_metrics(y_test, y_test_pred6)
    
    print(f" Train: MAE={mae6_t:.2f}, RMSE={rmse6_t:.2f}, R²={r26_t:.3f}, MAPE={mape6_t:.2f}%")
    print(f" Val:   MAE={mae6_v:.2f}, RMSE={rmse6_v:.2f}, R²={r26_v:.3f}, MAPE={mape6_v:.2f}%")
    print(f" Test:  MAE={mae6:.2f}, RMSE={rmse6:.2f}, R²={r26:.3f}, MAPE={mape6:.2f}% | Gap: {r26_t - r26:.3f}")
    
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
        'test_r2': r26, 'test_rmse': rmse6, 'test_mae': mae6, 
        'test_mape': mape6, 'cv_r2': cv_r2_6
    }
    model6.save_model('xgb_model.json')

    # ==================== MODEL 7: Random Forest ====================
    print("\n7️⃣ Random Forest (higher mins)...")
    model7 = RandomForestRegressor(
        n_estimators=50, max_depth=8, min_samples_split=50, 
        min_samples_leaf=20, max_features=0.5, random_state=42
    )
    model7.fit(X_train, y_train)
    
    y_train_pred7 = model7.predict(X_train)
    y_val_pred7 = model7.predict(X_val)
    y_test_pred7 = model7.predict(X_test)
    
    mae7_t, rmse7_t, r27_t, mape7_t = calc_metrics(y_train, y_train_pred7)
    mae7_v, rmse7_v, r27_v, mape7_v = calc_metrics(y_val, y_val_pred7)
    mae7, rmse7, r27, mape7 = calc_metrics(y_test, y_test_pred7)
    
    print(f" Train: MAE={mae7_t:.2f}, RMSE={rmse7_t:.2f}, R²={r27_t:.3f}, MAPE={mape7_t:.2f}%")
    print(f" Val:   MAE={mae7_v:.2f}, RMSE={rmse7_v:.2f}, R²={r27_v:.3f}, MAPE={mape7_v:.2f}%")
    print(f" Test:  MAE={mae7:.2f}, RMSE={rmse7:.2f}, R²={r27:.3f}, MAPE={mape7:.2f}% | Gap: {r27_t - r27:.3f}")
    
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
        'test_r2': r27, 'test_rmse': rmse7, 'test_mae': mae7, 
        'test_mape': mape7, 'cv_r2': cv_r2_7
    }
    joblib.dump(model7, 'rf_model.pkl')

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY (Test Metrics)")
    print("="*60)
    results_summary = pd.DataFrame([
        {
            'Model': res['name'], 
            'Test MAE': res['test_mae'], 
            'Test RMSE': res['test_rmse'],
            'Test R²': res['test_r2'], 
            'Test MAPE (%)': res['test_mape'], 
            'CV R²': res.get('cv_r2', np.nan)
        }
        for res in results.values()
    ]).sort_values('Test R²', ascending=False)
    print(results_summary.to_string(index=False, float_format='%.3f'))
    results_summary.to_csv('training_results.csv', index=False)
    print("\n✅ Training complete!")
    
    # Save selected features for future use
    with open('selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    return results, selected_features, df_feat


# =============================================================================
# FUTURE PREDICTION FUNCTIONS
# =============================================================================

def calc_us_aqi(row):
    """
    Calculate US AQI from pollutant concentrations (simplified version)
    You should replace this with your actual implementation
    """
    # Placeholder - replace with your actual calc_us_aqi function
    pm25 = row.get('pm2_5', 0)
    # Simplified AQI calculation based on PM2.5
    if pm25 <= 12.0:
        return int((50/12.0) * pm25)
    elif pm25 <= 35.4:
        return int(50 + ((100-50)/(35.4-12.0)) * (pm25-12.0))
    elif pm25 <= 55.4:
        return int(100 + ((150-100)/(55.4-35.4)) * (pm25-35.4))
    elif pm25 <= 150.4:
        return int(150 + ((200-150)/(150.4-55.4)) * (pm25-55.4))
    elif pm25 <= 250.4:
        return int(200 + ((300-200)/(250.4-150.4)) * (pm25-150.4))
    else:
        return int(300 + ((500-300)/(500.4-250.4)) * (pm25-250.4))


def engineer_features(df):
    """
    Add temporal features to dataframe
    You should replace this with your actual engineer_features function
    """
    df = df.copy()
    df['hour'] = df['datetime_utc'].dt.hour
    df['month'] = df['datetime_utc'].dt.month
    df['day_of_week'] = df['datetime_utc'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def fetch_forecast_pollutants(lat, lon, api_key):
    """
    Fetch forecast pollutant data from OpenWeatherMap API
    """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"
    r = requests.get(url).json()

    records = []
    UTC = timezone.utc
    
    for item in r["list"]:
        dt = datetime.fromtimestamp(item["dt"], tz=UTC)
        comp = item["components"]
        aqi_1_5 = item["main"]["aqi"]

        # Convert to US AQI
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


def run_future_predictions(selected_features, df_clean, LAT, LON):
    """
    Run future AQI predictions using trained models
    """
    print("\n" + "="*70)
    print("SECTION 6: FUTURE AQI PREDICTION USING FORECAST POLLUTANTS")
    print("="*70)

    # Fetch forecast data
    forecast_df = fetch_forecast_pollutants(LAT, LON, os.getenv("OWM_API_KEY"))
    print(f"✅ Forecast pollutant data received: {forecast_df.shape[0]} hours")

    # Clean & Feature Engineering
    pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    for col in pollutants:
        if col in forecast_df.columns:
            forecast_df[col] = forecast_df[col].clip(lower=0)
            forecast_df[col] = forecast_df[col].fillna(df_clean[col].median())
            forecast_df[col] = forecast_df[col].clip(upper=df_clean[col].quantile(0.995))

    forecast_feat = engineer_features(forecast_df)
    X_future = forecast_feat[selected_features].fillna(0)

    # Scale where needed
    scaler = joblib.load('scaler.pkl')
    X_future_scaled = scaler.transform(X_future)

    # Initialize predictions dataframe
    pred = pd.DataFrame({
        "datetime_utc": forecast_df["datetime_utc"],
        "Actual_AQI": forecast_df["Actual_AQI"].astype(float)
    })

    # 1. Linear Regression
    model1 = pickle.load(open('linear_model.pkl', 'rb'))
    X_sm = sm.add_constant(X_future_scaled, has_constant='add')
    pred["Linear"] = model1.predict(X_sm)

    # 2. Polynomial Regression (PM2.5)
    coeffs = np.load('poly_coeffs.npy')
    pm2_idx = selected_features.index('pm2_5') if 'pm2_5' in selected_features else 0
    pred["Polynomial"] = np.polyval(coeffs, X_future_scaled[:, pm2_idx])

    # 3. PyTorch Linear Model
    m3 = LinearModel(len(selected_features))
    m3.load_state_dict(torch.load("linear_model.pth"))
    m3.eval()
    with torch.no_grad():
        pred["PyTorch_Linear"] = m3(torch.tensor(X_future_scaled, dtype=torch.float32)).numpy().flatten()

    # 4. PyTorch MLP Model
    m4 = MLP(len(selected_features))
    m4.load_state_dict(torch.load("mlp_model.pth"))
    m4.eval()
    with torch.no_grad():
        pred["PyTorch_MLP"] = m4(torch.tensor(X_future_scaled, dtype=torch.float32)).numpy().flatten()

    # 5. Gradient Boosting
    gb = joblib.load("gb_model.pkl")
    pred["GB"] = gb.predict(X_future)

    # 6. XGBoost
    xg = xgb.XGBRegressor()
    xg.load_model("xgb_model.json")
    pred["XGB"] = xg.predict(X_future)

    # 7. Random Forest
    rf = joblib.load("rf_model.pkl")
    pred["RF"] = rf.predict(X_future)

    # Ensemble
    model_cols = ['Linear', 'Polynomial', 'PyTorch_Linear', 'PyTorch_MLP', 'GB', 'XGB', 'RF']
    pred["Ensemble"] = pred[model_cols].mean(axis=1)

    # Clip to US AQI range
    for col in pred.columns[2:]:
        pred[col] = np.clip(pred[col], 0, 500)

    # Identify Best Model Per Timestamp
    pred["Closest_Model"] = (pred[model_cols].sub(pred["Actual_AQI"], axis=0)
                             .abs().idxmin(axis=1))

    # Compute Summary Metrics
    model_map = {
        'Linear': 'Linear',
        'Polynomial': 'Polynomial',
        'PyTorch_Linear': 'PyTorch Linear',
        'PyTorch_MLP': 'PyTorch MLP',
        'GB': 'Gradient Boosting',
        'XGB': 'XGBoost',
        'RF': 'Random Forest',
        'Ensemble': 'Ensemble'
    }

    summary_data = []
    y_true = pred['Actual_AQI']

    for model_col in model_cols + ['Ensemble']:
        if model_col in model_map:
            y_pred = pred[model_col]
            mae, rmse, r2, mape = calc_metrics(y_true, y_pred)
            summary_data.append({
                'Model': model_map[model_col],
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            })

    comparison_summary = pd.DataFrame(summary_data).sort_values('R²', ascending=False)

    # Save Outputs
    pred.to_csv("future_aqi_predictions.csv", index=False)
    comparison_summary.to_csv("future_prediction_comparison.csv", index=False)
    print("💾 Saved: future_aqi_predictions.csv")
    print("💾 Saved: future_prediction_comparison.csv")

    print("\n📊 Sample Predictions:")
    print(pred.head(10))
    print("\n📊 Forecast Performance Summary:")
    print(comparison_summary)
    
    return pred, comparison_summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("AQI PREDICTION MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Run training pipeline
    results, selected_features, df_feat = main()
    
    # Optional: Run future predictions if you have the required data
    # Uncomment and configure the following lines if you want to run predictions
    """
    # You need to provide these values
    LAT = 24.8607  # Example: Karachi latitude
    LON = 67.0011  # Example: Karachi longitude
    
    # You also need df_clean from your data preparation
    # This should be loaded or prepared before calling run_future_predictions
    df_clean = df_feat  # Or load your cleaned dataframe
    
    pred, comparison = run_future_predictions(selected_features, df_clean, LAT, LON)
    """
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - linear_model.pkl")
    print("  - poly_coeffs.npy")
    print("  - linear_model.pth")
    print("  - mlp_model.pth")
    print("  - gb_model.pkl")
    print("  - xgb_model.json")
    print("  - rf_model.pkl")
    print("  - scaler.pkl")
    print("  - selected_features.pkl")
    print("  - training_results.csv")
    print("\nAll models trained and saved successfully!")
