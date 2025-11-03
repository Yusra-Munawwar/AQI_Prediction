"""
Complete AQI Prediction Model Training Pipeline
Includes data fetching, preprocessing, model training, and evaluation
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
from datetime import datetime
warnings.filterwarnings('ignore')

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
# METRICS CALCULATION
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

def compare_and_save_best_results(results_summary, prev_path='previous_best_results.csv', current_path='training_results.csv'):
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
        results_summary.to_csv('training_results.csv', index=False)
        print("\n✅ Training complete! Checkpoints saved in 'checkpoints/' directory.")
   
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Fetch data
    df_feat = fetch_from_hopsworks(feature_group_name="aqi_features", version=1, for_training=True)
   
    # Prepare data
    (X_train, X_val, X_test, y_train, y_val, y_test,
     X_train_scaled, X_val_scaled, X_test_scaled, selected_features) = prepare_data(df_feat)
   
    # Train all models
    results = train_all_models(
        X_train, X_val, X_test, y_train, y_val, y_test,
        X_train_scaled, X_val_scaled, X_test_scaled, selected_features
    )
   
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
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
    print(" - training_results.csv (best across runs)")
    print(" - Checkpoints in 'checkpoints/' for resuming/inspection")


