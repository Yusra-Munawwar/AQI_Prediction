# =============================================================================
# AQI TRAINING & FORECAST PIPELINE (Combined Script with Checkpointing)
# =============================================================================
import pandas as pd
import numpy as np
import joblib
import hopsworks
import os
import json
import requests
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from datetime import datetime as dt_datetime
import time

def utcfromtimestamp(ts):
    return pd.Timestamp(dt_datetime.utcfromtimestamp(ts), tz='UTC')
# --- GLOBAL CONFIGURATION ---
API_KEY = os.getenv("OWM_API_KEY") 
LAT = 24.8607    # Karachi latitude
LON = 67.0011    # Karachi longitude
HISTORIC_PATH = "data/2years_features_clean.csv" 
PERFORMANCE_FILE = "model_artifacts/best_model_performance.json"
CHECKPOINT_MODEL_PATH = "model_artifacts/best_model.pkl"

# =============================================================================
# PART 1: MODEL TRAINING
# =============================================================================

# ----------------------------------------------------------------------
# 1. FETCH DATA (Training Data)
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("PART 1: FETCH TRAINING DATA")
print("="*80)

def fetch_training_data():
    try:
        # HOPSWORKS_API_KEY is retrieved from environment variables
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_features", version=4)
        cols = ['datetime_utc', 'us_aqi'] + [
            'pm2_5', 'pm10', 'co', 'no2', 'so2',
            'month_cos', 'total_pm', 'total_gases',
            'no2_o3_ratio', 'pm2_5_rolling_3h', 'pm10_rolling_3h',
            'co_rolling_3h', 'pm2_5_co_interaction',
            'hour_sin', 'hour_cos'
        ]
        df = fg.select(cols).read()
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', utc=True)
        print(f"Fetched from Hopsworks: {df.shape}")
        return df
    except Exception as e:
        print(f"Hopsworks failed: {e}")
        # Fallback path
        df = pd.read_csv(HISTORIC_PATH) 
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
        print(f"Fallback CSV: {df.shape}")
        return df

df = fetch_training_data()
df = df.sort_values('datetime_utc').reset_index(drop=True)

# ----------------------------------------------------------------------
# 2. FEATURES (NO LAG)
# ----------------------------------------------------------------------
SELECTED_FEATURES = [
    'pm2_5', 'pm10', 'co', 'no2', 'so2',
    'month_cos', 'total_pm', 'total_gases',
    'no2_o3_ratio', 'pm2_5_rolling_3h', 'pm10_rolling_3h',
    'co_rolling_3h', 'pm2_5_co_interaction',
    'hour_sin', 'hour_cos'
]

X = df[SELECTED_FEATURES].fillna(0).values
y = df['us_aqi'].values

# ----------------------------------------------------------------------
# 3. SPLIT
# ----------------------------------------------------------------------
n = len(X)
train_end = int(0.7 * n)
val_end   = int(0.8 * n)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

# ----------------------------------------------------------------------
# 4. SCALING
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(scaler, "model_artifacts/scaler.pkl")

# ----------------------------------------------------------------------
# 5. METRICS
# ----------------------------------------------------------------------
def calc_metrics(y_true, y_pred, name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    print(f"{name:18} MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | R²: {r2:6.3f} | MAPE: {mape:6.2f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

# ----------------------------------------------------------------------
# 6. TIME-SERIES CV
# ----------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

# ----------------------------------------------------------------------
# 7. TRAIN MODELS (SAVE ALL)
# ----------------------------------------------------------------------
models      = {}
cv_scores   = {}
all_metrics = {"train": {}, "val": {}, "test": {}, "cv_mae": {}}

print("\n" + "="*80)
print("TRAINING & SAVING ALL MODELS")
print("="*80)

def train_with_cv(model, name, use_scaled=False):
    X_tr = X_train_scaled if use_scaled else X_train
    X_va = X_val_scaled   if use_scaled else X_val
    X_te = X_test_scaled    if use_scaled else X_test

    # --- 5-fold CV ---
    cv_mae = []
    for tr_idx, val_idx in tscv.split(X_tr):
        X_f_tr, X_f_va = X_tr[tr_idx], X_tr[val_idx]
        y_f_tr, y_f_va = y_train[tr_idx], y_train[val_idx]

        # Clone model to avoid refit issues
        if name == "xgboost":
            m = xgb.XGBRegressor(**model.get_params())
            m.fit(X_f_tr, y_f_tr, eval_set=[(X_f_va, y_f_va)], verbose=False)
        elif name == "lightgbm":
            m = lgb.LGBMRegressor(**model.get_params())
            m.fit(X_f_tr, y_f_tr, eval_set=[(X_f_va, y_f_va)], 
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        else:
            m = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
            m.fit(X_f_tr, y_f_tr)

        pred = m.predict(X_f_va)
        cv_mae.append(mean_absolute_error(y_f_va, pred))

    cv_mae_mean = np.mean(cv_mae)
    print(f"{name:12} CV MAE: {cv_mae_mean:.3f}")

    # --- Final fit on full train ---
    final_model = model
    final_model.fit(X_tr, y_train)

    # --- Predictions ---
    pred_train = final_model.predict(X_tr)
    pred_val   = final_model.predict(X_va)
    pred_test  = final_model.predict(X_te)

    # Store metrics
    all_metrics['train'][name] = calc_metrics(y_train, pred_train, f"{name} Train")
    all_metrics['val'][name]   = calc_metrics(y_val,   pred_val,   f"{name} Val")
    all_metrics['test'][name]  = calc_metrics(y_test,  pred_test,  f"{name} Test")
    all_metrics['cv_mae'][name] = cv_mae_mean

    # Save this model
    model_path = f"model_artifacts/model_{name}.pkl"
    joblib.dump(final_model, model_path)

    # Store in dict
    models[name] = final_model
    cv_scores[name] = cv_mae_mean
    return models, cv_scores

# --------------------- TRAIN ALL MODELS ---------------------

print("1. XGBoost")
xgb_model = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)
models, cv_scores = train_with_cv(xgb_model, "xgboost")

print("2. LightGBM")
lgb_model = lgb.LGBMRegressor(n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)
models, cv_scores = train_with_cv(lgb_model, "lightgbm")

print("3. CatBoost")
cb_model = cb.CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=False)
models, cv_scores = train_with_cv(cb_model, "catboost")

print("4. Random Forest")
rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42, n_jobs=-1)
models, cv_scores = train_with_cv(rf_model, "rf")

print("5. Gradient Boosting")
gb_model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_split=20, min_samples_leaf=10, random_state=42)
models, cv_scores = train_with_cv(gb_model, "gb")

print("6. Ridge")
ridge_model = Ridge(alpha=10.0, random_state=42)
models, cv_scores = train_with_cv(ridge_model, "ridge", use_scaled=True)

print("7. Linear")
lr_model = LinearRegression()
models, cv_scores = train_with_cv(lr_model, "linear", use_scaled=True)


# ----------------------------------------------------------------------
# 8. BEST MODEL (WITH CHECKPOINT LOGIC)
# ----------------------------------------------------------------------

# 1. IDENTIFY CURRENT RUN'S BEST MODEL BASED ON CV MAE
best_name_cv = min(cv_scores, key=cv_scores.get)
best_mod_cv = models[best_name_cv]
new_cv_mae = cv_scores[best_name_cv]
is_new_best = False # Default status

print("\n" + "="*80)
print(f"CURRENT RUN BEST (CV): {best_name_cv.upper()} | CV MAE: {new_cv_mae:.3f}")
print("="*80)

# Final test prediction for the CV-selected model (for logging purposes)
X_test_fin = X_test_scaled if best_name_cv in ["ridge", "linear"] else X_test
y_test_pred = best_mod_cv.predict(X_test_fin)
new_test_metrics = calc_metrics(y_test, y_test_pred, "NEW TEST METRICS")
new_test_mae = new_test_metrics['mae']


# --- CHECKPOINT LOGIC (PRIORITIZING CV MAE) ---
historical_best_cv_mae = float('inf')
historical_best_name = "N/A" # Used for tracking historical name

# 🛑 FIX: Initialize variables for final summary/Part 2 in case loading fails or this is the first run.
# This ensures Section 9 and 10 do not crash on NameError.
current_best_mae_test_only = float('inf') # Only used to mirror old summary print, not for checkpoint
current_best_name_test_only = "N/A"       # Only used to mirror old summary print

# Load previous best performance from the JSON file
if os.path.exists(PERFORMANCE_FILE):
    try:
        with open(PERFORMANCE_FILE, "r") as f:
            prev_performance = json.load(f)
            # IMPORTANT: Load the historical CV MAE for comparison
            historical_best_cv_mae = prev_performance.get("cv_mae", float('inf')) 
            historical_best_name = prev_performance.get("model_name", "N/A")
            # Load old Test MAE values for the final summary printout (Section 10) only
            current_best_mae_test_only = prev_performance.get("test_mae", float('inf'))
            current_best_name_test_only = prev_performance.get("model_name", "N/A") 
            
            print(f"Historical Best CV MAE: {historical_best_cv_mae:.3f} ({historical_best_name})")
    except Exception as e:
        print(f"Error loading {PERFORMANCE_FILE}: {e}. Starting fresh checkpoint.")
else:
    print("No previous best model performance found. Saving current CV best model.")

# 2. COMPARE AND SAVE/PROMOTE THE MODEL
if new_cv_mae < historical_best_cv_mae:
    # A NEW, MORE ROBUST MODEL IS FOUND!
    print(f"✅ New model ({best_name_cv.upper()}) is more robust (CV MAE {new_cv_mae:.3f} < {historical_best_cv_mae:.3f}). Saving checkpoint.")
    
    # 1. Save the model as the new official 'best' checkpoint
    joblib.dump(best_mod_cv, CHECKPOINT_MODEL_PATH)
    print(f"NEW BEST Model SAVED: {CHECKPOINT_MODEL_PATH}")
    
    # 2. Update the performance tracking file with the new CV metrics
    performance_data = {
        "model_name": best_name_cv,
        "cv_mae": new_cv_mae,
        "test_mae": new_test_mae, # Keep test MAE for reference
        "test_r2": new_test_metrics['r2'],
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance_data, f, indent=2)

    is_new_best = True
    final_best_name = best_name_cv
    final_test_metrics = new_test_metrics
else:
    print(f"❌ Current CV best ({best_name_cv.upper()} MAE {new_cv_mae:.3f}) is not better than the historical best (MAE {historical_best_cv_mae:.3f}).")
    print(f"Keeping the previously saved {CHECKPOINT_MODEL_PATH} checkpoint.")
    
    is_new_best = False
    # Use the historical name/metrics for the final summary if we kept the old model
    final_best_name = historical_best_name 
    final_test_metrics = new_test_metrics 

# --- Prepare variables for Section 9 and 10 ---
# The name of the model currently in the best_model.pkl file (for Part 2 scaling check)
current_checkpoint_name = final_best_name 

print("\n" + "="*80)
print("CHECKPOINT COMPLETE")
print("="*80)


# ----------------------------------------------------------------------
# 9. SAVE ALL METRICS & CONFIG
# ----------------------------------------------------------------------
joblib.dump(SELECTED_FEATURES, "model_artifacts/selected_features.pkl")

# Save a comprehensive metrics file for the *current* run
with open("model_artifacts/metrics_current_run.json", "w") as f:
    json.dump({
        "trained_best_model": best_name_cv, # Use the current run's best CV model name
        "new_test_metrics": new_test_metrics,
        "cv_mae": cv_scores,
        "test_mae_all": {name: all_metrics['test'][name]['mae'] for name in models},
        "historical_best_cv_mae": historical_best_cv_mae, # Use the correct historical CV MAE
        "is_new_historical_best": is_new_best,
        "features": SELECTED_FEATURES,
        "scaler_needed": best_name_cv in ["ridge", "linear"],
        "all_models_saved": [f"model_{name}.pkl" for name in models],
        "all_metrics": all_metrics
    }, f, indent=2)

print(f"Current run metrics saved to model_artifacts/metrics_current_run.json")

# ----------------------------------------------------------------------
# 10. TRAINING FINAL SUMMARY
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("TRAINING COMPLETE – CHECKPOINT STATUS")
print("="*80)
# Use the correct names and CV MAE values for the summary
print(f"Best Model from Current Run : {best_name_cv.upper()}")
print(f"Current Run Test MAE        : {new_test_metrics['mae']:.3f}")
# 🛑 FIX: Use the historical name/MAE derived from the file load for the summary
print(f"Historical Best Test MAE    : {current_best_mae_test_only:.3f} ({current_best_name_test_only})") 
print(f"STATUS                      : {'✅ PROMOTED NEW BEST' if is_new_best else '❌ KEPT OLD BEST'}")
print(f"Saved Checkpoint Location   : {CHECKPOINT_MODEL_PATH}")
print("="*80)

# =============================================================================
# PART 2: FORECASTING
# =============================================================================

# ----------------------------------------------------------------------
# 0. CONFIG 
# ----------------------------------------------------------------------
MODEL_PATHS = {
    "xgboost":   "model_artifacts/model_xgboost.pkl",
    "lightgbm":  "model_artifacts/model_lightgbm.pkl",
    "catboost":  "model_artifacts/model_catboost.pkl",
    "rf":        "model_artifacts/model_rf.pkl",
    "gb":        "model_artifacts/model_gb.pkl",
    "ridge":     "model_artifacts/model_ridge.pkl",
    "linear":    "model_artifacts/model_linear.pkl"
}

# Add the historically best checkpoint model to the prediction list
MODEL_PATHS['best_checkpoint'] = CHECKPOINT_MODEL_PATH 


print("\n" + "#"*80)
print("PART 2: 96-HOUR AQI FORECAST")
print("#"*80)

# Load scaler & features (Keep loading from disk)
scaler = joblib.load("model_artifacts/scaler.pkl")
SELECTED_FEATURES = joblib.load("model_artifacts/selected_features.pkl")

# Load models
models = {}
for name, path in MODEL_PATHS.items():
    try:
        # Skip the checkpoint if it wasn't created on the first run
        if name == 'best_checkpoint' and not os.path.exists(path):
             print(f"Warning: {path} not found (first run?). Skipping checkpoint prediction.")
             continue
        models[name] = joblib.load(path)
    except Exception as e:
        print(f"Failed to load {name} from {path}: {e}")

print(f"Loaded {len(models)} models for prediction.")

# ----------------------------------------------------------------------
# 1. US-AQI CONVERSION (Same as Part 1)
# ----------------------------------------------------------------------
breakpoints = {
    'pm2_5': [(0.0, 0, 9.0, 50), (9.1, 51, 35.4, 100), (35.5, 101, 55.4, 150),
              (55.5, 151, 125.4, 200), (125.5, 201, 225.4, 300), (225.5, 301, 500.4, 500)],
    'pm10':  [(0, 0, 54, 50), (55, 51, 154, 100), (155, 101, 254, 150),
              (255, 151, 354, 200), (355, 201, 424, 300), (425, 301, 504, 400),
              (505, 401, 604, 500)],
    'o3':    [(0.000, 0, 0.054, 50), (0.055, 51, 0.070, 100), (0.071, 101, 0.085, 150),
              (0.086, 151, 0.105, 200), (0.106, 201, 0.200, 300), (0.201, 301, 0.404, 400),
              (0.405, 401, 0.604, 500)],
    'no2':   [(0.000, 0, 0.053, 50), (0.054, 51, 0.100, 100), (0.101, 101, 0.360, 150),
              (0.361, 151, 0.649, 200), (0.650, 201, 0.854, 300), (0.855, 301, 1.049, 400),
              (1.050, 401, 2.104, 500)],
    'so2':   [(0.000, 0, 0.004, 50), (0.005, 51, 0.009, 100), (0.010, 101, 0.014, 150),
              (0.015, 151, 0.035, 200), (0.036, 201, 0.075, 300), (0.076, 301, 0.185, 400),
              (0.186, 401, 0.604, 500)],
    'co':    [(0.0, 0, 4.4, 50), (4.5, 51, 9.4, 100), (9.5, 101, 12.4, 150),
              (12.5, 151, 15.4, 200), (15.5, 201, 30.4, 300), (30.5, 301, 50.4, 500)]
}

def to_epa_units(c, pollutant):
    if pd.isna(c) or c <= 0: return 0.0
    if pollutant == 'co':    return c / 1145.0
    if pollutant == 'o3':    return c / 1960.6
    if pollutant == 'no2':   return c / 1881.1
    if pollutant == 'so2':   return c / 2620.0
    return c

def calc_sub_aqi(c, pollutant):
    c_epa = to_epa_units(c, pollutant)
    bps = breakpoints.get(pollutant, [])
    if c_epa <= 0: return 0
    prev_high = -np.inf
    for c_low, i_low, c_high, i_high in bps:
        if c_epa > prev_high and c_epa <= c_high:
            return i_low + ((i_high - i_low) * (c_epa - c_low)) / (c_high - c_low)
        prev_high = c_high
    return 500

def calc_us_aqi(row):
    pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    sub_aqis = [calc_sub_aqi(row.get(p, 0), p) for p in pollutants]
    return max(sub_aqis)

# ----------------------------------------------------------------------
# 2. FETCH 96-HOUR FORECAST
# ----------------------------------------------------------------------
from datetime import datetime as dt_datetime
import time

def utcfromtimestamp(ts):
    return pd.Timestamp(dt_datetime.utcfromtimestamp(ts), tz='UTC')

def get_pollution_forecast(lat, lon, api_key):
    if not api_key:
        raise ValueError("OWM_API_KEY is missing or empty!")
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"
    print(f"Requesting forecast from: {url}")
    
    try:
        resp = requests.get(url, timeout=10)
        print(f"API Response Status: {resp.status_code}")
        
        if resp.status_code == 401:
            raise ValueError("Invalid OWM_API_KEY: Unauthorized (401). Check your OpenWeatherMap API key.")
        if resp.status_code == 429:
            raise ValueError("Rate limited by OpenWeatherMap (429). Too many requests.")
        if resp.status_code != 200:
            raise ValueError(f"API error {resp.status_code}: {resp.text}")
        
        data = resp.json()
        if 'list' not in data or len(data['list']) == 0:
            raise ValueError("Empty forecast data returned from API.")
        
        rows = []
        for item in data['list']:
            dt = utcfromtimestamp(item['dt'])
            comp = item['components']
            rows.append({
                'datetime_utc': dt,
                'pm2_5': comp.get('pm2_5', 0), 
                'pm10': comp.get('pm10', 0), 
                'co': comp.get('co', 0),
                'no2': comp.get('no2', 0), 
                'o3': comp.get('o3', 0), 
                'so2': comp.get('so2', 0)
            })
        df = pd.DataFrame(rows)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
        print(f"Successfully fetched {len(df)} forecast hours.")
        return df.head(96)
    
    except requests.exceptions.Timeout:
        raise ValueError("Request to OpenWeatherMap timed out.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Failed to connect to OpenWeatherMap. Check internet or DNS.")
    except Exception as e:
        print(f"Unexpected error in get_pollution_forecast: {type(e).__name__}: {e}")
        raise
print("Fetching 96-hour forecast...")
try:
    df_forecast = get_pollution_forecast(LAT, LON, API_KEY)
    df_forecast['Actual_AQI'] = df_forecast.apply(calc_us_aqi, axis=1).round().astype(int).clip(0, 500)
except Exception as e:
    print(f"FORECAST FAILED: {e}")
    os.makedirs("data", exist_ok=True)
    dummy = pd.DataFrame({
        'datetime': [pd.Timestamp.now(tz='UTC')], 'Actual_AQI': [0],
        'xgboost': [0], 'lightgbm': [0], 'catboost': [0], 'rf': [0], 'gb': [0],
        'ridge': [0], 'linear': [0], 'Closest_Model': ['error']
    })
    if os.path.exists(CHECKPOINT_MODEL_PATH):
        dummy['best_checkpoint'] = [0]
    dummy.to_csv("data/future_aqi_predictions.csv", index=False)
    pd.DataFrame([{"Model": "ERROR", "MAE": 999, "RMSE": 999, "R²": -1, "MAPE": 999}]).to_csv("data/future_prediction_comparison.csv", index=False)
    print("Pipeline completed with forecast fallback.")
    exit(0)

# ----------------------------------------------------------------------
# 3. LOAD HISTORIC DATA
# ----------------------------------------------------------------------
print(f"Loading historic data from {HISTORIC_PATH}...")
df_hist = pd.read_csv(HISTORIC_PATH)
df_hist['datetime_utc'] = pd.to_datetime(df_hist['datetime_utc'], format='ISO8601', utc=True)
df_hist = df_hist.sort_values('datetime_utc').reset_index(drop=True)
last_3 = df_hist.tail(3)

# ----------------------------------------------------------------------
# 4. BUILD FEATURES
# ----------------------------------------------------------------------
def build_features(df_fc, last_hist):
    df = df_fc.copy()
    df['hour'] = df['datetime_utc'].dt.hour
    df['month'] = df['datetime_utc'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['total_pm'] = df['pm2_5'] + df['pm10']
    df['total_gases'] = df['co'] + df['no2'] + df['o3'] + df['so2']
    df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
    df['pm2_5_co_interaction'] = df['pm2_5'] * df['co']

    combined = pd.concat([last_hist[['pm2_5', 'pm10', 'co']], df[['pm2_5', 'pm10', 'co']]], ignore_index=True)
    df['pm2_5_rolling_3h'] = combined['pm2_5'].rolling(3, min_periods=1).mean().iloc[-96:].values
    df['pm10_rolling_3h']  = combined['pm10'].rolling(3, min_periods=1).mean().iloc[-96:].values
    df['co_rolling_3h']    = combined['co'].rolling(3, min_periods=1).mean().iloc[-96:].values

    if 'us_aqi' in last_hist.columns:
        df['us_aqi_lag1'] = last_hist['us_aqi'].iloc[-1]
    else:
        df['us_aqi_lag1'] = 0 
        
    df = df.drop(columns=['hour', 'month'], errors='ignore')
    return df

df_fc_feat = build_features(df_forecast, last_3)
X = df_fc_feat[SELECTED_FEATURES].fillna(0).values

# ----------------------------------------------------------------------
# 5. PREDICT WITH ALL MODELS
# ----------------------------------------------------------------------
predictions = {'datetime': df_fc_feat['datetime_utc'], 'Actual_AQI': df_fc_feat['Actual_AQI']}
model_names = list(models.keys())

for name in model_names:
    model = models[name]
    
    # 🛑 FIX: Cleaned up the scaling logic for best_checkpoint
    if name in ["ridge", "linear"]:
        X_input = scaler.transform(X)
    elif name == "best_checkpoint" and current_checkpoint_name in ["ridge", "linear"]:
        X_input = scaler.transform(X)
    else:
        X_input = X

    pred = model.predict(X_input)
    pred = np.clip(pred.round().astype(int), 0, 500)
    predictions[name] = pred

# Closest Model (Only compare the 7 trained models + checkpoint for error)
model_names_for_error = [name for name in model_names if name != 'best_checkpoint']
abs_errors = np.abs(np.array([predictions[m] for m in model_names_for_error]) - predictions['Actual_AQI'].values)
closest_idx = np.argmin(abs_errors, axis=0)
predictions['Closest_Model'] = [model_names_for_error[i] for i in closest_idx]

# ----------------------------------------------------------------------
# 6. SAVE future_aqi_predictions.csv (EXACT COLUMNS)
# ----------------------------------------------------------------------
cols = ['datetime', 'Actual_AQI', 'xgboost', 'lightgbm', 'catboost', 'rf', 'gb', 'ridge', 'linear']
if 'best_checkpoint' in predictions:
    cols.append('best_checkpoint')
cols.append('Closest_Model')

df_pred = pd.DataFrame(predictions)[cols]
df_pred.to_csv("data/future_aqi_predictions.csv", index=False)
print("Saved: future_aqi_predictions.csv")

# ----------------------------------------------------------------------
# 7. METRICS → future_prediction_comparison.csv
# ----------------------------------------------------------------------
y_true = df_pred['Actual_AQI'].values
metrics_list = []

for name in [n for n in model_names if n in df_pred.columns]:
    y_pred = df_pred[name].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100 
    metrics_list.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R²": round(r2, 3),
        "MAPE": round(mape, 2)
    })

df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv("data/future_prediction_comparison.csv", index=False)
print("Saved: future_prediction_comparison.csv")
print("\n" + df_metrics.to_string(index=False))

# ----------------------------------------------------------------------
# 8. PLOT — 7 INDIVIDUAL GRAPHS (One per Model)
# ----------------------------------------------------------------------
print("Generating model comparison plots...")

os.makedirs("model_comparison_plots", exist_ok=True)

actual = df_pred['Actual_AQI'].values
dates = df_pred['datetime']

for name in [n for n in model_names if n in df_pred.columns]:
    pred = df_pred[name].values
    
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label='Actual AQI (OWM)', color='black', linewidth=2.5, marker='o', markersize=3)
    
    color = 'teal'
    if name == 'best_checkpoint':
        color = 'red'
        
    plt.plot(dates, pred, label=f'{name.upper()} Prediction', color=color, linewidth=2.5, alpha=0.9)
    plt.fill_between(dates, actual, pred, color='lightgray', alpha=0.5, label='Error Band')
    
    plt.title(f'96-Hour AQI Forecast: Actual vs {name.upper()}', fontsize=15, fontweight='bold')
    plt.xlabel('Time (UTC)', fontsize=12)
    plt.ylabel('US AQI', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    safe_name = name.replace(" ", "_")
    plt.savefig(f"model_comparison_plots/{safe_name}_vs_actual.png", dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()

print(f"{len([n for n in model_names if n in df_pred.columns])} plots saved in: model_comparison_plots/")
print("\nAll done!")
