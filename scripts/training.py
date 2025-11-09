# =============================================================================
# AQI TRAINING AND FORECASTING PIPELINE (training.py)
# =============================================================================
import pandas as pd
import numpy as np
import joblib
import hopsworks
import os
import json
import requests
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
# Note: HOPSWORKS_API_KEY is read from environment variable or set above
API_KEY = "29e4f8ef9151633260fb36745ed19012"  # OWM API Key for Forecast
LAT = 24.8607
LON = 67.0011
HISTORIC_PATH = "data/2years_features_clean.csv"
MODEL_ARTIFACTS_DIR = "model_artifacts"
PLOTS_DIR = "model_comparison_plots"

# Features used for training (Lag features were excluded in the new list)
SELECTED_FEATURES = [
    'pm2_5', 'pm10', 'co', 'no2', 'so2',
    'month_cos', 'total_pm', 'total_gases',
    'no2_o3_ratio', 'pm2_5_rolling_3h', 'pm10_rolling_3h',
    'co_rolling_3h', 'pm2_5_co_interaction',
    'hour_sin', 'hour_cos'
]

# =============================================================================
# PART 1: MODEL TRAINING AND CHECKPOINTING
# =============================================================================

# ----------------------------------------------------------------------
# 1. FETCH DATA
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("PART 1: MODEL TRAINING AND CHECKPOINTING")
print("FETCHING TRAINING DATA")
print("="*80)

def fetch_training_data():
    """Fetches data from Hopsworks Feature Store or falls back to CSV."""
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_features", version=4)
        
        # Ensure 'us_aqi' and 'datetime_utc' are included for training/splitting
        cols_to_fetch = ['datetime_utc', 'us_aqi'] + SELECTED_FEATURES
        
        df = fg.select(cols_to_fetch).read()
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', utc=True)
        print(f"Fetched from Hopsworks: {df.shape}")
        return df
    except Exception as e:
        print(f"Hopsworks failed: {e}. Falling back to CSV.")
        df = pd.read_csv(HISTORIC_PATH)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
        # Ensure features are present, fill missing with 0
        df = df[['datetime_utc', 'us_aqi'] + SELECTED_FEATURES].fillna(0)
        print(f"Fallback CSV: {df.shape}")
        return df

df = fetch_training_data()
df = df.sort_values('datetime_utc').reset_index(drop=True)

# ----------------------------------------------------------------------
# 2. DATA PREPARATION (NO LAGS USED IN X)
# ----------------------------------------------------------------------
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

os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
joblib.dump(scaler, f"{MODEL_ARTIFACTS_DIR}/scaler.pkl")

# ----------------------------------------------------------------------
# 5. METRICS FUNCTION
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
    """Trains a model with TimeSeries CV and fits the final model."""
    X_tr = X_train_scaled if use_scaled else X_train
    X_va = X_val_scaled   if use_scaled else X_val
    X_te = X_test_scaled  if use_scaled else X_test

    # --- 5-fold CV ---
    cv_mae = []
    for tr_idx, val_idx in tscv.split(X_tr):
        X_f_tr, X_f_va = X_tr[tr_idx], X_tr[val_idx]
        y_f_tr, y_f_va = y_train[tr_idx], y_train[val_idx]

        m = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
        
        if name == "xgboost":
            # XGBoost training for CV fold
            dtrain = xgb.DMatrix(X_f_tr, label=y_f_tr)
            dval   = xgb.DMatrix(X_f_va, label=y_f_va)
            params = model.get_xgb_params()
            bst = xgb.train(
                params, dtrain, num_boost_round=model.n_estimators, evals=[(dval, "eval")],
                early_stopping_rounds=50, verbose_eval=False
            )
            pred = bst.predict(dval)
        elif name == "lightgbm":
            # LightGBM training for CV fold
            m.fit(X_f_tr, y_f_tr, eval_set=[(X_f_va, y_f_va)], 
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred = m.predict(X_f_va)
        else:
            # All other models (RF, GB, Ridge, Linear, CatBoost)
            m.fit(X_f_tr, y_f_tr)
            pred = m.predict(X_f_va)

        cv_mae.append(mean_absolute_error(y_f_va, pred))

    cv_mae_mean = np.mean(cv_mae)
    print(f"{name:12} CV MAE: {cv_mae_mean:.3f}")

    # --- Final fit on full train set ---
    final_model = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
    
    if name == "xgboost":
        # Final XGBoost fit
        dtrain_full = xgb.DMatrix(X_tr, label=y_train)
        dval_full   = xgb.DMatrix(X_va, label=y_val)
        params = model.get_xgb_params()
        final_model = xgb.train(
            params, dtrain_full, num_boost_round=model.n_estimators, evals=[(dval_full, "eval")],
            early_stopping_rounds=50, verbose_eval=False
        )
    elif name == "lightgbm":
        # Final LightGBM fit
        final_model.fit(X_tr, y_train, eval_set=[(X_va, y_val)], 
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    else:
        # Final fit for all others
        final_model.fit(X_tr, y_train)

    # --- Predictions for metrics ---
    if name == "xgboost":
        pred_train = final_model.predict(xgb.DMatrix(X_tr))
        pred_val   = final_model.predict(xgb.DMatrix(X_va))
        pred_test  = final_model.predict(xgb.DMatrix(X_te))
    else:
        pred_train = final_model.predict(X_tr)
        pred_val   = final_model.predict(X_va)
        pred_test  = final_model.predict(X_te)

    # Store metrics
    all_metrics['train'][name] = calc_metrics(y_train, pred_train, f"{name} Train")
    all_metrics['val'][name]   = calc_metrics(y_val,   pred_val,   f"{name} Val")
    all_metrics['test'][name]  = calc_metrics(y_test,  pred_test,  f"{name} Test")
    all_metrics['cv_mae'][name] = cv_mae_mean

    # Save this model
    model_path = f"{MODEL_ARTIFACTS_DIR}/model_{name}.pkl"
    joblib.dump(final_model, model_path)
    print(f"Saved: {model_path}")

    # Store in dict
    models[name] = final_model
    cv_scores[name] = cv_mae_mean

# --------------------- TRAIN ALL MODELS ---------------------

xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.7, 
    colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1
)
train_with_cv(xgb_model, "xgboost")

lgb_model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.7, 
    colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1
)
train_with_cv(lgb_model, "lightgbm")

cb_model = cb.CatBoostRegressor(
    iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=False
)
train_with_cv(cb_model, "catboost")

rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_split=20, min_samples_leaf=10, 
    random_state=42, n_jobs=-1
)
train_with_cv(rf_model, "rf")

gb_model = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, 
    min_samples_split=20, min_samples_leaf=10, random_state=42
)
train_with_cv(gb_model, "gb")

ridge_model = Ridge(alpha=10.0, random_state=42)
train_with_cv(ridge_model, "ridge", use_scaled=True)

lr_model = LinearRegression()
train_with_cv(lr_model, "linear", use_scaled=True)

# ----------------------------------------------------------------------
# 8. BEST MODEL AND CHECKPOINT (Minimal logic for current run only)
# ----------------------------------------------------------------------
best_name = min(cv_scores, key=cv_scores.get)
best_mod  = models[best_name]

print("\n" + "="*80)
print(f"BEST MODEL (CURRENT RUN): {best_name.upper()} | CV MAE: {cv_scores[best_name]:.3f}")
print("="*80)

# Save best model separately for inference (uses the name in the path)
best_model_path = f"{MODEL_ARTIFACTS_DIR}/best_model_{best_name}.pkl"
joblib.dump(best_mod, best_model_path)
print(f"Best model also saved: {best_model_path}")

# Final test prediction with best model for summary
X_test_fin = X_test_scaled if best_name in ["ridge", "linear"] else X_test
if best_name == "xgboost":
    y_test_pred = best_mod.predict(xgb.DMatrix(X_test_fin))
else:
    y_test_pred = best_mod.predict(X_test_fin)
test_metrics = calc_metrics(y_test, y_test_pred, "FINAL TEST")

# ----------------------------------------------------------------------
# 9. SAVE ALL METRICS & CONFIG
# ----------------------------------------------------------------------
joblib.dump(SELECTED_FEATURES, f"{MODEL_ARTIFACTS_DIR}/selected_features.pkl")

with open(f"{MODEL_ARTIFACTS_DIR}/metrics.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "cv_mae": cv_scores,
        "test_mae_all": {name: all_metrics['test'][name]['mae'] for name in models},
        "best_test": test_metrics,
        "features": SELECTED_FEATURES,
        "scaler_needed": best_name in ["ridge", "linear"],
        "all_models_saved": [f"model_{name}.pkl" for name in models],
        "all_metrics": all_metrics
    }, f, indent=2)

print(f"All metrics saved to {MODEL_ARTIFACTS_DIR}/metrics.json")

# ----------------------------------------------------------------------
# 10. FINAL SUMMARY
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("TRAINING COMPLETE – ALL MODELS SAVED")
print("="*80)
print(f"Best Model          : {best_name.upper()}")
print(f"CV MAE              : {cv_scores[best_name]:.3f}")
print(f"Test MAE            : {test_metrics['mae']:.3f}")
print(f"Test R²             : {test_metrics['r2']:.3f}")
print("="*80)

# =============================================================================
# PART 2: 96-HOUR AQI FORECAST
# =============================================================================
print("\n" + "#"*80)
print("PART 2: 96-HOUR AQI FORECAST")
print("#"*80)

# ----------------------------------------------------------------------
# 0. CONFIG AND MODEL LOADING
# ----------------------------------------------------------------------
MODEL_PATHS = {f"{name}": f"{MODEL_ARTIFACTS_DIR}/model_{name}.pkl" for name in models.keys()}
SCALER_PATH = f"{MODEL_ARTIFACTS_DIR}/scaler.pkl"
FEATURES_PATH = f"{MODEL_ARTIFACTS_DIR}/selected_features.pkl"

# Load scaler & features
scaler = joblib.load(SCALER_PATH)
SELECTED_FEATURES = joblib.load(FEATURES_PATH)

# Load models
forecast_models = {}
for name, path in MODEL_PATHS.items():
    try:
        forecast_models[name] = joblib.load(path)
    except Exception as e:
        print(f"Failed to load {name}: {e}")

print(f"Loaded {len(forecast_models)} models for forecasting.")

# ----------------------------------------------------------------------
# 1. US-AQI CONVERSION (Copied from your provided code)
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
    if pollutant == 'co':  return c / 1145.0
    if pollutant == 'o3':  return c / 1960.6
    if pollutant == 'no2':  return c / 1881.1
    if pollutant == 'so2':  return c / 2620.0
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
def get_pollution_forecast(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"API error: {resp.text}")
    data = resp.json()
    rows = []
    for item in data['list']:
        dt = datetime.fromtimestamp(item['dt'], tz=timezone.utc)
        comp = item['components']
        rows.append({
            'datetime_utc': dt,
            'pm2_5': comp['pm2_5'], 'pm10': comp['pm10'], 'co': comp['co'],
            'no2': comp['no2'], 'o3': comp['o3'], 'so2': comp['so2']
        })
    df = pd.DataFrame(rows)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    return df.head(96)

print("Fetching 96-hour forecast...")
df_forecast = get_pollution_forecast(LAT, LON, API_KEY)
df_forecast['Actual_AQI'] = df_forecast.apply(calc_us_aqi, axis=1).round().astype(int).clip(0, 500)

# ----------------------------------------------------------------------
# 3. LOAD HISTORIC DATA
# ----------------------------------------------------------------------
print(f"Loading historic data from {HISTORIC_PATH}...")
df_hist = pd.read_csv(HISTORIC_PATH)
df_hist['datetime_utc'] = pd.to_datetime(df_hist['datetime_utc'], utc=True, errors='coerce')
df_hist = df_hist.dropna(subset=['datetime_utc', 'us_aqi'])
df_hist = df_hist.sort_values('datetime_utc').reset_index(drop=True)
last_3 = df_hist.tail(3)

# ----------------------------------------------------------------------
# 4. BUILD FEATURES (Matching the selected features for inference)
# ----------------------------------------------------------------------
def build_features(df_fc, last_hist):
    """Builds the 15 selected features for the forecast dataframe."""
    df = df_fc.copy()
    
    # Temporal features
    df['hour'] = df['datetime_utc'].dt.hour
    df['month'] = df['datetime_utc'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Interaction/Derived features
    df['total_pm'] = df['pm2_5'] + df['pm10']
    df['total_gases'] = df['co'] + df['no2'] + df['o3'] + df['so2']
    df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
    df['pm2_5_co_interaction'] = df['pm2_5'] * df['co']

    # Rolling means (using last 3 historic points + forecast data)
    combined = pd.concat([last_hist[['pm2_5', 'pm10', 'co']], df[['pm2_5', 'pm10', 'co']]], ignore_index=True)
    
    # Calculate rolling means on combined, then take only the last 96 forecast rows
    df['pm2_5_rolling_3h'] = combined['pm2_5'].rolling(3, min_periods=1).mean().iloc[-len(df):].values
    df['pm10_rolling_3h']  = combined['pm10'].rolling(3, min_periods=1).mean().iloc[-len(df):].values
    df['co_rolling_3h']    = combined['co'].rolling(3, min_periods=1).mean().iloc[-len(df):].values

    df = df.drop(columns=['hour', 'month'], errors='ignore')
    
    # Fill any remaining NaNs (should only be at the very start of the rolling window)
    df = df.fillna(0)
    
    return df

df_fc_feat = build_features(df_forecast, last_3)
X_df = df_fc_feat[SELECTED_FEATURES].fillna(0)
X = X_df.values

# ----------------------------------------------------------------------
# 5. PREDICT WITH ALL MODELS
# ----------------------------------------------------------------------
from xgboost import DMatrix

predictions = {'datetime': df_fc_feat['datetime_utc'], 'Actual_AQI': df_fc_feat['Actual_AQI']}
model_names = list(forecast_models.keys())

for name in model_names:
    print(f"Predicting with {name}...")
    model = forecast_models[name]
    
    X_input = X_df.values # Default to unscaled numpy array

    try:
        if name == "xgboost":
            dmat = DMatrix(X_input, feature_names=SELECTED_FEATURES)
            pred = model.predict(dmat)
        elif name in ["lightgbm", "catboost", "rf", "gb"]:
            pred = model.predict(X_input)
        elif name in ["ridge", "linear"]:
            # Scale for linear models
            pred = model.predict(scaler.transform(X_input))
        else:
            raise ValueError(f"Unknown model name: {name}")

        # Clip predictions to valid AQI range and round
        pred = np.clip(np.round(pred).astype(int), 0, 500)
        predictions[name] = pred
    except Exception as e:
        print(f"Prediction error for {name}: {e}")
        predictions[name] = np.full(len(X_df), np.nan)

# Closest Model (based on error vs OWM's calculated AQI)
abs_errors = np.abs(np.array([predictions[m] for m in model_names]) - predictions['Actual_AQI'].values)
closest_idx = np.argmin(abs_errors, axis=0)
predictions['Closest_Model'] = [model_names[i] for i in closest_idx]

# ----------------------------------------------------------------------
# 6. SAVE future_aqi_predictions.csv
# ----------------------------------------------------------------------
cols = ['datetime', 'Actual_AQI'] + model_names + ['Closest_Model']
df_pred = pd.DataFrame(predictions)[cols]
df_pred.to_csv("future_aqi_predictions.csv", index=False)
print("Saved: future_aqi_predictions.csv")

# ----------------------------------------------------------------------
# 7. METRICS → future_prediction_comparison.csv
# ----------------------------------------------------------------------
y_true = df_pred['Actual_AQI'].values
metrics_list = []

for name in model_names:
    y_pred = df_pred[name].values
    # Filter out NaNs if prediction failed for a model
    valid_mask = ~np.isnan(y_pred)
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]
    
    if len(y_true_v) > 0:
        mae = mean_absolute_error(y_true_v, y_pred_v)
        rmse = np.sqrt(mean_squared_error(y_true_v, y_pred_v))
        r2 = r2_score(y_true_v, y_pred_v)
        mape = np.mean(np.abs((y_true_v - y_pred_v) / (y_true_v + 1e-6))) * 100
    else:
        mae, rmse, r2, mape = np.nan, np.nan, np.nan, np.nan
        
    metrics_list.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R²": round(r2, 3),
        "MAPE": round(mape, 2)
    })

df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv("future_prediction_comparison.csv", index=False)
print("Saved: future_prediction_comparison.csv")
print("\n" + df_metrics.to_string(index=False))

# ----------------------------------------------------------------------
# 8. PLOT — INDIVIDUAL MODEL GRAPHS
# ----------------------------------------------------------------------
print("Generating individual model comparison plots...")
os.makedirs(PLOTS_DIR, exist_ok=True)

actual = df_pred['Actual_AQI'].values
dates = df_pred['datetime']

for name in model_names:
    pred = df_pred[name].values
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label='Actual AQI (OWM)', color='black', linewidth=2.5, marker='o', markersize=3)
    plt.plot(dates, pred, label=f'{name.upper()} Prediction', color='teal', linewidth=2.5, alpha=0.9)
    plt.fill_between(dates, actual, pred, color='lightgray', alpha=0.5, label='Error Band')

    plt.title(f'96-Hour AQI Forecast: Actual vs {name.upper()}', fontsize=15, fontweight='bold')
    plt.xlabel('Time (UTC)', fontsize=12)
    plt.ylabel('US AQI', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_name = name.replace(" ", "_")
    plt.savefig(f"{PLOTS_DIR}/{safe_name}_vs_actual.png", dpi=200, bbox_inches='tight')
    plt.close()

print(f"{len(model_names)} individual plots saved in: {PLOTS_DIR}/")
print("\nAll done!")
