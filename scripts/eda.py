# ============================================================================
# IMPORTS & GLOBAL SETTINGS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
import traceback

# NOTE: The HOPSWORKS_API_KEY should be set externally (e.g., via GitHub Secrets or environment variable).
#os.environ['HOPSWORKS_API_KEY'] = '...' # REMOVED HARDCODED KEY

# Handle hopsworks import gracefully
try:
    import hopsworks
except ImportError:
    print("Warning: hopsworks package not found. Feature store upload functionality will fail.")
    # Create mock hopsworks module for smooth execution of the file content
    class MockProject:
        def get_feature_store(self): return self
    class MockFG:
        def read(self): return pd.DataFrame()
        def filter(self, condition): return self
        def delete(self): pass
        def insert(self, df, wait=True): print("MOCK INSERT: Data inserted (not really).")
        def __init__(self):
            self.id = 1234
            self.features = []
            self.primary_key = []
            self.event_time = 'datetime_utc'
            self.description = 'Mock FG'
    class MockHopsworks:
        def login(self, api_key_value): 
            if api_key_value:
                return MockProject()
            raise ValueError("HOPSWORKS_API_KEY not provided.")
        def get_feature_store(self): return self
        def get_feature_group(self, name, version): return None
        def get_feature_groups(self, name): return []
        def create_feature_group(self, **kwargs): return MockFG()
    hopsworks = MockHopsworks()

# --- INPUT DATA FILE ---
# Assuming '2years_us_aqi_final.csv' is in the current directory as per the first code block.
df = pd.read_csv('data/2years_us_aqi_final.csv', parse_dates=['datetime_utc'])

# ============================================================================
# 1. DATA CLEANING
# ============================================================================
def clean_data(df):
    """
    Clean the dataset: duplicates, negatives, outliers (conservative), missing values.
    """
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    df_clean = df.copy()

    # 1. Remove duplicates by timestamp
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['datetime_utc'], keep='first')
    print(f"✅ Removed {initial_len - len(df_clean)} duplicates.")

    # 2. Clip negatives to 0 (pollutants can't be negative)
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    for col in pollutant_cols:
        if col in df_clean.columns:
            negatives = (df_clean[col] < 0).sum()
            if negatives > 0:
                print(f"⚠️ Clipped {negatives} negatives in {col} to 0.")
            df_clean[col] = df_clean[col].clip(lower=0)

    # 3. Conservative outlier removal (99.5th percentile cap instead of 3*IQR)
    for col in pollutant_cols:
        if col in df_clean.columns:
            p995 = df_clean[col].quantile(0.995)
            outliers = (df_clean[col] > p995).sum()
            if outliers > 0:
                print(f"⚠️ Capped {outliers} extreme outliers in {col} at 99.5th percentile ({p995:.1f}).")
            df_clean[col] = df_clean[col].clip(upper=p995)

    # 4. Missing values: Forward/backward fill (time-series appropriate), then median
    for col in pollutant_cols + ['us_aqi']:
        if col in df_clean.columns:
            missing = df_clean[col].isnull().sum()
            if missing > 0:
                # Use ffill/bfill for time series, then median for any gaps remaining at ends
                df_clean[col] = df_clean[col].ffill().bfill().fillna(df_clean[col].median())
                print(f"✅ Filled {missing} missing values in {col}.")

    print(f"\n✅ Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean

# Apply cleaning
df_clean = clean_data(df)

# Save cleaned data
df_clean.to_csv('data/2years_cleaned.csv', index=False)
print("💾 Saved to 'data/2years_cleaned.csv'")


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
# Reload cleaned data for safety, although df_clean is available
df_clean = pd.read_csv('data/2years_cleaned.csv', parse_dates=['datetime_utc'])

def engineer_features(df):
    """
    Add temporal, interaction features and TARGET LAGS (us_aqi_lag1, us_aqi_lag24).
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    df_feat = df.copy()

    # CRITICAL FIX: Ensure 'datetime_utc' is correctly cast as a timezone-aware datetime object.
    df_feat['datetime_utc'] = pd.to_datetime(df_feat['datetime_utc'], format='mixed', utc=True)
    print("✅ Ensured 'datetime_utc' is a timezone-aware datetime object.")


    # Sort by time (critical for time-series)
    df_feat = df_feat.sort_values('datetime_utc').reset_index(drop=True)

    # 1. Temporal features (cyclical encoding)
    df_feat['hour'] = df_feat['datetime_utc'].dt.hour
    df_feat['month'] = df_feat['datetime_utc'].dt.month
    df_feat['day_of_week'] = df_feat['datetime_utc'].dt.dayofweek
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)

    # Cyclical encoding (better for time patterns)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    print("✅ Added temporal features (hour, month cyclical encoding).")

    # 2. Pollutant interactions & ratios
    df_feat['pm_ratio'] = df_feat['pm2_5'] / (df_feat['pm10'] + 1e-6)
    df_feat['total_pm'] = df_feat['pm2_5'] + df_feat['pm10']

    gas_cols = ['co', 'no2', 'o3', 'so2']
    # Ensure all gas_cols exist before summing
    existing_gas_cols = [col for col in gas_cols if col in df_feat.columns]
    df_feat['total_gases'] = df_feat[existing_gas_cols].sum(axis=1)

    # Additional ratios
    df_feat['no2_o3_ratio'] = df_feat['no2'] / (df_feat['o3'] + 1e-6)
    df_feat['pm2_5_co_interaction'] = df_feat['pm2_5'] * df_feat['co']
    print("✅ Added pollutant interactions & ratios.")

    # 3. Rolling averages (short-term trends, NO lag of target)
    df_feat['pm2_5_rolling_3h'] = df_feat['pm2_5'].rolling(window=3, min_periods=1).mean()
    df_feat['pm10_rolling_3h'] = df_feat['pm10'].rolling(window=3, min_periods=1).mean()
    df_feat['co_rolling_3h'] = df_feat['co'].rolling(window=3, min_periods=1).mean()
    print("✅ Added 3-hour rolling averages for PM2.5, PM10, CO.")

    # =========================================================
    # 🎯 TARGET LAG FEATURES (CRUCIAL FOR FORECASTING)
    # =========================================================
    df_feat['us_aqi_lag1'] = df_feat['us_aqi'].shift(1)
    df_feat['us_aqi_lag24'] = df_feat['us_aqi'].shift(24)
    print("✅ Added target lag features (us_aqi_lag1, us_aqi_lag24).")

    # Clean up intermediate columns used only for feature calculation
    df_feat = df_feat.drop(columns=['hour', 'month', 'day_of_week'], errors='ignore')

    print(f"\n✅ Feature engineering complete. New shape: {df_feat.shape}")
    return df_feat

# Apply feature engineering
df_feat = engineer_features(df_clean)

# Save
df_feat.to_csv('data/2years_features.csv', index=False)
print("💾 Saved to 'data/2years_features.csv'")

# Preview key columns
print("\nPreview of engineered features:")
print(df_feat[['datetime_utc', 'us_aqi', 'us_aqi_lag1', 'us_aqi_lag24', 'pm2_5', 'total_pm',
              'hour_sin', 'month_sin', 'pm2_5_rolling_3h']].head(26))


# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) & FEATURE SELECTION
# ============================================================================

# Load features (reloading from saved CSV ensures integrity)
df_feat = pd.read_csv('data/2years_features.csv', parse_dates=['datetime_utc'])

# =================================================================
# FIX: Re-derive 'month' and 'hour' from datetime_utc for plotting
# =================================================================
# Ensure datetime is TZ-aware (as saved from previous step)
df_feat['datetime_utc'] = pd.to_datetime(df_feat['datetime_utc'], format='mixed', utc=True)
df_feat['hour'] = df_feat['datetime_utc'].dt.hour
df_feat['month'] = df_feat['datetime_utc'].dt.month
# =================================================================

print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS (UPDATED)")
print("="*50)

# 1. Basic statistics
print("\nAQI Statistics:")
print(df_feat['us_aqi'].describe())

# 2. Correlations (subset for key features)
features_for_corr = [
     'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2',
     'us_aqi_lag1', 'us_aqi_lag24', # LAGS ADDED
     'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
     'pm_ratio', 'total_pm', 'total_gases', 'no2_o3_ratio',
     'pm2_5_rolling_3h', 'pm10_rolling_3h', 'co_rolling_3h',
     'pm2_5_co_interaction', 'is_weekend',
     'us_aqi'
]

# Filter only existing columns
features_for_corr = [f for f in features_for_corr if f in df_feat.columns]

corr = df_feat[features_for_corr].corr()['us_aqi'].sort_values(ascending=False)
print("\n📊 Top Correlations with US AQI:")
print(corr.head(15))

# 3. Feature selection (Modified)
correlation_threshold = 0.3
selected_features = [f for f in features_for_corr if f != 'us_aqi' and abs(corr[f]) > correlation_threshold]

# Manually ensure crucial time features are included for tree models
manual_additions = ['us_aqi_lag1', 'us_aqi_lag24', 'hour_sin', 'hour_cos']
for feature in manual_additions:
    if feature not in selected_features and feature in df_feat.columns:
          selected_features.append(feature)

print(f"\n✅ Selected Features (Corr > {correlation_threshold} + Target Lags + Hour Cos/Sin): {len(selected_features)} features")
print(selected_features)

# Save selected features to 'selected_features.txt'
with open('selected_features.txt', 'w') as f:
    f.write(','.join(selected_features))
print("💾 Saved selected features to 'selected_features.txt'")

# CRITICAL ADDITION: Save selected features to 'data/selected_text.text'
os.makedirs('data', exist_ok=True) 
try:
    with open('data/selected_text.text', 'w') as f:
        f.write(','.join(selected_features))
    print("💾 Saved selected features to 'data/selected_text.text'")
except Exception as e:
    print(f"⚠️ Error saving to data/selected_text.text: {e}")
# END CRITICAL ADDITION


# 4. Visualization
plt.figure(figsize=(16, 10))

# Subplot 1: AQI Distribution
plt.subplot(2, 3, 1)
df_feat['us_aqi'].hist(bins=50, edgecolor='black', color='steelblue')
plt.title('AQI Distribution', fontsize=12, fontweight='bold')
plt.xlabel('US AQI')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# Subplot 2: AQI Over Time
plt.subplot(2, 3, 2)
plt.plot(df_feat['datetime_utc'], df_feat['us_aqi'], linewidth=0.5, color='darkred', alpha=0.7)
plt.title('AQI Over Time (2 Years)', fontsize=12, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('US AQI')
plt.grid(alpha=0.3)

# Subplot 3: AQI by Month
plt.subplot(2, 3, 3)
monthly_aqi = df_feat.groupby('month')['us_aqi'].mean()
plt.bar(monthly_aqi.index, monthly_aqi.values, color='teal', alpha=0.7)
plt.title('Average AQI by Month', fontsize=12, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average AQI')
plt.xticks(range(1, 13))
plt.grid(alpha=0.3)

# Subplot 4: Correlation Heatmap (Top 10 features)
plt.subplot(2, 3, 4)
top_features = corr.head(15).index.tolist()
if 'us_aqi' in top_features: top_features.remove('us_aqi')
top_features = ['us_aqi'] + top_features[:10] # Ensure AQI is first
# Filter df_feat for existing columns before calculating corr_matrix
top_features = [f for f in top_features if f in df_feat.columns]
corr_matrix = df_feat[top_features].corr()
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right', fontsize=8)
plt.yticks(range(len(top_features)), top_features, fontsize=8)
plt.title('Correlation Heatmap (Top Features)', fontsize=12, fontweight='bold')

# Subplot 5: PM2.5 vs AQI Scatter
plt.subplot(2, 3, 5)
plt.scatter(df_feat['pm2_5'], df_feat['us_aqi'], alpha=0.3, s=5, color='purple')
plt.title('PM2.5 vs US AQI', fontsize=12, fontweight='bold')
plt.xlabel('PM2.5 (µg/m³)')
plt.ylabel('US AQI')
plt.grid(alpha=0.3)

# Subplot 6: AQI by Hour (Average)
plt.subplot(2, 3, 6)
hourly_aqi = df_feat.groupby('hour')['us_aqi'].mean()
plt.plot(hourly_aqi.index, hourly_aqi.values, marker='o', color='orange', linewidth=2)
plt.title('Average AQI by Hour', fontsize=12, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Average AQI')
plt.xticks(range(0, 24, 3))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots_updated.png', dpi=150, bbox_inches='tight')
print("\n💾 EDA plots saved to 'eda_plots_updated.png'")

# Add AQI category and re-save features CSV
df_feat['aqi_category'] = pd.cut(df_feat['us_aqi'],
                                 bins=[0, 50, 100, 150, 200, 300, 500],
                                 labels=['Good', 'Moderate', 'Unhealthy-SG', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])

df_feat.to_csv('data/2years_features.csv', index=False)  # Re-save with category


# ============================================================================
# 4. HOPSWORKS FEATURE STORE INTEGRATION FUNCTIONS (Selected Features Only)
# ============================================================================

# Use the features selected in the EDA step
# This list is manually updated based on the common feature selection output.
SELECTED_FEATURES = [
    'pm2_5', 'pm10', 'co', 'no2', 'so2',
    'us_aqi_lag1', 'us_aqi_lag24',
    'month_cos', 'total_pm', 'total_gases',
    'no2_o3_ratio', 'pm2_5_rolling_3h', 'pm10_rolling_3h',
    'co_rolling_3h', 'pm2_5_co_interaction',
    'hour_sin', 'hour_cos'
]

def create_clean_feature_csv(input_path='data/2years_features.csv', output_path='data/2years_features_clean.csv'):
    """Prepares a CSV with only the required columns for Hopsworks upload."""
    print(f"\nCreating clean feature CSV: {output_path}")
    df = pd.read_csv(input_path)

    # Safely parse datetime
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True, format='mixed', errors='coerce')

    # Drop rows with invalid datetime
    if df['datetime_utc'].isna().any():
        print(f"Dropping {df['datetime_utc'].isna().sum()} rows with invalid datetime_utc")
        df = df.dropna(subset=['datetime_utc']).reset_index(drop=True)

    # Required columns (Target + Features + Time)
    required_cols = ['datetime_utc', 'us_aqi'] + SELECTED_FEATURES
    
    # Check for missing columns in the source DF and fill them if necessary
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns in source CSV → filling with 0: {missing}")
        for c in missing:
            df[c] = 0.0

    df_clean = df[required_cols].copy()
    df_clean = df_clean.sort_values('datetime_utc').reset_index(drop=True)

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved {len(df_clean)} rows × {len(df_clean.columns)} cols → {output_path}")
    print(f"Columns: {list(df_clean.columns)}")
    return df_clean


def validate_dataframe_for_hopsworks(df):
    """Pre-processes the DataFrame for Hopsworks, including timestamp conversion."""
    df_clean = df.copy()
    issues = []

    # 1. datetime_utc → ms int64
    if 'datetime_utc' in df_clean.columns:
        if not pd.api.types.is_datetime64tz_dtype(df_clean['datetime_utc']):
            df_clean['datetime_utc'] = pd.to_datetime(df_clean['datetime_utc'], utc=True, errors='coerce')
        
        # Drop out-of-range timestamps
        bad = (df_clean['datetime_utc'].isna() |
               (df_clean['datetime_utc'] < pd.Timestamp('1678-01-01', tz='UTC')) |
               (df_clean['datetime_utc'] > pd.Timestamp('2262-04-11', tz='UTC')))
        if bad.sum():
            print(f"Dropping {bad.sum()} rows with invalid timestamps")
            df_clean = df_clean[~bad].reset_index(drop=True)
            issues.append(f"Dropped {bad.sum()} invalid timestamps")
            
        # Convert to milliseconds since epoch
        df_clean['datetime_utc'] = (df_clean['datetime_utc'].astype('int64') // 1_000_000).astype('int64')
        issues.append("Converted datetime_utc to ms")

    # 2. Force numeric to float64 (Hopsworks stores as double)
    for c in df_clean.select_dtypes(include=[np.number]).columns:
        if c != 'datetime_utc':
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce').astype('float64')

    # 3. Fill NaN/inf
    df_clean = df_clean.fillna(0)
    df_clean = df_clean.replace([np.inf, -np.inf], 0)
    issues.append("Filled NaN / inf")

    print("\nVALIDATION COMPLETE")
    if issues:
        for i in issues: print(f"   • {i}")
    print(f"Shape: {df_clean.shape}")
    if 'datetime_utc' in df_clean.columns:
        mn = pd.to_datetime(df_clean['datetime_utc'].min(), unit='ms', utc=True)
        mx = pd.to_datetime(df_clean['datetime_utc'].max(), unit='ms', utc=True)
        print(f"Date range: {mn} → {mx}")
    return df_clean


def list_feature_groups(name="aqi_features"):
    print("\n" + "="*70)
    print(f"LISTING FEATURE GROUPS: {name}")
    print("="*70)
    try:
        if not os.getenv("HOPSWORKS_API_KEY"):
            print("HOPSWORKS_API_KEY not set. Cannot list feature groups.")
            return []
            
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        fgs = fs.get_feature_groups(name=name)
        if not fgs:
            print("No versions found")
            return []
        print(f"Found {len(fgs)} version(s):")
        for fg in fgs:
            print(f"\n  Version {fg.version}:")
            print(f"    Features: {len(fg.features)}")
            print(f"    Primary key: {fg.primary_key}")
            print(f"    Event time: {fg.event_time}")
            print(f"    Description: {fg.description}")
        return fgs
    except Exception as e:
        print(f"Error listing feature groups: {e}")
        return []


def delete_feature_group(name="aqi_features", version=4):
    print("\n" + "="*70)
    print(f"DELETING FEATURE GROUP: {name} v{version}")
    print("="*70)
    try:
        if not os.getenv("HOPSWORKS_API_KEY"):
            print("HOPSWORKS_API_KEY not set. Cannot delete feature group.")
            return

        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=name, version=version)
        if fg is not None:
            fg.delete()
            print(f"Deleted {name} v{version}")
        else:
            print("Already deleted or never existed")
    except Exception as e:
        print(f"Feature group not found or already deleted: {e}")


def upload_to_hopsworks(df, feature_group_name="aqi_features", version=4):
    print("\n" + "="*70)
    print("UPLOADING TO HOPSWORKS FEATURE STORE")
    print("="*70)
    try:
        if not os.getenv("HOPSWORKS_API_KEY"):
            print("HOPSWORKS_API_KEY environment variable not set. Skipping upload.")
            return

        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        print(f"Connected to project: {project.name}")
        fs = project.get_feature_store()

        # 1. Validate
        df_up = validate_dataframe_for_hopsworks(df)

        # Add id
        df_up.insert(0, 'id', range(1, len(df_up) + 1))
        print("Added 'id' primary key")

        # Deduplicate
        before = len(df_up)
        df_up = df_up.sort_values('datetime_utc').drop_duplicates(subset=['datetime_utc'], keep='last').reset_index(drop=True)
        if len(df_up) < before:
            print(f"Removed {before - len(df_up)} duplicate timestamps")
            df_up['id'] = range(1, len(df_up) + 1)

        print(f"\nUploading {len(df_up)} rows")

        # FG params
        fg_params = {
            "name": feature_group_name,
            "version": version,
            "primary_key": ["id"],
            "event_time": "datetime_utc",
            "description": "AQI prediction: 17 features + us_aqi + datetime_utc",
            "online_enabled": False,
        }

        # Get or create
        fg = None
        try:
            candidate = fs.get_feature_group(name=feature_group_name, version=version)
            if candidate is not None:
                fg = candidate
                print(f"Found existing FG v{version} – will append")
        except Exception:
            pass # Continue to create if not found
        
        if fg is None:
            print(f"Creating new feature group v{version}")
            fg = fs.create_feature_group(**fg_params)

        # Insert
        print("Inserting data...")
        fg.insert(df_up, wait=True)
        print(f"\nUPLOADED & MATERIALISED! {feature_group_name} v{version}")
        print(f"   Rows: {len(df_up)}")

        try:
            p_id = getattr(project, "project_id", "PROJECT_ID")
            fs_id = getattr(fs, "id", "FS_ID")
            fg_id = fg.id
            print(f"   View: https://c.app.hopsworks.ai/p/{p_id}/fs/{fs_id}/fg/{fg_id}")
        except Exception:
            pass

        return fg

    except Exception as e:
        print(f"\nUPLOAD FAILED: {e}")
        traceback.print_exc()
        raise


def fetch_from_hopsworks(feature_group_name="aqi_features", version=4,
                         for_training=False, start_date=None, end_date=None,
                         max_retries=5, retry_delay=30):
    print("\n" + "="*70)
    print("FETCHING FROM HOPSWORKS" if not for_training else "FETCHING TRAINING DATA")
    print("="*70)

    for attempt in range(max_retries):
        try:
            if not os.getenv("HOPSWORKS_API_KEY"):
                raise ValueError("HOPSWORKS_API_KEY environment variable not set.")

            project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name=feature_group_name, version=version)

            cols_to_select = ['id', 'datetime_utc', 'us_aqi'] + SELECTED_FEATURES
            query = fg.select(cols_to_select)

            if start_date and end_date:
                print(f"Filtering {start_date} → {end_date}")
                s = int(pd.Timestamp(start_date, tz='UTC').value // 1_000_000)
                e = int(pd.Timestamp(end_date, tz='UTC').value // 1_000_000)
                df = query.filter((fg.datetime_utc >= s) & (fg.datetime_utc <= e)).read()
            else:
                df = query.read()

            # Post-process timestamp and ID
            if 'datetime_utc' in df.columns:
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', utc=True)
            if 'id' in df.columns:
                df = df.drop('id', axis=1)

            print(f"Fetched! Shape: {df.shape}")
            if 'datetime_utc' in df.columns:
                print(f"Date range: {df['datetime_utc'].min()} → {df['datetime_utc'].max()}")
            return df

        except Exception as e:
            if "hoodie.properties" in str(e) and attempt < max_retries - 1:
                print(f"Materialisation not ready (attempt {attempt+1}/{max_retries}). Waiting {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"Fetch failed: {e}")
                # ---------- FALLBACK ----------
                print("Falling back to clean CSV...")
                try:
                    df = pd.read_csv('data/2years_features_clean.csv')
                    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
                    df = df[['datetime_utc', 'us_aqi'] + SELECTED_FEATURES]
                    print(f"Fallback CSV shape: {df.shape}")
                    return df
                except Exception as fb:
                    print(f"Fallback failed: {fb}")
                    raise


def compare_datasets(local_df, hops_df, tolerance=1e-6):
    print("\n" + "="*70)
    print("DATASET COMPARISON")
    print("="*70)

    cols = ['datetime_utc', 'us_aqi'] + SELECTED_FEATURES
    local = local_df[cols].sort_values('datetime_utc').reset_index(drop=True)
    hops = hops_df[cols].sort_values('datetime_utc').reset_index(drop=True)

    print(f"Date range local: {local['datetime_utc'].min()} → {local['datetime_utc'].max()}")
    print(f"Date range hops : {hops['datetime_utc'].min()} → {hops['datetime_utc'].max()}")

    if len(local) != len(hops):
        print(f"Row count mismatch: local={len(local)} hops={len(hops)}")
        min_len = min(len(local), len(hops))
        local, hops = local.iloc[:min_len], hops.iloc[:min_len]

    numeric_cols = local.select_dtypes(include=[np.number]).columns
    # Align dtypes for reliable comparison
    for c in numeric_cols:
         if hops[c].dtype != local[c].dtype:
             hops[c] = hops[c].astype(local[c].dtype)
             
    diffs = (local[numeric_cols] - hops[numeric_cols]).abs()
    max_diff = diffs.max().max()

    print(f"Max numeric difference: {max_diff:.2e}")
    print("DATA MATCH!" if max_diff < tolerance else "DATA MISMATCH!")

    if 'us_aqi' in local.columns:
        print("\nus_aqi stats:")
        for stat in ['mean', 'std', 'min', 'max']:
            lv = getattr(local['us_aqi'], stat)()
            hv = getattr(hops['us_aqi'], stat)()
            diff = abs(lv - hv)
            print(f"  {stat:4}: local={lv:8.2f} hops={hv:8.2f} diff={diff:8.2f}")

    print("\n" + "="*70)
    return max_diff < tolerance


# ============================================================================
# FINAL PIPELINE EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # -----------------------------------------------------
    # RUN CLEANING AND FEATURE ENGINEERING TO GENERATE FILES
    # -----------------------------------------------------
    # Clean data (Saves '2years_cleaned.csv')
    df_clean_data = clean_data(df)

    # Engineer features (Saves '2years_features.csv')
    df_feat_data = engineer_features(df_clean_data)

    # Run EDA/Feature Selection (Saves 'selected_features.txt' and 'data/selected_text.text')
    # NOTE: The selected_features list in memory is now up-to-date.
    # The EDA code block handles loading/re-deriving intermediate columns for plotting.
    
    # Reload the full feature DF including the newly added 'aqi_category' column.
    df_full_features = pd.read_csv('data/2years_features.csv', parse_dates=['datetime_utc'])
    df_full_features['datetime_utc'] = pd.to_datetime(df_full_features['datetime_utc'], utc=True)
    
    # -----------------------------------------------------
    # HOPSWORKS PIPELINE
    # -----------------------------------------------------
    FG_VERSION = 4
    CLEAN_CSV = 'data/2years_features_clean.csv'
    
    # 0. List existing FGs
    list_feature_groups("aqi_features")

    # 1. Delete old version for idempotency
    delete_feature_group("aqi_features", version=FG_VERSION)

    # 2. Prepare clean data (Saves '2years_features_clean.csv' with only SELECTED_FEATURES)
    df_local = create_clean_feature_csv(input_path='data/2years_features.csv', output_path=CLEAN_CSV)

    # 3. Upload to Hopsworks
    upload_to_hopsworks(df_local, version=FG_VERSION)

    # 4. Fetch & Compare
    hops_df = fetch_from_hopsworks("aqi_features", version=FG_VERSION)
    if hops_df is not None:
        compare_datasets(df_local, hops_df)

    # 5. Training Data Fetch
    if 'datetime_utc' in df_local.columns:
        training_df = fetch_from_hopsworks(
            "aqi_features", version=FG_VERSION, for_training=True,
            start_date=df_local['datetime_utc'].min().isoformat(),
            end_date=df_local['datetime_utc'].max().isoformat()
        )
        if training_df is not None:
            print(f"\nTRAINING DATA READY: {training_df.shape}")
            print(f"Columns: {list(training_df.columns)}")
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)
