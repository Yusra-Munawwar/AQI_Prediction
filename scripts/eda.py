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

# Optional: Set this environment variable externally or directly here
# os.environ['HOPSWORKS_API_KEY'] = 'vuBdrdFVzNRmWkGY.yHZNYVXLB7UwT7UH4xnCbe2jHcEMW67hrJPewZrJqXUMs5RbseOBAAUuDHO991af'
# Note: hopsworks import is typically handled internally by the methods, 
# but if run in a non-notebook environment, the import is needed.
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
        def login(self, api_key_value): return MockProject()
        def get_feature_store(self): return self
        def get_feature_group(self, name, version): return None
        def get_feature_groups(self, name): return []
        def create_feature_group(self, **kwargs): return MockFG()
    hopsworks = MockHopsworks()


# Load data with AQI (from Section 1)
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
                print(f"⚠️  Clipped {negatives} negatives in {col} to 0.")
            df_clean[col] = df_clean[col].clip(lower=0)

    # 3. Conservative outlier removal (99.5th percentile cap instead of 3*IQR)
    for col in pollutant_cols:
        if col in df_clean.columns:
            p995 = df_clean[col].quantile(0.995)
            outliers = (df_clean[col] > p995).sum()
            if outliers > 0:
                print(f"⚠️  Capped {outliers} extreme outliers in {col} at 99.5th percentile ({p995:.1f}).")
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

# Save selected features for later use
with open('selected_features.txt', 'w') as f:
    f.write(','.join(selected_features))
print("💾 Saved selected features to 'selected_features.txt'")

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
# 4. HOPSWORKS FEATURE STORE INTEGRATION FUNCTIONS
# ============================================================================

def validate_dataframe_for_hopsworks(df):
    df_clean = df.copy()
    issues = []

    # 1. Multi-index handling
    if isinstance(df_clean.index, pd.MultiIndex):
        df_clean = df_clean.reset_index(drop=True)
        issues.append("Reset MultiIndex")
    if isinstance(df_clean.columns, pd.MultiIndex):
        df_clean.columns = ['_'.join(map(str, col)).strip() for col in df_clean.columns.values]
        issues.append("Flattened MultiIndex columns")
    if df_clean.columns.duplicated().any():
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        issues.append("Removed duplicate columns")

    # 2. day_of_week (must be added **before** timestamp conversion)
    if 'day_of_week' not in df_clean.columns and 'datetime_utc' in df_clean.columns:
        if df_clean['datetime_utc'].dtype != 'datetime64[ns]':
            df_clean['datetime_utc'] = pd.to_datetime(df_clean['datetime_utc'], errors='coerce', utc=True)
        df_clean['day_of_week'] = df_clean['datetime_utc'].dt.dayofweek
        issues.append("Added 'day_of_week' column")
        print("Added 'day_of_week' (0=Mon, 6=Sun)")

    # 3. datetime_utc → ms int64
    if 'datetime_utc' in df_clean.columns:
        if df_clean['datetime_utc'].dtype != 'datetime64[ns]':
            df_clean['datetime_utc'] = pd.to_datetime(df_clean['datetime_utc'], errors='coerce', utc=True)
        # drop out-of-range timestamps
        bad = (df_clean['datetime_utc'].isna() |
               (df_clean['datetime_utc'] < pd.Timestamp('1678-01-01', tz='UTC')) |
               (df_clean['datetime_utc'] > pd.Timestamp('2262-04-11', tz='UTC')))
        if bad.sum():
            print(f"Dropping {bad.sum()} rows with invalid timestamps")
            df_clean = df_clean[~bad].reset_index(drop=True)
            issues.append(f"Dropped {bad.sum()} invalid timestamps")
        df_clean['datetime_utc'] = (df_clean['datetime_utc'].astype('int64') // 1_000_000).astype('int64')
        issues.append("Converted datetime_utc to ms timestamp")

    # 4. force numeric columns to float64 (Hopsworks stores everything as double)
    int_cols = ['us_aqi', 'hour', 'month', 'day_of_week', 'is_weekend']
    for c in int_cols:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce').fillna(0).astype('float64')
            issues.append(f"Converted '{c}' → float64")

    # 5. categorical / object → string
    for c in df_clean.select_dtypes('category').columns:
        df_clean[c] = df_clean[c].astype(str)
    for c in df_clean.select_dtypes('object').columns:
        df_clean[c] = df_clean[c].fillna('unknown').astype(str)

    # 6. remaining numeric → float64
    for c in df_clean.select_dtypes(include=[np.number]).columns:
        if c not in int_cols and c != 'datetime_utc':
            df_clean[c] = df_clean[c].astype('float64')

    # 7. NaN / inf
    df_clean = df_clean.fillna({c: 0 for c in df_clean.select_dtypes(include=[np.number]).columns})
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
    issues.append("Filled NaN / inf")

    # 8. sanitise column names
    bad = [c for c in df_clean.columns if not c.replace('_', '').isalnum()]
    for c in bad:
        new = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in c)
        df_clean = df_clean.rename(columns={c: new})
    if bad:
        issues.append("Sanitised column names")

    print("\nVALIDATION COMPLETE")
    if issues:
        for i in issues: print(f"   • {i}")
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
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        fgs = fs.get_feature_groups(name=name)
        if not fgs:
            print("No versions found")
            return []
        print(f"Found {len(fgs)} version(s):")
        for fg in fgs:
            print(f"\n   Version {fg.version}:")
            print(f"     Features: {len(fg.features)}")
            print(f"     Primary key: {fg.primary_key}")
            print(f"     Event time: {fg.event_time}")
            print(f"     Description: {fg.description}")
        return fgs
    except Exception as e:
        print(f"Error: {e}")
        return []


def delete_feature_group(name="aqi_features", version=4):
    print("\n" + "="*70)
    print(f"DELETING FEATURE GROUP: {name} v{version}")
    print("="*70)
    try:
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
        # Check for API key presence
        if not os.getenv("HOPSWORKS_API_KEY"):
            print("HOPSWORKS_API_KEY environment variable not set. Skipping upload.")
            return

        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        print(f"Connected to project: {project.name}")
        fs = project.get_feature_store()

        # -------------------------------------------------
        # 1. Validate & prepare dataframe
        # -------------------------------------------------
        df_up = validate_dataframe_for_hopsworks(df)

        if 'id' not in df_up.columns:
            # Create a simple unique ID for the primary key
            df_up.insert(0, 'id', range(1, len(df_up) + 1))
            print("Added 'id' primary key")

        if 'datetime_utc' in df_up.columns:
            before = len(df_up)
            # Remove duplicates by timestamp, keeping the last (most recent) measurement
            df_up = (df_up.sort_values('datetime_utc')
                          .drop_duplicates(subset=['datetime_utc'], keep='last')
                          .reset_index(drop=True))
            if len(df_up) < before:
                print(f"Removed {before - len(df_up)} duplicate timestamps")
                df_up['id'] = range(1, len(df_up) + 1) # Re-index IDs after dropping duplicates

        print(f"\nUploading {len(df_up)} rows")
        print(f"Columns: {list(df_up.columns)}")

        # -------------------------------------------------
        # 2. FG definition
        # -------------------------------------------------
        fg_params = {
            "name": feature_group_name,
            "version": version,
            "primary_key": ["id"],
            "description": "AQI features with pollutants, temporal features, interactions. Target: us_aqi",
            "online_enabled": False,
        }
        if 'datetime_utc' in df_up.columns:
            fg_params["event_time"] = "datetime_utc"

        # -------------------------------------------------
        # 3. CRITICAL: Get or CREATE
        # -------------------------------------------------
        fg = None
        try:
            candidate = fs.get_feature_group(name=feature_group_name, version=version)
            if candidate is not None:
                fg = candidate
                print(f"Found existing FG v{version} – will append")
            else:
                # If get_feature_group returns None or raises an error indicating not found, create it
                print(f"Creating new feature group v{version}")
                fg = fs.create_feature_group(**fg_params)
        except Exception as e:
            # This catch handles specific exceptions if the FG doesn't exist.
            print(f"Attempt to retrieve FG failed ({e}) → will create new")
            fg = fs.create_feature_group(**fg_params)


        # -------------------------------------------------
        # 4. INSERT + WAIT
        # -------------------------------------------------
        print("Inserting data and waiting for materialisation job to finish...")
        # Ensure the insertion column order matches the feature group schema
        fg.insert(df_up, wait=True)
        print(f"\nUPLOADED & MATERIALISED! {feature_group_name} v{version}")
        print(f"   Rows: {len(df_up)}")

        # URL
        try:
            p_id = getattr(project, "project_id", "PROJECT_ID")
            fs_id = getattr(fs, "id", "FS_ID")
            fg_id = fg.id
            print(f"   View: https://c.app.hopsworks.ai/p/{p_id}/fs/{fs_id}/fg/{fg_id}")
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
            # Check for API key presence
            if not os.getenv("HOPSWORKS_API_KEY"):
                raise ValueError("HOPSWORKS_API_KEY environment variable not set.")

            project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name=feature_group_name, version=version)

            if start_date and end_date:
                print(f"Filtering {start_date} → {end_date}")
                # Convert ISO 8601 strings to milliseconds timestamp (int64) for filtering
                s = int(pd.Timestamp(start_date, tz='UTC').value // 1_000_000)
                e = int(pd.Timestamp(end_date, tz='UTC').value // 1_000_000)
                df = fg.filter((fg.datetime_utc >= s) & (fg.datetime_utc <= e)).read()
            else:
                df = fg.read()

            # post-process
            if 'datetime_utc' in df.columns:
                # Convert back from ms timestamp (int64) to datetime object
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', utc=True)
                bad = (df['datetime_utc'].isna() |
                       (df['datetime_utc'] < pd.Timestamp('1678-01-01', tz='UTC')) |
                       (df['datetime_utc'] > pd.Timestamp('2262-04-11', tz='UTC')))
                if bad.sum():
                    print(f"Dropping {bad.sum()} invalid timestamps")
                    df = df[~bad].reset_index(drop=True)

            int_cols = ['us_aqi', 'hour', 'month', 'day_of_week', 'is_weekend']
            for c in int_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int64')

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
                print("Falling back to local CSV...")
                try:
                    df = pd.read_csv('data/2years_features.csv')
                    if 'datetime_utc' in df.columns:
                        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce', utc=True)
                    if 'day_of_week' not in df.columns and 'datetime_utc' in df.columns:
                        df['day_of_week'] = df['datetime_utc'].dt.dayofweek
                    print(f"Fallback CSV shape: {df.shape}")
                    return df
                except Exception as fb:
                    print(f"Fallback also failed: {fb}")
                    raise


def compare_datasets(local_df, hops_df, tolerance=1e-6):
    print("\n" + "="*70)
    print("DATASET COMPARISON")
    print("="*70)

    # add day_of_week if missing
    for d, name in [(local_df, 'local'), (hops_df, 'hops')]:
        if 'day_of_week' not in d.columns and 'datetime_utc' in d.columns:
            d['day_of_week'] = d['datetime_utc'].dt.dayofweek
            print(f"Added day_of_week to {name}")

    # Remove temporary columns that might be in one but not the other
    local = local_df.copy().drop(columns=['hour', 'month'], errors='ignore')
    hops = hops_df.copy().drop(columns=['hour', 'month'], errors='ignore')

    local = local.sort_values('datetime_utc').reset_index(drop=True)
    hops  = hops.sort_values('datetime_utc').reset_index(drop=True)

    print(f"Date range local: {local['datetime_utc'].min()} → {local['datetime_utc'].max()}")
    print(f"Date range hops : {hops['datetime_utc'].min()} → {hops['datetime_utc'].max()}")

    if len(local) != len(hops):
        print(f"Row count mismatch: local={len(local)} hops={len(hops)}")
        min_len = min(len(local), len(hops))
        local = local.iloc[:min_len]
        hops  = hops.iloc[:min_len]

    # columns
    expected_extra = {'id'}
    core_hops = set(hops.columns) - expected_extra
    if set(local.columns) == core_hops:
        print("Core columns match")
    else:
        print("Columns differ:")
        print("   missing in hops :", set(local.columns) - core_hops)
        print("   extra   in hops :", set(hops.columns) - expected_extra - set(local.columns))

    # dtypes
    print("\nData types:")
    for c in sorted(local.columns):
        if c in hops.columns:
            l = str(local[c].dtype)
            h = str(hops[c].dtype)
            print(f"   {'OK' if l==h else 'DIFF'} {c:25} local={l:20} hops={h:20}")

    # stats on target
    if 'us_aqi' in local.columns and 'us_aqi' in hops.columns:
        print("\nus_aqi stats:")
        for stat in ['mean', 'std', 'min', 'max']:
            lv = getattr(local['us_aqi'], stat)()
            hv = getattr(hops['us_aqi'], stat)()
            diff = abs(lv - hv)
            print(f"   {stat:4}: local={lv:8.2f} hops={hv:8.2f} diff={diff:8.2f}")

    print("\n" + "="*70)
    return set(local.columns) == core_hops


# ============================================================================
# FINAL PIPELINE EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 0: LIST EXISTING FEATURE GROUPS")
    print("="*70)
    list_feature_groups("aqi_features")

    # Assuming version 4 is the desired version for the current feature set
    FG_VERSION = 4

    print("\n" + "="*70)
    print(f"STEP 1: DELETE VERSION {FG_VERSION} (safe)")
    print("="*70)
    delete_feature_group("aqi_features", version=FG_VERSION)

    print("\n" + "="*70)
    print(f"STEP 2: UPLOAD TO HOPSWORKS (v{FG_VERSION})")
    print("="*70)

    # Load the latest local features created by the EDA step (including aqi_category)
    df_local = pd.read_csv('data/2years_features.csv')
    if 'datetime_utc' in df_local.columns:
        df_local['datetime_utc'] = pd.to_datetime(df_local['datetime_utc'], errors='coerce', utc=True)
    print(f"Local CSV shape: {df_local.shape}")

    try:
        fg = upload_to_hopsworks(df_local, version=FG_VERSION)
        print("\nUPLOAD SUCCESSFUL")

        print("\n" + "="*70)
        print("STEP 3: FETCH & COMPARE")
        print("="*70)
        hops_df = fetch_from_hopsworks("aqi_features", version=FG_VERSION)
        if hops_df is not None:
            compare_datasets(df_local, hops_df)

        print("\n" + "="*70)
        print("STEP 4: TRAINING DATA")
        print("="*70)
        if 'datetime_utc' in df_local.columns:
            training_df = fetch_from_hopsworks(
                "aqi_features", version=FG_VERSION, for_training=True,
                start_date=df_local['datetime_utc'].min().isoformat(),
                end_date=df_local['datetime_utc'].max().isoformat()
            )
            if training_df is not None:
                print(f"TRAINING DATA READY: {training_df.shape}")
    except Exception as e:
        print(f"\nFATAL ERROR IN HOPSWORKS PIPELINE: {e}")
        print("Please check your HOPSWORKS_API_KEY and connection.")
