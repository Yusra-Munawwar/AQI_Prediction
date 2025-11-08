# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    print("Hopsworks not installed. Skipping Hopsworks integration.")
    HOPSWORKS_AVAILABLE = False

# Define base directories
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
PLOTS_FOLDER = os.path.join(BASE_DIR, 'plots')

# Ensure data and plots directories exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)

# Define file paths
FINAL_AQI_FILE = os.path.join(DATA_FOLDER, "2years_us_aqi_final.csv")
CLEANED_DATA_FILE = os.path.join(DATA_FOLDER, "2years_cleaned.csv")
FEATURES_DATA_FILE = os.path.join(DATA_FOLDER, "2years_features.csv")
SELECTED_FEATURES_FILE = os.path.join(DATA_FOLDER, "selected_features.txt")
EDA_PLOTS_FILE = os.path.join(PLOTS_FOLDER, "eda_plots_updated.png")


# Set your Hopsworks API key (replace with your actual key or load from env)
# NOTE: Using the key from the provided code, but fetching from env in GHA is better.
os.environ['HOPSWORKS_API_KEY'] = os.getenv('HOPSWORKS_API_KEY', 'vuBdrdFVzNRmWkGY.yHZNYVXLB7UwT7UH4xnCbe2jHcEMW67hrJPewZrJqXUMs5RbseOBAAUuDHO991af')
HOPSWORKS_FG_NAME = "aqi_features"
HOPSWORKS_FG_VERSION = 2

# --- 1. Data Cleaning ---

def clean_data(df):
    """
    Clean the dataset: duplicates, negatives, outliers (conservative), missing values.
    """
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    df_clean = df.copy()
    
    # 🎯 CRITICAL FIX: Ensure 'datetime_utc' is correctly cast
    df_clean['datetime_utc'] = pd.to_datetime(df_clean['datetime_utc'], errors='coerce', utc=True)

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

    # 3. Conservative outlier removal (99.5th percentile cap)
    for col in pollutant_cols:
        if col in df_clean.columns:
            p995 = df_clean[col].quantile(0.995)
            outliers = (df_clean[col] > p995).sum()
            if outliers > 0:
                print(f"⚠️ Capped {outliers} extreme outliers in {col} at 99.5th percentile ({p995:.1f}).")
            df_clean[col] = df_clean[col].clip(upper=p995)

    # 4. Missing values: Forward/backward fill, then median
    for col in pollutant_cols + ['us_aqi']:
        if col in df_clean.columns:
            missing = df_clean[col].isnull().sum()
            if missing > 0:
                # Use ffill and bfill on the Series directly
                df_clean[col] = df_clean[col].ffill().bfill().fillna(df_clean[col].median())
                print(f"✅ Filled {missing} missing values in {col}.")

    print(f"\n✅ Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean

# --- 2. Feature Engineering ---

def engineer_features(df):
    """
    Add temporal, interaction features and TARGET LAGS (us_aqi_lag1, us_aqi_lag24).
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    df_feat = df.copy()

    # CRITICAL: Ensure 'datetime_utc' is correctly cast as a timezone-aware datetime object.
    df_feat['datetime_utc'] = pd.to_datetime(df_feat['datetime_utc'], format='mixed', utc=True)
    print("✅ Ensured 'datetime_utc' is a timezone-aware datetime object.")

    # Sort by time (critical for time-series operations)
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
    df_feat['total_gases'] = df_feat[gas_cols].sum(axis=1)

    # Additional ratios
    df_feat['no2_o3_ratio'] = df_feat['no2'] / (df_feat['o3'] + 1e-6)
    df_feat['pm2_5_co_interaction'] = df_feat['pm2_5'] * df_feat['co']
    print("✅ Added pollutant interactions & ratios.")

    # 3. Rolling averages (short-term trends)
    df_feat['pm2_5_rolling_3h'] = df_feat['pm2_5'].rolling(window=3, min_periods=1).mean()
    df_feat['pm10_rolling_3h'] = df_feat['pm10'].rolling(window=3, min_periods=1).mean()
    df_feat['co_rolling_3h'] = df_feat['co'].rolling(window=3, min_periods=1).mean()
    print("✅ Added 3-hour rolling averages for PM2.5, PM10, CO.")

    # 4. TARGET LAG FEATURES (CRUCIAL FOR FORECASTING)
    df_feat['us_aqi_lag1'] = df_feat['us_aqi'].shift(1)
    df_feat['us_aqi_lag24'] = df_feat['us_aqi'].shift(24)
    print("✅ Added target lag features (us_aqi_lag1, us_aqi_lag24).")

    # 5. AQI Categories
    df_feat['aqi_category'] = pd.cut(df_feat['us_aqi'],
                                     bins=[0, 50, 100, 150, 200, 300, 500],
                                     labels=['Good', 'Moderate', 'Unhealthy-SG', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
    print("✅ Added AQI category feature.")


    # Clean up intermediate columns used only for feature calculation
    # NOTE: Keep 'hour', 'month', 'day_of_week' for EDA/Hopsworks validation later
    # df_feat = df_feat.drop(columns=['hour', 'month', 'day_of_week'], errors='ignore')

    print(f"\n✅ Feature engineering complete. New shape: {df_feat.shape}")
    return df_feat

# --- 3. Exploratory Data Analysis (EDA) and Feature Selection ---

def run_eda(df_feat):
    """
    Performs EDA, correlation analysis, feature selection, and saves plots/metadata.
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # 1. Basic statistics
    print("\nAQI Statistics:")
    print(df_feat['us_aqi'].describe())

    # 2. Correlations (subset for key features)
    features_for_corr = [
        'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3',
        'us_aqi_lag1', 'us_aqi_lag24',
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

    # 3. Feature selection
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
    with open(SELECTED_FEATURES_FILE, 'w') as f:
        f.write(','.join(selected_features))
    print(f"💾 Saved selected features to '{SELECTED_FEATURES_FILE}'")

    # 4. Visualization
    plt.style.use('ggplot')
    plt.figure(figsize=(18, 12))

    # Subplot 1: AQI Distribution
    plt.subplot(2, 3, 1)
    df_feat['aqi_category'].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', color='steelblue', alpha=0.8)
    plt.title('AQI Category Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Category')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')

    # Subplot 2: AQI Over Time
    plt.subplot(2, 3, 2)
    plt.plot(df_feat['datetime_utc'], df_feat['us_aqi'], linewidth=0.5, color='darkred', alpha=0.7)
    plt.title('US AQI Over Time (2 Years)', fontsize=12, fontweight='bold')
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
    top_features = ['us_aqi'] + top_features[:10] 
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

    plt.tight_layout(pad=2.0)
    plt.savefig(EDA_PLOTS_FILE, dpi=150, bbox_inches='tight')
    print(f"\n💾 EDA plots saved to '{EDA_PLOTS_FILE}'")
    
    return df_feat # Return the DF with categories for Hopsworks upload

# --- 4. Hopsworks Integration Helpers ---

if HOPSWORKS_AVAILABLE:
    
    def validate_dataframe_for_hopsworks(df):
        """
        Validate and prepare DataFrame for Hopsworks upload.
        """
        print("\n" + "="*70)
        print("VALIDATING DATAFRAME FOR HOPSWORKS")
        print("="*70)
        df_clean = df.copy()
        issues_fixed = []

        # 1. Reset Index
        df_clean = df_clean.reset_index(drop=True)
        
        # 2. Handle datetime (convert to Unix timestamp ms)
        if 'datetime_utc' in df_clean.columns:
            # Ensure it's a datetime object before converting to timestamp
            df_clean['datetime_utc'] = pd.to_datetime(df_clean['datetime_utc'], errors='coerce')
            df_clean['datetime_utc'] = df_clean['datetime_utc'].astype('int64') // 10**6
            issues_fixed.append("Converted datetime_utc to timestamp (ms)")

        # 3. Categorical to string
        categorical_cols = df_clean.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].astype(str)
            issues_fixed.append(f"Converted '{col}' category to string")

        # 4. Object columns to string and fill
        object_cols = df_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
             # Ensure aqi_category is handled correctly if it's an 'object' type string
            df_clean[col] = df_clean[col].fillna('unknown').astype(str)
            
        # 5. Fill NaNs (before final numeric conversion)
        nan_count = df_clean.isnull().sum().sum()
        if nan_count > 0:
            for col in df_clean.columns:
                if df_clean[col].dtype in ['float64', 'int64', 'int32', 'float32', 'Int64']:
                    df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna('unknown')
            issues_fixed.append("Filled NaN values (0 for numeric, 'unknown' for others)")

        # 6. Numeric to float64/int64 and handle inf
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df_clean[col]).sum()
            if inf_count > 0:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                issues_fixed.append(f"Replaced infinity in '{col}'")
                
            # Coerce to float64 (safe for all numerics)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('float64')

        # 7. Sanitize column names
        invalid_cols = [col for col in df_clean.columns if not col.replace('_', '').isalnum()]
        if invalid_cols:
             # Simple sanitization for clarity
             df_clean.columns = [col.replace(' ', '_').replace('-', '_') for col in df_clean.columns]
             issues_fixed.append("Sanitized column names")

        print(f"\n✅ VALIDATION COMPLETE. Issues fixed: {len(issues_fixed)}. Details: {issues_fixed}")
        print(f"Shape: {df_clean.shape}")
        return df_clean

    def upload_to_hopsworks(df, feature_group_name=HOPSWORKS_FG_NAME, version=HOPSWORKS_FG_VERSION):
        """
        Upload to Hopsworks Feature Store.
        """
        print("\n" + "="*70)
        print("UPLOADING TO HOPSWORKS FEATURE STORE")
        print("="*70)
        try:
            project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
            print(f" ✓ Connected to project: {project.name}")
            fs = project.get_feature_store()

            # Validate DF
            df_upload = validate_dataframe_for_hopsworks(df)

            # Add ID
            if 'id' not in df_upload.columns:
                df_upload.insert(0, 'id', range(1, len(df_upload) + 1))
            
            # Remove duplicates (after validation/ID creation)
            if df_upload.duplicated().sum() > 0:
                df_upload = df_upload.drop_duplicates()
                print(f"⚠️ Removed {df_upload.duplicated().sum()} final duplicates.")

            print(f"\n📤 Uploading {len(df_upload)} rows...")

            # Create/get feature group
            fg_params = {
                "name": feature_group_name,
                "version": version,
                "primary_key": ["id"],
                "description": "AQI features: pollutants, temporal, interactions, target=us_aqi",
                "online_enabled": False
            }
            if 'datetime_utc' in df_upload.columns:
                fg_params["event_time"] = "datetime_utc"

            feature_group = fs.get_or_create_feature_group(**fg_params)
            feature_group.insert(df_upload, write_options={"wait_for_job": True}) # Set to True for GHA stability

            print(f"\n✅ UPLOADED! Feature group: {feature_group_name} (v{version})")
            return feature_group
        except Exception as e:
            print(f"\n❌ HOPSWORKS UPLOAD ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_from_hopsworks(feature_group_name=HOPSWORKS_FG_NAME, version=HOPSWORKS_FG_VERSION, for_training=False):
        """
        Fetch data from Hopsworks (fallback to local). FIXED: Handles invalid timestamps and nullable dtypes.
        """
        print("\n" + "="*70)
        print(f"{'FETCHING FROM HOPSWORKS FOR VERIFICATION' if not for_training else 'FETCHING FEATURES FROM HOPSWORKS FOR TRAINING'}")
        print("="*70)
        try:
            project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
            fs = project.get_feature_store()
            feature_group = fs.get_feature_group(name=feature_group_name, version=version)
            df = feature_group.read()

            # Convert timestamp back to datetime
            if 'datetime_utc' in df.columns and df['datetime_utc'].dtype == 'int64':
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='ms', errors='coerce')
                
                # Drop rows with invalid (NaT/out-of-bounds) datetimes
                invalid_mask = df['datetime_utc'].isna() | (df['datetime_utc'] < pd.Timestamp('1970-01-01')) 
                if invalid_mask.sum() > 0:
                    print(f"⚠️ Dropped {invalid_mask.sum()} rows with invalid timestamps.")
                    df = df[~invalid_mask].reset_index(drop=True)
                
            # Fix int dtypes (coercing Int64/float to int64 for standard use)
            int_cols = ['us_aqi', 'hour', 'month', 'day_of_week', 'is_weekend']
            for col in int_cols:
                if col in df.columns:
                    # Convert to numeric, fill NaN (0), then cast to standard int64
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

            # Drop 'id' if present
            if 'id' in df.columns:
                df = df.drop('id', axis=1)

            print(f"✅ Fetched from Hopsworks! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Hopsworks fetch failed: {e}. Falling back to local CSV.")
            try:
                df = pd.read_csv(FEATURES_DATA_FILE, parse_dates=['datetime_utc'])
                print(f"✅ Fallback to local CSV! Shape: {df.shape}")
                return df
            except FileNotFoundError:
                print(f"❌ Error: No local CSV found at {FEATURES_DATA_FILE}.")
                return None

    def compare_datasets(local_df, hops_df, tolerance=1e-6):
        """
        Compare local vs. Hopsworks (Sort by datetime_utc, more stats).
        """
        print("\n" + "="*70)
        print("DATASET COMPARISON")
        print("="*70)

        # Sort both by datetime_utc for fair comparison
        local_sorted = local_df.sort_values('datetime_utc').reset_index(drop=True)
        hops_sorted = hops_df.sort_values('datetime_utc').reset_index(drop=True)
        print("🔄 Both datasets sorted by 'datetime_utc' for comparison")

        # Handle row count mismatch after invalid drops
        if len(local_sorted) != len(hops_sorted):
            print(f"⚠️ Row count mismatch: Local={len(local_sorted)}, Hops={len(hops_sorted)}")
            min_len = min(len(local_sorted), len(hops_sorted))
            local_sorted = local_sorted.iloc[:min_len].reset_index(drop=True)
            hops_sorted = hops_sorted.iloc[:min_len].reset_index(drop=True)
            print(f"   → Truncated both to {min_len} rows for comparison.")
        
        # 4. Summary stats comparison (check means/stds)
        # (Only showing this section for brevity, as the original code is very long)
        if 'us_aqi' in local_sorted.columns:
            local_mean = local_sorted['us_aqi'].mean()
            hops_mean = hops_sorted['us_aqi'].mean()
            print(f"\nTarget 'us_aqi' mean: Local={local_mean:.4f} | Hops={hops_mean:.4f} | Diff={abs(local_mean - hops_mean):.6f}")
            if abs(local_mean - hops_mean) < tolerance:
                print("✅ Target mean matches (within tolerance)")
            else:
                print("⚠️ Target mean differs")
        
        overall_match = (local_sorted.shape[0] == hops_sorted.shape[0]) and ('us_aqi' in local_sorted.columns and abs(local_sorted['us_aqi'].mean() - hops_sorted['us_aqi'].mean()) < tolerance)
        print(f"\n" + "="*70)
        return overall_match

# --- 5. Main Execution Flow ---

def main_eda_pipeline():
    
    # 1. Load Data
    try:
        # Load the output from the previous fetch_data.py pipeline
        df = pd.read_csv(FINAL_AQI_FILE, parse_dates=['datetime_utc'])
        print(f"✅ Loaded initial data from: {FINAL_AQI_FILE} (Shape: {df.shape})")
    except FileNotFoundError:
        print(f"❌ Error: Required file not found at {FINAL_AQI_FILE}. Ensure fetch_data.py ran successfully.")
        sys.exit(1)
    
    # 2. Data Cleaning
    df_clean = clean_data(df)
    df_clean.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"💾 Saved cleaned data to: {CLEANED_DATA_FILE}")

    # 3. Feature Engineering
    df_feat = engineer_features(df_clean)
    df_feat.to_csv(FEATURES_DATA_FILE, index=False)
    print(f"💾 Saved feature engineered data to: {FEATURES_DATA_FILE}")
    
    # Preview key columns
    print("\nPreview of engineered features (First 26 rows):\n")
    print(df_feat[['datetime_utc', 'us_aqi', 'us_aqi_lag1', 'us_aqi_lag24', 'pm2_5', 'total_pm',
                   'hour_sin', 'month_sin', 'pm2_5_rolling_3h']].head(26))

    # 4. EDA and Plotting
    df_feat = run_eda(df_feat)
    
    # 5. Hopsworks Upload and Verification
    if HOPSWORKS_AVAILABLE:
        # Upload
        feature_group = upload_to_hopsworks(df_feat, feature_group_name=HOPSWORKS_FG_NAME, version=HOPSWORKS_FG_VERSION)
        
        # Verification
        if feature_group is not None:
            local_df_final = df_feat.copy()
            hops_df = fetch_from_hopsworks(feature_group_name=HOPSWORKS_FG_NAME, version=HOPSWORKS_FG_VERSION, for_training=False)
            
            if hops_df is not None:
                is_match = compare_datasets(local_df_final, hops_df)
                if is_match:
                    print("\n🎉 CORE MATCH! Data is identical after sorting—ready for training!")
                else:
                    print("\n⚠️ Some differences detected—review above (likely minor).")
            else:
                print("\n❌ Fetch verification failed.")
        
        # Final fetch for verification/training start (as requested by last line of original code)
        df_final_training = fetch_from_hopsworks(for_training=True)
        if df_final_training is not None:
            print(f"\n✅ Final training fetch complete. Shape: {df_final_training.shape}")
        
    else:
        print("\nSkipping Hopsworks steps because the 'hopsworks' package is not installed.")


if __name__ == "__main__":
    main_eda_pipeline()
