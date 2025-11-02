import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hopsworks

def clean_data(df):
    """Clean the dataset: duplicates, negatives, outliers, missing values"""
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)
    
    df_clean = df.copy()
    
    # 1. Remove duplicates by timestamp
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['datetime_utc'], keep='first')
    print(f"✅ Removed {initial_len - len(df_clean)} duplicates.")
    
    # 2. Clip negatives to 0
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
                df_clean[col] = df_clean[col].ffill().bfill().fillna(df_clean[col].median())
                print(f"✅ Filled {missing} missing values in {col}.")
    
    print(f"\n✅ Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean

def engineer_features(df):
    """Add temporal and interaction features"""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    df_feat = df.copy()
    
    # Sort by time
    df_feat = df_feat.sort_values('datetime_utc').reset_index(drop=True)
    
    # 1. Temporal features (cyclical encoding)
    df_feat['hour'] = df_feat['datetime_utc'].dt.hour
    df_feat['month'] = df_feat['datetime_utc'].dt.month
    df_feat['day_of_week'] = df_feat['datetime_utc'].dt.dayofweek
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
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
    df_feat['no2_o3_ratio'] = df_feat['no2'] / (df_feat['o3'] + 1e-6)
    df_feat['pm2_5_co_interaction'] = df_feat['pm2_5'] * df_feat['co']
    print("✅ Added pollutant interactions & ratios.")
    
    # 3. Rolling averages (short-term trends)
    df_feat['pm2_5_rolling_3h'] = df_feat['pm2_5'].rolling(window=3, min_periods=1).mean()
    df_feat['pm10_rolling_3h'] = df_feat['pm10'].rolling(window=3, min_periods=1).mean()
    df_feat['co_rolling_3h'] = df_feat['co'].rolling(window=3, min_periods=1).mean()
    print("✅ Added 3-hour rolling averages for PM2.5, PM10, CO.")
    
    print(f"\n✅ Feature engineering complete. New shape: {df_feat.shape}")
    return df_feat

def perform_eda(df_feat):
    """Perform exploratory data analysis"""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # 1. Basic statistics
    print("\nAQI Statistics:")
    print(df_feat['us_aqi'].describe())
    
    # 2. Correlations
    features_for_corr = [
        'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'pm_ratio', 'total_pm', 'total_gases', 'no2_o3_ratio',
        'pm2_5_rolling_3h', 'pm10_rolling_3h', 'co_rolling_3h',
        'pm2_5_co_interaction', 'is_weekend',
        'us_aqi'
    ]
    
    features_for_corr = [f for f in features_for_corr if f in df_feat.columns]
    corr = df_feat[features_for_corr].corr()['us_aqi'].sort_values(ascending=False)
    print("\n📊 Top Correlations with US AQI:")
    print(corr.head(15))
    
    # 3. Feature selection (|correlation| > 0.3)
    selected_features = [f for f in features_for_corr if f != 'us_aqi' and abs(corr[f]) > 0.3]
    print(f"\n✅ Selected Features (|corr| > 0.3): {len(selected_features)} features")
    print(selected_features)
    
    # Save selected features
    with open('data/selected_features.txt', 'w') as f:
        f.write(','.join(selected_features))
    print("💾 Saved selected features to 'data/selected_features.txt'")
    
    # 4. Visualization
    os.makedirs('data', exist_ok=True)
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
    
    # Subplot 4: Correlation Heatmap
    plt.subplot(2, 3, 4)
    top_features = corr.head(11).index.tolist()
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
    
    # Subplot 6: AQI by Hour
    plt.subplot(2, 3, 6)
    hourly_aqi = df_feat.groupby('hour')['us_aqi'].mean()
    plt.plot(hourly_aqi.index, hourly_aqi.values, marker='o', color='orange', linewidth=2)
    plt.title('Average AQI by Hour', fontsize=12, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average AQI')
    plt.xticks(range(0, 24, 3))
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/eda_plots_updated.png', dpi=150, bbox_inches='tight')
    print("\n💾 EDA plots saved to 'data/eda_plots_updated.png'")
    plt.close()
    
    return selected_features

def validate_dataframe_for_hopsworks(df):
    """Validate and prepare DataFrame for Hopsworks"""
    print("\n" + "="*70)
    print("VALIDATING DATAFRAME FOR HOPSWORKS")
    print("="*70)
    
    df_clean = df.copy()
    issues_fixed = []
    
    # 1. Reset MultiIndex if present
    if isinstance(df_clean.index, pd.MultiIndex):
        df_clean = df_clean.reset_index(drop=True)
        issues_fixed.append("Reset MultiIndex")
    
    # 2. Handle MultiIndex columns
    if isinstance(df_clean.columns, pd.MultiIndex):
        df_clean.columns = ['_'.join(map(str, col)).strip() for col in df_clean.columns.values]
        issues_fixed.append("Flattened MultiIndex columns")
    
    # 3. Remove duplicate columns
    if df_clean.columns.duplicated().any():
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        issues_fixed.append("Removed duplicate columns")
    
    # 4. Fix mixed types
    for col in df_clean.columns:
        types_in_col = df_clean[col].apply(type).unique()
        if len(types_in_col) > 2:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    issues_fixed.append(f"Converted '{col}' to numeric")
                except:
                    df_clean[col] = df_clean[col].astype(str)
                    issues_fixed.append(f"Converted '{col}' to string")
    
    # 5. Handle datetime (convert to Unix timestamp ms)
    if 'datetime_utc' in df_clean.columns:
        df_clean['datetime_utc'] = pd.to_datetime(df_clean['datetime_utc'], errors='coerce')
        df_clean['datetime_utc'] = df_clean['datetime_utc'].astype('int64') // 10**6
        issues_fixed.append("Converted datetime_utc to timestamp")
    
    # 6. Categorical to string
    categorical_cols = df_clean.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str)
        issues_fixed.append(f"Converted '{col}' category to string")
    
    # 7. Object columns to string
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_clean[col] = df_clean[col].fillna('unknown').astype(str)
    
    # 8. Numeric to float64/int64
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].dtype not in ['int64', 'float64']:
            if df_clean[col].apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                df_clean[col] = df_clean[col].astype('int64')
            else:
                df_clean[col] = df_clean[col].astype('float64')
    
    # 9. Fill NaNs
    nan_count = df_clean.isnull().sum().sum()
    if nan_count > 0:
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna('unknown')
        issues_fixed.append("Filled NaN values")
    
    # 10. Handle inf
    for col in numeric_cols:
        inf_count = np.isinf(df_clean[col]).sum()
        if inf_count > 0:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            issues_fixed.append(f"Replaced infinity in '{col}'")
    
    # 11. Sanitize column names
    invalid_cols = [col for col in df_clean.columns if not col.replace('_', '').isalnum()]
    if invalid_cols:
        for col in invalid_cols:
            new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in col)
            df_clean = df_clean.rename(columns={col: new_col})
        issues_fixed.append("Sanitized column names")
    
    print(f"\n✅ VALIDATION COMPLETE. Issues fixed: {len(issues_fixed)}")
    print(f"Shape: {df_clean.shape}")
    return df_clean

def upload_to_hopsworks(df, feature_group_name="aqi_features", version=1):
    """Upload to Hopsworks Feature Store"""
    print("\n" + "="*70)
    print("UPLOADING TO HOPSWORKS FEATURE STORE")
    print("="*70)
    
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        print(f"✓ Connected to project: {project.name}")
        
        fs = project.get_feature_store()
        
        # Validate DF
        df_upload = validate_dataframe_for_hopsworks(df)
        
        # Add ID if missing
        if 'id' not in df_upload.columns:
            df_upload.insert(0, 'id', range(1, len(df_upload) + 1))
        
        # Fill remaining NaNs and ensure numerics are float64
        df_upload = df_upload.fillna(0)
        numeric_cols = df_upload.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col != 'id' and col != 'datetime_utc':
                df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').astype('float64')
        
        # Remove duplicates
        if df_upload.duplicated().sum() > 0:
            df_upload = df_upload.drop_duplicates()
        
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
        feature_group.insert(df_upload, write_options={"wait_for_job": False})
        
        print(f"\n✅ UPLOADED! Feature group: {feature_group_name} (v{version})")
        return feature_group
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_datasets(local_df, hops_df, tolerance=1e-6):
    """Compare local vs. Hopsworks datasets"""
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
        print(f"→ Truncated both to {min_len} rows for comparison.")
    
    # 1. Shape comparison
    print(f"Local shape (aligned): {local_sorted.shape}")
    print(f"Hopsworks shape (aligned): {hops_sorted.shape}")
    
    # 2. Columns comparison
    local_cols = set(local_sorted.columns)
    hops_cols = set(hops_sorted.columns)
    expected_extras = {'id'}
    core_hops_cols = hops_cols - expected_extras
    
    print(f"\nLocal columns ({len(local_cols)}): {sorted(local_cols)}")
    print(f"Hopsworks columns ({len(hops_cols)}): {sorted(hops_cols)}")
    
    if local_cols == core_hops_cols:
        print("✅ Core columns match exactly")
    else:
        print("⚠️ Column mismatch!")
        print("Missing in Hopsworks:", local_cols - core_hops_cols)
        print("Extra in Hopsworks:", hops_cols - expected_extras - local_cols)
    
    # 3. Summary stats comparison
    if 'us_aqi' in local_sorted.columns:
        local_mean = local_sorted['us_aqi'].mean()
        hops_mean = hops_sorted['us_aqi'].mean()
        local_std = local_sorted['us_aqi'].std()
        hops_std = hops_sorted['us_aqi'].std()
        
        print(f"\nTarget 'us_aqi' stats:")
        print(f"Mean: Local={local_mean:.4f} | Hops={hops_mean:.4f} | Diff={abs(local_mean - hops_mean):.6f}")
        print(f"Std: Local={local_std:.4f} | Hops={hops_std:.4f} | Diff={abs(local_std - hops_std):.6f}")
        
        if abs(local_mean - hops_mean) < tolerance and abs(local_std - hops_std) < tolerance:
            print("✅ Target stats match")
            return True
    
    return False

def fetch_from_hopsworks_verify(feature_group_name="aqi_features", version=1):
    """Fetch from Hopsworks for verification"""
    print("\n" + "="*70)
    print("FETCHING FROM HOPSWORKS FOR VERIFICATION")
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
        print(f"❌ Fetch failed: {e}")
        return None

def main():
    """Main execution function"""
    # Load data
    df = pd.read_csv('data/2years_us_aqi_final.csv', parse_dates=['datetime_utc'])
    
    # Clean data
    df_clean = clean_data(df)
    df_clean.to_csv('data/2years_cleaned.csv', index=False)
    print("💾 Saved to 'data/2years_cleaned.csv'")
    
    # Feature engineering
    df_feat = engineer_features(df_clean)
    
    # Add AQI category
    df_feat['aqi_category'] = pd.cut(
        df_feat['us_aqi'],
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=['Good', 'Moderate', 'Unhealthy-SG', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    )
    
    df_feat.to_csv('data/2years_features.csv', index=False)
    print("💾 Saved to 'data/2years_features.csv'")
    
    # Perform EDA
    selected_features = perform_eda(df_feat)
    
    # Preview key columns
    print("\nPreview of engineered features:")
    print(df_feat[['datetime_utc', 'us_aqi', 'pm2_5', 'total_pm',
                    'hour_sin', 'month_sin', 'pm2_5_rolling_3h']].head(10))
    
    # Upload to Hopsworks
    feature_group = upload_to_hopsworks(df_feat, feature_group_name="aqi_features", version=1)
    print("\n💾 Features uploaded to Hopsworks Feature Store!")
    
    # Verify upload
    if feature_group is not None:
        hops_df = fetch_from_hopsworks_verify(feature_group_name="aqi_features", version=1)
        if hops_df is not None:
            is_match = compare_datasets(df_feat, hops_df)
            if is_match:
                print("\n🎉 VERIFICATION COMPLETE! Data uploaded successfully!")
            else:
                print("\n⚠️ Some differences detected—review above.")
    
    print("\n✅ EDA and upload complete!")

if __name__ == "__main__":
    main()
