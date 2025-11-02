import requests
import pandas as pd
import time
import os
import numpy as np
from datetime import datetime, timedelta

# === Configuration ===
API_KEY = os.getenv('OWM_API_KEY', '29e4f8ef9151633260fb36745ed19012')
LAT = 24.8607  # Karachi latitude
LON = 67.0011  # Karachi longitude

def fetch_historical_data():
    """Fetch 2 years of historical air quality data"""
    print("\n" + "="*70)
    print("FETCHING HISTORICAL DATA (2 YEARS)")
    print("="*70)
    
    # Calculate timestamps for past 2 years
    end_time = int(time.time())
    start_time = int((datetime.now() - timedelta(days=730)).timestamp())
    
    # API URL
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_time}&end={end_time}&appid={API_KEY}"
    print(f"Fetching from: {url}")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
    data = response.json()
    if "list" not in data:
        print(f"No data returned: {data}")
        return None
    
    records = []
    for item in data["list"]:
        dt = datetime.utcfromtimestamp(item["dt"])
        components = item["components"]
        aqi = item["main"]["aqi"]
        records.append({
            "datetime_utc": dt,
            "aqi": aqi,
            **components
        })
    
    df = pd.DataFrame(records)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df.set_index("datetime_utc", inplace=True)
    
    # Resample to 2-hour intervals
    df_2h = df.resample("2H").mean().reset_index()
    print(f"\n✅ Historical data shape: {df_2h.shape}")
    print(df_2h.head())
    
    return df_2h

def fetch_realtime_aqi(lat, lon, api_key):
    """Fetch real-time air quality data"""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['list'][0]['components']
            dt = datetime.utcnow()
            new_row = {'datetime_utc': dt, **data}
            print(f"✅ Fetched real-time data: {dt}")
            return new_row
        else:
            print(f"❌ API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Fetch error: {e}")
        return None

def append_realtime_data(csv_path='data/2years.csv'):
    """Append real-time data to existing CSV"""
    print("\n" + "="*70)
    print("APPENDING REAL-TIME DATA")
    print("="*70)
    
    new_row = fetch_realtime_aqi(LAT, LON, API_KEY)
    if new_row:
        # Load existing CSV
        df_existing = pd.read_csv(csv_path, parse_dates=['datetime_utc'])
        
        # Create new row DataFrame
        new_row_df = pd.DataFrame([new_row])
        
        # Align columns
        for col in df_existing.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = np.nan
        new_row_df = new_row_df[df_existing.columns]
        
        # Append
        df_updated = pd.concat([df_existing, new_row_df], ignore_index=True)
        df_updated.to_csv(csv_path, index=False)
        print(f"✅ Appended real-time row. New shape: {df_updated.shape}")
        return df_updated
    else:
        print("⚠️ No new data appended.")
        return pd.read_csv(csv_path, parse_dates=['datetime_utc'])

def calculate_us_aqi(df):
    """Calculate US AQI from pollutant concentrations"""
    print("\n" + "="*70)
    print("CALCULATING US AQI")
    print("="*70)
    
    # Updated breakpoints (PM2.5: 2024 EPA; others standard)
    breakpoints = {
        'pm2_5': [
            (0.0, 0, 9.0, 50), (9.1, 51, 35.4, 100), (35.5, 101, 55.4, 150),
            (55.5, 151, 125.4, 200), (125.5, 201, 225.4, 300), (225.5, 301, 500.4, 500)
        ],
        'pm10': [
            (0, 0, 54, 50), (55, 51, 154, 100), (155, 101, 254, 150),
            (255, 151, 354, 200), (355, 201, 424, 300), (425, 301, 504, 400),
            (505, 401, 604, 500)
        ],
        'o3': [
            (0.000, 0, 0.054, 50), (0.055, 51, 0.070, 100), (0.071, 101, 0.085, 150),
            (0.086, 151, 0.105, 200), (0.106, 201, 0.200, 300), (0.201, 301, 0.404, 400),
            (0.405, 401, 0.604, 500)
        ],
        'no2': [
            (0.000, 0, 0.053, 50), (0.054, 51, 0.100, 100), (0.101, 101, 0.360, 150),
            (0.361, 151, 0.649, 200), (0.650, 201, 0.854, 300), (0.855, 301, 1.049, 400),
            (1.050, 401, 2.104, 500)
        ],
        'so2': [
            (0.000, 0, 0.004, 50), (0.005, 51, 0.009, 100), (0.010, 101, 0.014, 150),
            (0.015, 151, 0.035, 200), (0.036, 201, 0.075, 300), (0.076, 301, 0.185, 400),
            (0.186, 401, 0.604, 500)
        ],
        'co': [
            (0.0, 0, 4.4, 50), (4.5, 51, 9.4, 100), (9.5, 101, 12.4, 150),
            (12.5, 151, 15.4, 200), (15.5, 201, 30.4, 300), (30.5, 301, 50.4, 500)
        ]
    }
    
    def to_epa_units(c, pollutant):
        if pd.isna(c) or c <= 0:
            return 0.0
        if pollutant in ['pm2_5', 'pm10']:
            return c
        elif pollutant == 'co':
            return c / 1145.0
        elif pollutant == 'o3':
            return c / 1960.6
        elif pollutant == 'no2':
            return c / 1881.1
        elif pollutant == 'so2':
            return c / 2620.0
        return 0.0
    
    def calc_sub_aqi(c, pollutant):
        c_epa = to_epa_units(c, pollutant)
        bps = breakpoints.get(pollutant, [])
        if c_epa <= 0:
            return 0
        prev_high = -np.inf
        for i in range(len(bps)):
            c_low, i_low, c_high, i_high = bps[i]
            if c_epa > prev_high and c_epa <= c_high:
                return i_low + ((i_high - i_low) * (c_epa - c_low)) / (c_high - c_low)
            prev_high = c_high
        return 500 if c_epa > bps[-1][2] else 0
    
    def calc_us_aqi(row):
        pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
        sub_aqis = [calc_sub_aqi(row.get(p, 0), p) for p in pollutants]
        return max(sub_aqis)
    
    # Calculate US AQI
    df['us_aqi'] = df.apply(calc_us_aqi, axis=1)
    df['us_aqi'] = np.clip(df['us_aqi'].round().astype(int), 0, 500)
    
    print(f"✅ US AQI calculated. Dataset shape: {df.shape}")
    print(df[['datetime_utc', 'aqi', 'us_aqi', 'pm2_5', 'pm10', 'co']].head(10))
    
    return df

def main():
    """Main execution function"""
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if historical data exists
    csv_path = 'data/2years.csv'
    if not os.path.exists(csv_path):
        print("Historical data not found. Fetching...")
        df = fetch_historical_data()
        if df is not None:
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved historical data to {csv_path}")
    
    # Append real-time data
    df = append_realtime_data(csv_path)
    
    # Calculate US AQI
    df = calculate_us_aqi(df)
    
    # Save with US AQI
    output_path = 'data/2years_us_aqi_final.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Final data saved to {output_path}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    main()