import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import os
import sys

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
EXISTING_DATA_FILE = os.path.join(DATA_FOLDER, "2years.csv")
FINAL_DATA_FILE = os.path.join(DATA_FOLDER, "2years_us_aqi_final.csv")

API_KEY = os.getenv("OWM_API_KEY")
LAT = 24.8607  # Karachi latitude
LON = 67.0011  # Karachi longitude

BREAKPOINTS = {
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
    """Converts OpenWeatherMap concentration (µg/m³ for gases, µg/m³ for particulates)
       to US EPA standard units (ppm for CO, ppb for O3/NO2/SO2, µg/m³ for particulates).
       Conversion factors are based on 25°C and 1 atm (or approximated values used in similar contexts).
    """
    if pd.isna(c) or c <= 0:
        return 0.0

    if pollutant == 'co':
        return c / 1145.0
    elif pollutant == 'o3':
        return c / 1960.6
    elif pollutant == 'no2':
        return c / 1881.1
    elif pollutant == 'so2':
        return c / 2620.0
    else:
        return c

def calc_sub_aqi(c, pollutant):
    """Calculates the Individual Air Quality Index (IAQI) for a pollutant."""
    c_epa = to_epa_units(c, pollutant)
    bps = BREAKPOINTS.get(pollutant, [])
    
    if c_epa <= 0:
        return 0
    
    if bps and c_epa > bps[-1][2]:
        c_low, i_low, c_high, i_high = bps[-1]
        return 500
        
    prev_high = -np.inf
    for i in range(len(bps)):
        c_low, i_low, c_high, i_high = bps[i]
        if c_epa > prev_high and c_epa <= c_high:
            return i_low + ((i_high - i_low) * (c_epa - c_low)) / (c_high - c_low)
        prev_high = c_high
        
    return 0

def calc_us_aqi(row):
    """Calculates the overall US AQI based on the maximum IAQI."""
    pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    sub_aqis = [calc_sub_aqi(row.get(p, 0), p) for p in pollutants]
    return max(sub_aqis)

def fetch_historical_data(lat, lon, api_key):
    """
    Fetches historical air quality data, resamples to 1-hour, and saves to CSV.
    This function overwrites the 2years.csv file.
    """
    print("\n" + "="*50)
    print("STEP 1: FETCHING HISTORICAL DATA")
    print("="*50)
    
    end_time = int(time.time()) 
    start_time = int((datetime.now() - timedelta(days=730)).timestamp()) 

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_time}&end={end_time}&appid={api_key}"
    print("Fetching data from:", url)

    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error fetching historical data: {response.status_code}, {response.text}")
        return None
    data = response.json()
    if "list" not in data:
        print("⚠️ No data returned in the 'list' key.")
        return None

    records = []
    for item in data["list"]:
        dt = datetime.utcfromtimestamp(item["dt"])
        components = item["components"]
        aqi = item["main"]["aqi"] 
        records.append({"datetime_utc": dt,"aqi": aqi, **components})
    
    if not records:
        print("⚠️ Historical API returned 0 records.")
        return None

    df = pd.DataFrame(records)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize('UTC')
    df.set_index("datetime_utc", inplace=True)
    df_1h = df.resample("H").mean().interpolate(method='linear').reset_index()

    print(f"\n✅ Historical data fetched. Raw points: {len(records)}. Resampled points: {len(df_1h)}")
    print("\nSample of resampled (1-hour interval) data:\n")
    print(df_1h.head())
    
    df_1h.to_csv(EXISTING_DATA_FILE, index=False)
    print(f"\n✅ Historical data saved to: {EXISTING_DATA_FILE}")
    
    return df_1h

def fetch_realtime_aqi(lat, lon, api_key):
    """
    Fetch real-time air quality data from OpenWeatherMap API.
    Returns a dict with datetime_utc and pollutant components.
    """
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['list'][0]
            components = data['components']
            dt_unix = data['dt'] 
            dt = datetime.utcfromtimestamp(dt_unix)
            aqi = data['main']['aqi'] 
            
            new_row = {'datetime_utc': dt, 'aqi': aqi, **components}  
            print(f"✅ Fetched real-time data: {dt} (AQI: {aqi})")
            return new_row
        else:
            print(f"❌ API error fetching real-time data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Fetch error: {e}")
        return None

def update_csv_with_realtime(df_existing, new_row):
    """Appends the new real-time row to the existing DataFrame and saves the updated CSV."""
    print("\n" + "="*50)
    print("STEP 2: APPENDING REAL-TIME DATA")
    print("="*50)
    new_row_dt_utc = pd.to_datetime(new_row['datetime_utc']).tz_localize('UTC')

    if not df_existing.empty:
        df_existing['datetime_utc'] = pd.to_datetime(df_existing['datetime_utc'])
        if df_existing['datetime_utc'].dt.tz is None:
            df_existing['datetime_utc'] = df_existing['datetime_utc'].dt.tz_localize('UTC')
        latest_existing_dt = df_existing['datetime_utc'].max() 
        time_difference = (new_row_dt_utc - latest_existing_dt).total_seconds()
        
        if time_difference < 60 and time_difference >= 0:
            print("⚠️ Real-time data timestamp is too close to the existing data's latest point. Skipping append to avoid duplicates.")
            return df_existing
        
        if new_row_dt_utc < latest_existing_dt:
             print(f"⚠️ Real-time data timestamp ({new_row_dt_utc}) is older than the existing data's latest point ({latest_existing_dt}). Skipping append.")
             return df_existing

    new_row_df = pd.DataFrame([new_row])
    new_row_df['datetime_utc'] = pd.to_datetime(new_row_df['datetime_utc']).dt.tz_localize('UTC')
    
    existing_cols = df_existing.columns.tolist()
    
    for col in existing_cols:
        if col not in new_row_df.columns:
            new_row_df[col] = np.nan
    
    new_row_df = new_row_df[existing_cols] 
    df_updated = pd.concat([df_existing, new_row_df], ignore_index=True)    
    df_updated = df_updated.sort_values(by='datetime_utc').drop_duplicates(subset=['datetime_utc'], keep='last')
    df_updated.to_csv(EXISTING_DATA_FILE, index=False)
    print(f"✅ Appended real-time row. New shape: {df_updated.shape}. Latest timestamp: {df_updated['datetime_utc'].max()}")
    
    return df_updated

def calculate_us_aqi_and_save(df):
    """Calculates US AQI based on the DataFrame and saves the final output."""
    print("\n" + "="*50)
    print("STEP 3: CALCULATING US AQI")
    print("="*50)
    
    if df is None or df.empty:
        print("⚠️ Cannot calculate US AQI: DataFrame is empty or None.")
        return None    
    df['us_aqi'] = df.apply(calc_us_aqi, axis=1)
    df['us_aqi'] = np.clip(df['us_aqi'].round().astype(int), 0, 500)
    df.to_csv(FINAL_DATA_FILE, index=False)
    print(f"✅ US AQI calculated. Dataset shape: {df.shape}")
    print("\nSample of final data with US AQI:\n")
    print(df[['datetime_utc', 'aqi', 'us_aqi', 'pm2_5', 'pm10', 'co']].tail(10))
    return df

def main_pipeline():
    """Main function to run the entire data fetching and processing pipeline."""
    df_historical = fetch_historical_data(LAT, LON, API_KEY)    
    if df_historical is None:
        if os.path.exists(EXISTING_DATA_FILE):
            try:
                print(f"⚠️ Historical fetch failed or returned no data. Loading existing data from {EXISTING_DATA_FILE}.")
                df_historical = pd.read_csv(EXISTING_DATA_FILE, parse_dates=['datetime_utc'])
            except Exception as e:
                print(f"❌ Error loading existing CSV: {e}")
                sys.exit(1)
        else:
            print("❌ No historical data could be fetched and no existing CSV found. Exiting.")
            sys.exit(1)
    new_row = fetch_realtime_aqi(LAT, LON, API_KEY)
    
    if new_row:
        df_updated = update_csv_with_realtime(df_historical, new_row)
    else:
        print("⚠️ No new real-time data to append.")
        df_updated = df_historical 
    calculate_us_aqi_and_save(df_updated)

if __name__ == "__main__":
    main_pipeline()

