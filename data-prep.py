# Imports
import os
import pandas as pd
import numpy as np
import holidays

# Paths
LOAD_PATH = "load_hist_data.csv"
WEATHER_PATH = "weather_data.csv"
OUTPUT_FILE = "prepared_training_data.csv"

# -------------------
# Aggregate weather
# -------------------
def aggregate_weather(weather_df):
    weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    weather_df["DateTime"] = weather_df["Date"] + pd.to_timedelta(weather_df["Hour"] - 1, unit="h")
    weather_agg = (
        weather_df.groupby(["DateTime"])
        .agg(Temp=("Temperature", "mean"))
        .reset_index()
    )
    return weather_agg

# -------------------
# Merge load + weather
# -------------------
def merge_load_weather(load_df, weather_agg):
    load_df["Date"] = pd.to_datetime(load_df["Date"])
    load_df["DateTime"] = load_df["Date"] + pd.to_timedelta(load_df["Hour"] - 1, unit="h")
    merged_df = pd.merge(load_df, weather_agg, on="DateTime", how="inner")
    return merged_df

# -------------------
# Feature engineering
# -------------------
def add_features(df):
    df = df.sort_values("DateTime").reset_index(drop=True)
    
    # Rolling averages
    df["Temp_roll3"] = df["Temp"].rolling(window=3, min_periods=1).mean()
    df["Temp_roll24"] = df["Temp"].rolling(window=24, min_periods=1).mean()
    
    # HDD & CDD
    df["HDD"] = (65 - df["Temp"]).clip(lower=0)
    df["CDD"] = (df["Temp"] - 65).clip(lower=0)
    
    # Polynomial terms
    df["Temp_centered"] = df["Temp"] - 65
    df["Temp_sq"] = df["Temp_centered"] ** 2
    df["Temp_cubed"] = df["Temp_centered"] ** 3
    
    # Time features
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfYear"] = df["DateTime"].dt.dayofyear
    df["Quarter"] = df["DateTime"].dt.quarter
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    
    # Holiday flag
    us_holidays = holidays.US(years=df["Year"].unique())
    holiday_dates = pd.to_datetime(list(us_holidays.keys()))
    df["IsHoliday"] = df["DateTime"].dt.normalize().isin(holiday_dates).astype(int)
    
    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    
    # Target: daily peak hour
    def assign_daily_peak(group):
        is_peak = np.zeros(len(group), dtype=int)
        max_pos = np.argmax(group['Load'].values)
        is_peak[max_pos] = 1
        group['is_peak_hour'] = is_peak
        return group
    
    df = df.groupby(df["DateTime"].dt.date).apply(assign_daily_peak).reset_index(drop=True)
    
    return df

# -------------------
# Main pipeline
# -------------------
load_df = pd.read_csv(LOAD_PATH)
weather_df = pd.read_csv(WEATHER_PATH)

weather_agg = aggregate_weather(weather_df)
merged_df = merge_load_weather(load_df, weather_agg)
final_df = add_features(merged_df)
final_df = final_df[(final_df["Year"] >= 2005) & (final_df["Year"] <= 2007)]

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Prepared dataset saved to {OUTPUT_FILE} with shape {final_df.shape}")

