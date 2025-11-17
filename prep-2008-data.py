# Imports
import os
import pandas as pd
import numpy as np
import holidays

# Paths
LOAD_PATH = "load_hist_data.csv"
WEATHER_PATH = "weather_data.csv"

OUTPUT_DIR_2008 = "outputs_2008"
os.makedirs(OUTPUT_DIR_2008, exist_ok=True)

weather_df = pd.read_csv(WEATHER_PATH)
weather_df["Date"] = pd.to_datetime(weather_df["Date"])
weather_2008 = weather_df[weather_df["Date"].dt.year == 2008].copy()

# Aggregate and feature engineering
weather_agg = weather_2008.groupby(["Date","Hour"]).agg(Temp=("Temperature","mean")).reset_index()
weather_agg["DateTime"] = weather_agg["Date"] + pd.to_timedelta(weather_agg["Hour"]-1, unit="h")
weather_agg = weather_agg.sort_values("DateTime").reset_index(drop=True)

weather_agg["Temp_roll3"] = weather_agg["Temp"].rolling(3,min_periods=1).mean()
weather_agg["Temp_roll24"] = weather_agg["Temp"].rolling(24,min_periods=1).mean()
weather_agg["HDD"] = (65-weather_agg["Temp"]).clip(0)
weather_agg["CDD"] = (weather_agg["Temp"]-65).clip(0)
weather_agg["Temp_centered"] = weather_agg["Temp"]-65
weather_agg["Temp_sq"] = weather_agg["Temp_centered"]**2
weather_agg["Temp_cubed"] = weather_agg["Temp_centered"]**3

weather_agg["Month"] = weather_agg["DateTime"].dt.month
weather_agg["DayOfWeek"] = weather_agg["DateTime"].dt.dayofweek
weather_agg["Hour"] = weather_agg["DateTime"].dt.hour
weather_agg["DayOfYear"] = weather_agg["DateTime"].dt.dayofyear
weather_agg["Quarter"] = weather_agg["DateTime"].dt.quarter
weather_agg["IsWeekend"] = (weather_agg["DayOfWeek"]>=5).astype(int)

us_holidays = holidays.US(years=[2008])
holiday_dates = pd.to_datetime(list(us_holidays.keys()))
weather_agg["IsHoliday"] = weather_agg["DateTime"].dt.normalize().isin(holiday_dates).astype(int)

weather_agg["hour_sin"] = np.sin(2*np.pi*weather_agg["Hour"]/24)
weather_agg["hour_cos"] = np.cos(2*np.pi*weather_agg["Hour"]/24)

OUTPUT_FILE_2008 = os.path.join(OUTPUT_DIR_2008,"prepared_2008_data.csv")
weather_agg.to_csv(OUTPUT_FILE_2008,index=False)
weather_agg.head()

