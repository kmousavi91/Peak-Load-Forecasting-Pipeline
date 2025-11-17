import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import holidays


LOAD_PATH = "load_hist_data.csv"
WEATHER_PATH = "weather_data.csv"
OUTPUT_FILE = "prepared_training_data.csv"

sns.set(style="whitegrid")

PLOTS_DIR = 'analysis_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(OUTPUT_FILE)

# Temperature stats
print(f"Mean Temperature: {df['Temp'].mean():.2f}")
print(f"Temperature Variance: {df['Temp'].var():.2f}")

# Scatter plot Temp vs Load
plt.figure(figsize=(10, 6))
plt.scatter(df['Temp'], df['Load'], alpha=0.5)
plt.title('Energy Load vs Temperature')
plt.xlabel('Temperature (Temp)')
plt.ylabel('Energy Load (Load)')
plt.grid(True)
plt.show()

# Peak hour percentage
total_hours = len(df)
num_peaks = df['is_peak_hour'].sum()
print(f"Total peak hours: {num_peaks} / {total_hours} ({num_peaks/total_hours*100:.2f}%)")

# Hourly peak frequency
hourly_peak_counts = df.groupby('Hour')['is_peak_hour'].sum()
hourly_peak_percentage = hourly_peak_counts / df.groupby('Hour')['is_peak_hour'].count() * 100
plt.figure(figsize=(12,6))
sns.barplot(x=hourly_peak_percentage.index, y=hourly_peak_percentage.values, palette="Reds")
plt.title('Percentage of Peak Hours per Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Percentage of Peaks (%)')
plt.show()

# Boxplot Load vs Peak/Non-Peak
plt.figure(figsize=(8,6))
sns.boxplot(x='is_peak_hour', y='Load', data=df, palette="Set2")
plt.title('Load Distribution: Peak vs Non-Peak Hours')
plt.show()

# Correlation heatmap
numeric_cols = ['Temp', 'Temp_centered', 'Temp_sq', 'Temp_cubed', 'HDD', 'CDD', 'Temp_roll3', 'Temp_roll24', 'Load']
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

