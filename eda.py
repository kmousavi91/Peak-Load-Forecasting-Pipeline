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

PLOTS_DIR = "analysis_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(OUTPUT_FILE)

# --- Create a file to store textual results ---
results_path = os.path.join(PLOTS_DIR, "eda_results.txt")
results_file = open(results_path, "w")

def log(message):
    print(message)
    results_file.write(message + "\n")

# Temperature stats
log(f"Mean Temperature: {df['Temp'].mean():.2f}")
log(f"Temperature Variance: {df['Temp'].var():.2f}")

# Scatter plot Temp vs Load
plt.figure(figsize=(10, 6))
plt.scatter(df['Temp'], df['Load'], alpha=0.5)
plt.title('Energy Load vs Temperature')
plt.xlabel('Temperature (Temp)')
plt.ylabel('Energy Load (Load)')
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "scatter_temp_vs_load.png"))
plt.show()

# Peak hour percentage
total_hours = len(df)
num_peaks = df['is_peak_hour'].sum()
log(f"Total peak hours: {num_peaks} / {total_hours} ({num_peaks/total_hours*100:.2f}%)")

# Hourly peak frequency
hourly_peak_counts = df.groupby('Hour')['is_peak_hour'].sum()
hourly_peak_percentage = hourly_peak_counts / df.groupby('Hour')['is_peak_hour'].count() * 100

plt.figure(figsize=(12,6))
sns.barplot(x=hourly_peak_percentage.index, y=hourly_peak_percentage.values, palette="Reds")
plt.title('Percentage of Peak Hours per Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Percentage of Peaks (%)')
plt.savefig(os.path.join(PLOTS_DIR, "hourly_peak_percentage.png"))
plt.show()

# Boxplot Load vs Peak/Non-Peak
plt.figure(figsize=(8,6))
sns.boxplot(x='is_peak_hour', y='Load', data=df, palette="Set2")
plt.title('Load Distribution: Peak vs Non-Peak Hours')
plt.savefig(os.path.join(PLOTS_DIR, "boxplot_load_peak_vs_nonpeak.png"))
plt.show()

# Correlation heatmap
numeric_cols = ['Temp', 'Temp_centered', 'Temp_sq', 'Temp_cubed', 'HDD', 'CDD', 'Temp_roll3', 'Temp_roll24', 'Load']
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
plt.show()

# Close results file
results_file.close()
print(f"All plots saved to: {PLOTS_DIR}")
print(f"All analysis results saved to: {results_path}")

