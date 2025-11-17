import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR = "final_outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)



model = joblib.load(os.path.join(OUTPUTS_DIR,"rf_final_model.joblib"))
scaler = joblib.load(os.path.join(OUTPUTS_DIR,"scaler_rf_final.joblib"))
df_2008 = pd.read_csv("outputs_2008/prepared_2008_data.csv", parse_dates=["DateTime"])

FEATURES = [
    "Temp_centered", "Temp_sq", "Temp_cubed",
    "HDD", "CDD", "Temp_roll3", "Temp_roll24",
    "hour_sin", "hour_cos",
    "Month", "DayOfWeek", "DayOfYear", "Quarter", "IsWeekend", "IsHoliday"
]

# Scale features
CONT_FEATURES = ["Temp_centered", "Temp_sq", "Temp_cubed", "HDD", "CDD", "Temp_roll3", "Temp_roll24"]
df_2008.loc[:, CONT_FEATURES] = scaler.transform(df_2008[CONT_FEATURES])

# Predict probabilities
y_proba = model.predict_proba(df_2008[FEATURES])[:,1]
df_2008["Predicted_Probability"] = y_proba

# Pick single peak per day
df_2008["Predicted_Peak"] = 0
idx = df_2008.groupby(df_2008["DateTime"].dt.date)["Predicted_Probability"].idxmax()
df_2008.loc[idx,"Predicted_Peak"] = 1

# Prepare output CSV
output_df = df_2008.copy()
output_df["Hour"] = output_df["DateTime"].dt.hour + 1
output_df["Date"] = output_df["DateTime"].dt.date
output_df = output_df[["Date","Hour","Predicted_Probability","Predicted_Peak"]]
output_df.to_csv(os.path.join(OUTPUTS_DIR,"predictions_2008_pivot.csv"),index=False)

# Summary
total_hours = len(output_df)
num_peaks = output_df["Predicted_Peak"].sum()
print(f"Total predicted peaks in 2008: {num_peaks}/{total_hours} ({num_peaks/total_hours:.2%})")

# Heatmap
pivot_peak = output_df.pivot(index="Date", columns="Hour", values="Predicted_Peak")
plt.figure(figsize=(16,10))
sns.heatmap(pivot_peak,cmap="Reds",cbar_kws={"label":"Predicted Peak"})
plt.title("Predicted Peaks Heatmap 2008")
plt.show()

