import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve


LOAD_PATH = "load_hist_data.csv"
WEATHER_PATH = "weather_data.csv"
OUTPUT_FILE = "prepared_training_data.csv"

OUTPUT_DIR = "final_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

df = pd.read_csv(OUTPUT_FILE)

# Train/Test Split
train_df = df[df["Year"] <= 2006]
test_df = df[df["Year"] == 2007]

TARGET = "is_peak_hour"
FEATURES = [
    "Temp_centered", "Temp_sq", "Temp_cubed",
    "HDD", "CDD", "Temp_roll3", "Temp_roll24",
    "hour_sin", "hour_cos",
    "Month", "DayOfWeek", "DayOfYear", "Quarter", "IsWeekend", "IsHoliday"
]

X_tr = train_df[FEATURES]
y_tr = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

CONT_FEATURES = ["Temp_centered", "Temp_sq", "Temp_cubed", "HDD", "CDD",
                 "Temp_roll3", "Temp_roll24"]

scaler = StandardScaler()
X_tr.loc[:, CONT_FEATURES] = scaler.fit_transform(X_tr[CONT_FEATURES])
X_test.loc[:, CONT_FEATURES] = scaler.transform(X_test[CONT_FEATURES])

# Train RF
best_params = {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4,
               'max_features': None, 'max_depth': None}
model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
model.fit(X_tr, y_tr)

# Save model & scaler
joblib.dump(model, os.path.join(OUTPUT_DIR, "rf_final_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_rf_final.joblib"))

# Predict
best_threshold = 0.49
y_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= best_threshold).astype(int)

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "log_loss": log_loss(y_test, y_proba),
    "mean_squared_error_rule": ((y_proba - y_test.values) ** 2).mean()
}
print(metrics)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)
plt.show()

# Feature Importance
import seaborn as sns
fi = model.feature_importances_
fi_df = pd.DataFrame({"feature": X_tr.columns, "importance": fi}).sort_values("importance", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x="importance", y="feature", data=fi_df.head(20))
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_val:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

