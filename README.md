# **Peak Load Forecasting Pipeline**

<p align="center">
  <img src="project-logo.png" alt="Project Logo" width="200" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Model-Gradient%20Boosting-orange" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success" />
</p>

---

## 🚀 **Project Overview**

This repository contains a complete machine-learning pipeline for **electricity peak load forecasting**, designed for clarity, reproducibility, and real-world deployment. It includes:

* Full data preparation workflows
* Exploratory Data Analysis (EDA)
* Model training and evaluation
* Predictive deployment scripts
* Visualization utilities
* Year 2008 forecasting pipeline

The objective is to model and predict **peak load events** using engineered temporal and weather features.

---

## 📂 **Repository Structure**

```
├── data-prep.py              # Creates final training dataset from raw files
├── prep-2008-data.py         # Prepares 2008 forecasting dataset
├── eda.py                    # Exploratory data analysis & visualizations
├── train.py                  # Model training, evaluation, and saving
├── deploy.py                 # Loads trained model and generates predictions
├── load_hist_data.csv        # Historical load data (input)
├── weather_data.csv          # Historical weather data (input)
└── README.md                 # Project documentation
```

---

## 🧠 **Key Features**

### **1. Data Preparation**

Scripts: `data-prep.py`, `prep-2008-data.py`

* Cleans raw load + weather data
* Builds temporal features (hour, day, month, season, weekday)
* Merges holiday information
* Flags peak events using percentile thresholds
* Outputs model-ready datasets

### **2. Exploratory Data Analysis (EDA)**

Script: `eda.py`

Generates:

* Load distribution plots
* Weather correlations
* Peak-event heatmaps
* Seasonal & weekly patterns
* Correlation matrices

### **3. Model Training**

Script: `train.py`

Model: **Gradient Boosting Classifier** (sklearn)

Includes:

* Train/test split
* Feature selection
* Evaluation metrics:

  * Accuracy
  * Precision/Recall
  * F1 Score
  * ROC Curve + AUC
  * Log Loss
  * Confusion Matrix

Saves model as:

```
model.pkl
```

### **4. Deployment / Inference**

Script: `deploy.py`

* Loads trained model & 2008 dataset
* Predicts:

  * Binary classes
  * Prediction probabilities
* Generates heatmaps & visual summaries
* Saves:

```
predicted_2008.csv
```

---

## 🛠️ **Tech Stack**

* Python 3.10+
* Pandas, NumPy
* Scikit-Learn
* Matplotlib, Seaborn
* Holidays (calendar feature engineering)
* Joblib

---

## 📘 **How to Run the Project**

### **1. Install Requirements**

```bash
pip install -r requirements.txt
```

### **2. Prepare Training Data**

```bash
python data-prep.py
```

### **3. Perform EDA**

```bash
python eda.py
```

### **4. Train the Model**

```bash
python train.py
```

### **5. Prepare 2008 Data**

```bash
python prep-2008-data.py
```

### **6. Generate Predictions**

```bash
python deploy.py
```

---

## 📈 **Example Visualizations**

> Place your generated images in a folder such as `images/` and reference them here.

* **Correlation Heatmap**
  ![](images/correlation_heatmap.png)

* **Seasonal Peak Pattern Heatmap**
  ![](images/seasonal_heatmap.png)

* **ROC Curve**
  ![](images/roc_curve.png)

* **Confusion Matrix**
  ![](images/confusion_matrix.png)

---

## 📦 **requirements.txt**

Use the following dependencies:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
holidays
joblib
```

---

## 🪪 **License**

This project is licensed under the **MIT License**.

---

## 🤝 **Contributing**

Contributions, feature ideas, and pull requests are welcome!

---

## ⭐ **If you use this project, please give it a star!**
