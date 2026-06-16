# Macroeconomic Recession Forecasting with Machine Learning and Explainable AI

## Overview

This project predicts U.S. recessions using macroeconomic indicators from the Federal Reserve Economic Data (FRED) database.

The objective is to build an interpretable machine learning pipeline that can identify economic conditions associated with future recessions.

---

## Dataset

Data was collected from FRED and includes:

- GDP (GDP)
- Consumer Price Index (CPIAUCSL)
- Unemployment Rate (UNRATE)
- Federal Funds Rate (FEDFUNDS)
- U.S. Recession Indicator (USRECQ)

---

## Feature Engineering

The following features were created:

- GDP Growth Rate
- Inflation Rate
- Unemployment Change
- Interest Rate Change
- Lagged Economic Indicators

The target variable was defined as recession status two quarters ahead.

---

## Methodology

### Data Processing

- Merged macroeconomic datasets
- Converted dates to time-series format
- Handled missing values
- Created lagged features

### Models

1. Logistic Regression
2. Random Forest Classifier

### Validation

A time-based train-test split was used to prevent data leakage.

---

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and identify the most influential macroeconomic variables.

---

## Results

- Built an end-to-end forecasting pipeline
- Evaluated Logistic Regression and Random Forest models
- Investigated class imbalance challenges in recession prediction
- Applied SHAP for model interpretability

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- SHAP

---

## Repository Structure

```text
project/
│
├── data/
├── notebook/
│   └── recession_forecasting.ipynb
├── images/
├── README.md
└── requirements.txt
```

## Future Improvements

- Add Yield Curve indicators
- Add Industrial Production Index
- Experiment with XGBoost
- Deploy using Streamlit
- Compare with Deep Learning models

## Visualizations

### Feature Importance

![Feature Importance](images/feature_importance.png)

### SHAP Summary Plot

![SHAP Plot](images/shap_summary.png)

## Author

Deepanshi Madhukar

Economics Student | Aspiring Data Scientist | Machine Learning & Forecasting
