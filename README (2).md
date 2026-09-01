# Macroeconomic Recession Prediction

A machine learning project that predicts US recessions 2 quarters ahead using core macroeconomic indicators — GDP growth, inflation, unemployment, and interest rates.

## Overview

This project builds a binary classifier to forecast whether the US economy will be in a recession two quarters from now, using only publicly available macroeconomic data from FRED (Federal Reserve Economic Data). It walks through the full pipeline: data cleaning, feature engineering, model selection, and — most importantly — a realistic evaluation using time-series cross-validation, which surfaces a common and instructive failure mode in imbalanced classification problems.

## Data Sources

All data is quarterly, sourced from FRED:

| Series | Description |
|---|---|
| `GDP` | Gross Domestic Product |
| `CPIAUCSL` | Consumer Price Index (inflation) |
| `UNRATE` | Unemployment Rate |
| `FEDFUNDS` | Federal Funds Rate (interest rate) |
| `USRECQ` | US Recession Indicator (NBER-based, quarterly) |

## Features

Raw series are transformed into economically meaningful signals:

- `gdp_growth` — quarter-over-quarter % change in GDP
- `inflation` — quarter-over-quarter % change in CPI
- `unemp_change` — change in unemployment rate
- `rate_change` — change in the federal funds rate
- One-quarter lagged versions of each of the above (`*_lag1`)

**Target:** `recession`, shifted 2 quarters forward — i.e., the model predicts a recession 2 quarters before it happens, framing this as an early-warning system rather than a nowcast.

## Models

Three models were trained and compared:

1. **Logistic Regression** — baseline linear model
2. **Random Forest** — default settings
3. **Random Forest (class-balanced)** — `class_weight="balanced"` to address the ~12% positive class rate

Feature importance and SHAP values were used to interpret each model's decisions.

## Key Findings

This project's main insight came from evaluating the model *properly*, not from a single train/test split:

- With a naive 80/20 split and the default 0.5 classification threshold, models looked reasonable (~93–96% accuracy) — but that accuracy was almost entirely driven by the majority "no recession" class. Recall on actual recessions was close to 0%, similar to a naive baseline that always predicts "no recession."
- Switching to **`TimeSeriesSplit`** (walk-forward validation, which respects the chronological nature of the data) exposed this clearly: 4 of 5 folds had **0.00 recall**, meaning the model missed every real recession in those folds.
- Digging into predicted *probabilities* rather than hard labels showed the model did have real signal — **ROC-AUC ranged from 0.60–0.89** across folds — it just rarely pushed predictions above the default 0.5 cutoff, a classic symptom of class imbalance (~12% positive class).
- Sweeping the classification threshold via a precision-recall curve and settling on **threshold = 0.19** shifted the model to prioritize recall over precision — an appropriate tradeoff for an early-warning tool, where missing a real recession is more costly than a false alarm. This lifted recall to 0.33–0.83 across most folds (one fold with only 2 test recessions remained unreliable).

**Takeaway:** a model that looks strong by accuracy alone can be silently useless for the minority class it's actually meant to catch. Proper time-aware validation and threshold tuning were essential to get a usable result.

## Results Summary

| Approach | Accuracy | Recall (recession) | Notes |
|---|---|---|---|
| Naive baseline (always "no recession") | 96.5% | 0% | Reference point |
| Logistic Regression (single split) | 94.7% | 0% | Misleading due to class imbalance |
| Random Forest, balanced (single split) | 95.6% | 0% | Same issue |
| Random Forest, balanced, TimeSeriesSplit, default threshold | ~93–96% | 0% (4/5 folds) | Reveals the real problem |
| Random Forest, balanced, TimeSeriesSplit, threshold=0.19 | Lower | 33–83% (4/5 folds) | Tuned for early-warning use case |

## Project Structure

```
.
├── Recession_forecasting.ipynb   # Full analysis notebook
├── Data/                          # FRED CSVs (GDP, CPI, UNRATE, FEDFUNDS, USRECQ)
└── README.md
```

## Getting Started

1. Clone the repo and download the FRED series above into a `Data/` folder (or update the file paths in the notebook to point to your own location).
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn shap
   ```
3. Run `Recession_forecasting.ipynb` end to end.

## Limitations & Future Work

- Recession quarters are rare in the historical record, so even with time-series CV, some folds contain very few positive examples, making metrics noisy.
- Only four macro indicators are used; adding yield curve spread, credit spreads, or leading economic indices could improve signal.
- The model is not tuned for real-time/vintage data issues (FRED series are often revised after the fact), which would matter for genuine real-time deployment.
- Next steps could include gradient boosting models, probability calibration, and backtesting against specific historical recessions (2001, 2008, 2020).

## Disclaimer

This project is for educational and research purposes only and is not intended as financial or investment advice.
