# EV Charging Demand Prediction and Usage Pattern Analysis

An end-to-end machine-learning project for exploring electric-vehicle charging-session data, estimating session-level demand, classifying sessions into High Demand and Normal Demand, and forecasting aggregate hourly demand. An interactive Streamlit dashboard brings the analysis, model outputs, and explanations together in one application.

The dashboard loads saved model pipelines and analysis reports from the project directories. It does not retrain models when the application starts.

## Overview

The project addresses four related tasks:

- **Usage-pattern analysis:** explore charging records by location, charger type, charger company, and time.
- **Demand regression:** estimate the demand associated with an individual charging session.
- **Demand classification:** classify a session as High Demand or Normal Demand.
- **Time-series forecasting:** analyze hourly aggregate demand and compare simple forecasting methods.

The project also includes model evaluation and explainability artifacts to support interpretation of the results.

## Interactive dashboard

Run the Streamlit application with `uv run streamlit run app.py`. The dashboard contains four sections:

1. **Overview & Usage Patterns** — usage KPIs, recorded demand by location and charger type, time-based trends, demand-category distribution, and filters for date, location, charger company, and charger type.
2. **Predict Demand** — a manual-entry form for numerical demand estimation and High Demand / Normal Demand classification. The form derives time-based features from the selected session start date and time. Duration-free and duration-informed predictions are presented as distinct scenarios.
3. **Trends & Forecasts** — historical aggregate demand and a held-out comparison of a 24-hour moving average with simple exponential smoothing.
4. **Model Performance & Explainability** — regression and classification comparisons, evaluation outputs, confusion matrices, feature-importance reports, and saved SHAP visualizations.

Predictions are entered through the form; the dashboard does not provide a CSV-upload workflow.

## Repository structure

```text
Ev_Charging_Demand_Prediction/
├── app.py
├── data/
│   ├── raw/                         # Original charging-session dataset
│   └── processed/
│       ├── ev_charging_cleaned.csv
│       └── ev_charging_features.csv
├── models/
│   ├── lightgbm_without_duration_pipeline.joblib
│   ├── lightgbm_with_duration_pipeline.joblib
│   ├── lightgbm_classifier_without_duration_pipeline.joblib
│   └── lightgbm_classifier_with_duration_pipeline.joblib
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_regression.ipynb
│   ├── 05_classification.ipynb
│   └── 06_time_series.ipynb
├── reports/
│   ├── regression_model_comparison_chronological.csv
│   ├── regression_test_predictions.csv
│   ├── classification_model_comparison_chronological.csv
│   ├── classification_baseline_metrics.csv
│   ├── classification_confusion_matrix_without_duration.csv
│   ├── classification_confusion_matrix_with_duration.csv
│   ├── classification_test_predictions.csv
│   ├── time_series_forecast_comparison.csv
│   ├── time_series_test_forecasts.csv
│   ├── regression_permutation_importance_*.csv
│   ├── regression_shap_*.csv / .png
│   └── classification_shap_*.csv / .png
├── pyproject.toml
├── uv.lock
└── README.md
```

Keep the original dataset in `data/raw/`. The cleaned and engineered datasets are stored in `data/processed/`. The model files and reports must remain at the listed relative paths because the application loads them from these directories.

## Dataset and preparation

The project uses a CSV dataset of EV charging-session records with fields such as `UserID`, `ChargerID`, `ChargerCompany`, `Location`, `ChargerType`, start and end date/time information, `Duration`, and `Demand`. The original dataset is located in `data/raw/`.

### Data preparation summary

The cleaning and feature-engineering notebooks reported the following results:

- Original dataset: **72,856 rows**.
- Removed **13** extra duplicate occurrences and **17** records where the end timestamp preceded the start timestamp.
- Cleaned dataset: **72,826 rows**.
- Engineered dataset: **72,826 rows × 24 columns**.
- `DurationValidated` is available for **69,450** records; **3,376 records (4.64%)** do not have a validated duration value.
- Session start times range from **2021-09-30 17:14:26** to **2022-09-30 23:11:59**.

Engineered time features include start month, day of month, hour, day name, weekend indicator, and time-of-day category. The time-of-day categories are Night (00:00–05:59), Morning (06:00–11:59), Afternoon (12:00–16:59), and Evening (17:00–23:59).

The classification target uses the dataset median threshold of **14.1**: `High Demand` when `Demand >= 14.1` and `Normal Demand` when `Demand < 14.1`. The engineered dataset contains **36,439 High Demand** records and **36,387 Normal Demand** records.

> **Demand units:** The available project specification does not define the physical units or exact domain meaning of `Demand`. The application therefore displays the recorded demand values without labeling them as kW or kWh.

## Workflow and notebooks

The notebooks are intended to be run in this order:

1. `notebooks/01_data_cleaning.ipynb` — quality checks, timestamp validation, duplicate handling, and cleaning.
2. `notebooks/02_feature_engineering.ipynb` — temporal features and the High Demand / Normal Demand target.
3. `notebooks/03_eda.ipynb` — distributions, usage patterns, and relationships in the data.
4. `notebooks/04_regression.ipynb` — regression model comparison, evaluation, and explanations.
5. `notebooks/05_classification.ipynb` — classification model comparison, evaluation, and SHAP analysis.
6. `notebooks/06_time_series.ipynb` — hourly aggregation, stationarity check, and forecasting evaluation.

The notebooks generate processed data, fitted pipelines, and reports used by the dashboard. The app itself does not require the notebooks to be open or running.

## Methods

### Regression

Linear Regression, Decision Tree, Random Forest, XGBoost, and LightGBM were compared using a chronological train/test split. Two feature sets were evaluated:

- **Without duration:** charger/location categories and features available from the session start date and time.
- **With duration:** the same features plus `DurationValidated`.

The selected LightGBM pipelines are saved in `models/` and loaded by the dashboard.

### Classification

Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM were compared. The target is encoded as Normal Demand = 0 and High Demand = 1. Evaluation includes Accuracy, Precision, Recall, F1, and ROC-AUC. High Demand is the positive class for Precision, Recall, and F1 reporting.

### Time-series forecasting

Demand is aggregated by the hour in which each session starts. The chronological test period contains **1,754 hourly observations**. Forecasting was evaluated using rolling one-step-ahead predictions from a 24-hour moving average and simple exponential smoothing. These forecasts estimate **aggregate hourly demand**, not individual session demand.

### Explainability

The project includes regression permutation-importance reports, regression SHAP reports and plots, and classification SHAP reports and plots. LIME explanations are not included in the current implementation.

## Model results

The following values are from the saved chronological test-set comparison reports. Values are rounded for readability.

### Regression — LightGBM

| Feature set | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Without duration | 9.9416 | 170.1070 | 13.0425 | 0.1212 |
| With duration | 4.3466 | 58.4520 | 7.6454 | 0.6980 |

The duration-informed model has lower error on this evaluation. However, actual duration is generally not known when a session starts, so these results should not be presented as session-start performance. Use the duration-free pipeline for session-start estimates. If a user provides an estimated duration, clearly identify the prediction as conditional on that estimate.

The duration-free model has a modest R² score; its output should be interpreted as an estimate with meaningful uncertainty rather than a precise measurement.

### Classification — LightGBM

| Feature set | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Without duration | 0.6468 | 0.6203 | 0.7195 | 0.6662 | 0.6910 |
| With duration | 0.8720 | 0.8267 | 0.9347 | 0.8774 | 0.9448 |

The duration-informed metrics are conditional on `DurationValidated` and should not be interpreted as the performance of a model using only information available at session start.

### Hourly forecasting

| Method | MAE | RMSE |
|---|---:|---:|
| Simple Exponential Smoothing | 81.2079 | 107.6907 |
| 24-hour Moving Average | 110.7353 | 136.6587 |

Simple exponential smoothing performed better than the moving-average baseline on the held-out chronological period. These aggregate hourly-demand metrics are not directly comparable to the session-level regression metrics.

## Environment setup

The project uses Python and `uv` for dependency and environment management. The working environment used Python **3.13.6**. Keep `pyproject.toml` and `uv.lock` with the source code to support a reproducible setup.

### 1. Install `uv`

Follow the official installation instructions: <https://docs.astral.sh/uv/getting-started/installation/>.

### 2. Open a terminal at the project root

Example for Windows PowerShell:

```powershell
cd "C:\path\to\Ev_Charging_Demand_Prediction"
```

Use the path where you cloned or stored the project if it differs from this example.

### 3. Install project dependencies

From the directory containing `pyproject.toml` and `uv.lock`, run:

```powershell
uv sync
```

For a new environment, this installs the dependencies declared in the project manifest and lockfile. Model artifacts rely on compatible versions of scikit-learn, LightGBM, and their associated dependencies; retain the lockfile when sharing or deploying the project.

### 4. Launch the dashboard

Run the following command from the project root:

```powershell
uv run streamlit run app.py
```

Streamlit prints a local URL in the terminal, commonly `http://localhost:8501`. Open it in a browser. Stop the server with `Ctrl+C` in the terminal.

## Limitations and interpretation

- **Unspecified demand units:** the source material does not establish physical units for `Demand`.
- **Duration availability:** `DurationValidated` is generally known only after a session ends. Duration-informed metrics are not evidence of session-start prediction performance.
- **Moderate duration-free performance:** the no-duration LightGBM achieved R² = 0.1212 for regression and ROC-AUC = 0.6910 for classification on the chronological test set.
- **Fixed classification threshold:** the target uses the dataset median value of 14.1. For future deployments, define and version a threshold based on the intended reference/training data rather than silently recalculating it on each new batch.
- **No hyperparameter tuning:** models were compared without additional tuning to prioritize an end-to-end working application.
- **Explainability scope:** permutation importance and SHAP are included; LIME is not implemented.
- **Limited time coverage:** the dataset covers approximately one year. Performance may differ for future periods, new locations, or categories not represented in training.
- **Forecasting scope:** the moving average and exponential smoothing methods are simple baselines, not operational capacity-planning guarantees.
