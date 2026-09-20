# Individual Task 1 — Data Analysis (COSC2669/COSC2816)

Source code for the data analysis component of Individual Task 1, Part 1 — submitted as part of the executive summary report.

**Role context:** Data Scientist (Accenture, Job No. 14578388) — see full report for role analysis.

## Models

**1. Sales Forecasting** (`model1_sales_forecast.py`)
Random Forest Regressor predicting monthly retail sales per product category, using lag features, rolling averages, and calendar features. Trained on the [Superstore Sales dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting).

**2. Customer Segmentation** (`model2_customer_segmentation.py`)
K-Means clustering to segment customers by age, income, and spending score, with elbow-method and silhouette-score based selection of k. Trained on the [Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/hosseinbadrnezhad/mall-customer-segmentation-dataset).

## Setup

```bash
pip install -r requirements.txt
```

Download both datasets from the Kaggle links above and place them in `data/superstore/train.csv` and `data/mall_customer/store_customers.csv`.

## Run

```bash
python model1_sales_forecast.py
python model2_customer_segmentation.py
```

Outputs (metrics + plots) are saved to `outputs/`.

## Results Summary

| Model | Metric | Value |
|---|---|---|
| Sales Forecasting (Random Forest) | MAPE | 33.1% |
| Sales Forecasting (Random Forest) | R² | 0.122 |
| Customer Segmentation (K-Means, k=5) | Silhouette Score | 0.329 |

Full discussion of results, evaluation metric justification, and limitations is provided in the executive summary report (not included in this repository).
