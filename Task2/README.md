# Individual Task 2, Part 2: Deliberation on Task 1 (COSC2669)

Re-evaluation of the two models built in Individual Task 1, covering evaluation bias, learning curves and a fairness audit.

| Model | Dataset | Algorithm |
|---|---|---|
| 1. Sales forecasting | [Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) | Random Forest Regressor |
| 2. Customer segmentation | [Mall Customer Segmentation](https://www.kaggle.com/datasets/hosseinbadrnezhad/mall-customer-segmentation-dataset) | K-Means (k = 5) |

## What the script does

1. **Evaluation bias (Q1)**
   - Diagnoses the Task 1 train/test split, which split by row order rather than by date.
   - Re-evaluates Model 1 with a date-based hold-out and rolling-origin cross-validation, compared against naive and seasonal-naive baselines.
   - Assesses Model 2 stability with bootstrap resampling and the Adjusted Rand Index.
2. **Learning curves (Q2)**
   - Model 1: forecast error against the amount of training history.
   - Model 2: cluster stability and silhouette score against sample size.
3. **Fairness audit (Q3)**, using [Fairlearn](https://fairlearn.org/)
   - Model 2: selection rate and demographic parity by age group and gender for a promotion-targeting scenario, repeated with Age removed to test for proxy effects.
   - Model 1: forecast error by product category.

## Setup

```bash
pip install -r requirements.txt
```

Download both datasets from Kaggle and place them as follows:

```
data/
├── superstore/train.csv
└── mall_customer/store_customers.csv
```

## Run

```bash
python part2_analysis.py
```

Results are written to `outputs/task2/`:

| File | Content |
|---|---|
| `part2_results.txt` | All numerical results |
| `fig_split_diagnosis.png` | Original vs corrected train/test split |
| `fig_learning_curves.png` | Learning curves for both models |
| `fig_fairness_age.png` | Selection rate by age group |

Runtime is about one minute. Results were produced with fairlearn 0.14.0.

## Related

Task 1 models: `model1_sales_forecast.py`, `model2_customer_segmentation.py`.
