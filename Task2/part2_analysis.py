"""
Individual Task 2, Part 2: Deliberation on Task 1.

Re-evaluates the two Individual Task 1 models:
  Model 1  Random Forest sales forecast (Superstore Sales)
  Model 2  K-Means customer segmentation (Mall Customer Segmentation)

Analyses:
  Q1  Evaluation bias: split diagnosis, date-based hold-out, rolling-origin CV,
      naive baselines, and bootstrap stability for clustering
  Q2  Learning curves for both models
  Q3  Fairness audit with Fairlearn

Usage:
    python part2_analysis.py

Outputs are written to outputs/task2/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, count, demographic_parity_difference,
                               demographic_parity_ratio, selection_rate)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (adjusted_rand_score, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUT_DIR = "outputs/task2"
SUPERSTORE_PATH = "data/superstore/train.csv"
MALL_PATH = "data/mall_customer/store_customers.csv"

os.makedirs(OUT_DIR, exist_ok=True)
_log = open(f"{OUT_DIR}/part2_results.txt", "w")


def report(*args):
    """Print to console and to the results file."""
    print(*args)
    print(*args, file=_log)


def section(title):
    report("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def period_label(period):
    return f"{(period - 1) // 12}-{(period - 1) % 12 + 1:02d}"


def random_forest(seed=RANDOM_STATE):
    return RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed)


# -----------------------------------------------------------------------------
# Model 1: data preparation (identical to Task 1)
# -----------------------------------------------------------------------------
sales = pd.read_csv(SUPERSTORE_PATH)
sales["Order Date"] = pd.to_datetime(sales["Order Date"], format="%d/%m/%Y")
sales["Year"] = sales["Order Date"].dt.year
sales["Month"] = sales["Order Date"].dt.month

monthly = (sales.groupby(["Year", "Month", "Category"])["Sales"].sum()
                .reset_index()
                .sort_values(["Category", "Year", "Month"]))
by_cat = monthly.groupby("Category")["Sales"]
monthly["Sales_Lag1"] = by_cat.shift(1)
monthly["Sales_Lag2"] = by_cat.shift(2)
monthly["Sales_Lag12"] = by_cat.shift(12)  # seasonal-naive baseline only, not a model feature
monthly["Sales_RollMean3"] = by_cat.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
monthly["Quarter"] = (monthly["Month"] - 1) // 3 + 1
monthly = monthly.dropna(subset=["Sales_Lag1", "Sales_Lag2"]).reset_index(drop=True)
monthly["Category_enc"] = monthly["Category"].astype("category").cat.codes
monthly["Period"] = monthly["Year"] * 12 + monthly["Month"]

FEATURES = ["Year", "Month", "Quarter", "Category_enc",
            "Sales_Lag1", "Sales_Lag2", "Sales_RollMean3"]

# -----------------------------------------------------------------------------
# Q1a. Diagnose the original Task 1 split
# -----------------------------------------------------------------------------
section("Q1a. DIAGNOSIS OF THE ORIGINAL TASK 1 SPLIT")

# Task 1 split by row position while rows were still sorted by Category
orig = monthly.copy()
cut = int(len(orig) * 0.8)
train_orig, test_orig = orig.iloc[:cut], orig.iloc[cut:]
overlap = train_orig[train_orig.Period >= test_orig.Period.min()]

report(f"Rows: {len(orig)} | train {len(train_orig)} | test {len(test_orig)}")
report(f"Test categories: {sorted(test_orig['Category'].unique())}")
report(f"Test period:  {period_label(test_orig.Period.min())} to {period_label(test_orig.Period.max())}")
report(f"Train period: {period_label(train_orig.Period.min())} to {period_label(train_orig.Period.max())}")
report(f"Training rows dated inside the test window: {len(overlap)} "
       f"({len(overlap) / len(train_orig):.0%} of training data)")

pred = random_forest().fit(train_orig[FEATURES], train_orig["Sales"]).predict(test_orig[FEATURES])
report(f"Original-split metrics: MAPE {mape(test_orig.Sales, pred):.1f}% | "
       f"MAE {mean_absolute_error(test_orig.Sales, pred):,.0f} | "
       f"R2 {r2_score(test_orig.Sales, pred):.3f}")

# -----------------------------------------------------------------------------
# Q1b. Corrected evaluation
# -----------------------------------------------------------------------------
section("Q1b. CORRECTED EVALUATION")

monthly = monthly.sort_values(["Period", "Category"]).reset_index(drop=True)
periods = np.sort(monthly["Period"].unique())

# Date-based hold-out: last 20% of months, all categories
test_start = periods[int(len(periods) * 0.8)]
train = monthly[monthly.Period < test_start]
test = monthly[monthly.Period >= test_start]
model = random_forest().fit(train[FEATURES], train["Sales"])
test = test.assign(pred_rf=model.predict(test[FEATURES]))

report(f"Hold-out: train {period_label(train.Period.min())}..{period_label(train.Period.max())} "
       f"({len(train)} rows), test {period_label(test.Period.min())}..{period_label(test.Period.max())} "
       f"({len(test)} rows, all categories)")

holdout_rows = []
for name, col in [("Random Forest", "pred_rf"),
                  ("Naive (last month)", "Sales_Lag1"),
                  ("Seasonal naive (same month last year)", "Sales_Lag12")]:
    holdout_rows.append({"model": name,
                         "MAE": mean_absolute_error(test.Sales, test[col]),
                         "MAPE_%": mape(test.Sales, test[col]),
                         "R2": r2_score(test.Sales, test[col])})
report(pd.DataFrame(holdout_rows).round(3).to_string(index=False))

# Rolling-origin CV: expanding window, 4 folds of 6 months each
N_FOLDS, HORIZON = 4, 6
cv_rows = []
for k in range(N_FOLDS):
    t0 = periods[len(periods) - (N_FOLDS - k) * HORIZON]
    t1 = t0 + HORIZON
    fold_train = monthly[monthly.Period < t0]
    fold_test = monthly[(monthly.Period >= t0) & (monthly.Period < t1)]
    fold_pred = random_forest().fit(fold_train[FEATURES], fold_train["Sales"]).predict(fold_test[FEATURES])
    cv_rows.append({"fold": k + 1,
                    "train_rows": len(fold_train),
                    "test_window": f"{period_label(t0)}..{period_label(t1 - 1)}",
                    "RF_MAPE_%": mape(fold_test.Sales, fold_pred),
                    "SeasonalNaive_MAPE_%": mape(fold_test.Sales, fold_test.Sales_Lag12)})
cv = pd.DataFrame(cv_rows).round(1)
report("\nRolling-origin CV (expanding window):")
report(cv.to_string(index=False))
report(f"RF MAPE mean +/- sd: {cv['RF_MAPE_%'].mean():.1f} +/- {cv['RF_MAPE_%'].std():.1f}%")
report(f"Seasonal-naive MAPE mean +/- sd: {cv['SeasonalNaive_MAPE_%'].mean():.1f} "
       f"+/- {cv['SeasonalNaive_MAPE_%'].std():.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
categories = sorted(monthly.Category.unique())
splits = [(train_orig, test_orig, "Original Task 1 split (row order)"),
          (train, test, "Corrected split (by date)")]
for ax, (tr, te, title) in zip(axes, splits):
    for i, cat in enumerate(categories):
        ax.scatter(tr[tr.Category == cat].Period, [i] * (tr.Category == cat).sum(),
                   marker="s", s=14, color="tab:blue", label="train" if i == 0 else None)
        ax.scatter(te[te.Category == cat].Period, [i] * (te.Category == cat).sum(),
                   marker="s", s=14, color="tab:red", label="test" if i == 0 else None)
    ax.set_title(title)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ticks = periods[::6]
    ax.set_xticks(ticks)
    ax.set_xticklabels([period_label(t) for t in ticks], rotation=45, fontsize=8)
axes[0].set_ylim(-0.5, 2.9)
axes[0].legend(loc="upper left", ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_split_diagnosis.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Model 2: data preparation (identical to Task 1, k = 5)
# -----------------------------------------------------------------------------
CLUSTER_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
FINAL_K = 5

mall = pd.read_csv(MALL_PATH)
gender_col = next((c for c in mall.columns if c.lower() in ("gender", "genre", "sex")), None)
mall = mall.dropna(subset=CLUSTER_FEATURES).reset_index(drop=True)
X_scaled = StandardScaler().fit_transform(mall[CLUSTER_FEATURES])
mall["Cluster"] = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=10).fit(X_scaled).labels_

# -----------------------------------------------------------------------------
# Q1c. Clustering stability (bootstrap + Adjusted Rand Index)
# -----------------------------------------------------------------------------
section("Q1c. MODEL 2 STABILITY (bootstrap, Adjusted Rand Index)")
rng = np.random.default_rng(RANDOM_STATE)


def stability(X, k, n_boot=50, frac=1.0):
    """Mean and sd of ARI between the full-data clustering and bootstrap refits."""
    reference = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X).labels_
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=int(len(X) * frac), replace=True)
        boot = KMeans(n_clusters=k, random_state=int(rng.integers(1e6)), n_init=10).fit(X[idx])
        scores.append(adjusted_rand_score(reference, boot.predict(X)))
    return np.mean(scores), np.std(scores)


for k in (2, 3, 4, 5, 6):
    mu, sd = stability(X_scaled, k)
    report(f"k={k}: ARI {mu:.3f} +/- {sd:.3f}")

# -----------------------------------------------------------------------------
# Q2. Learning curves
# -----------------------------------------------------------------------------
section("Q2. LEARNING CURVES")

# Model 1: fixed test set (last 12 months); train on the most recent n months before it
lc_test_start = periods[-12]
lc_test = monthly[monthly.Period >= lc_test_start]
pool = periods[periods < lc_test_start]
sizes, train_err, test_err = [], [], []
for n_months in range(6, len(pool) + 1, 3):
    window = monthly[monthly.Period.isin(pool[-n_months:])]
    tr_scores, te_scores = [], []
    for seed in range(5):
        m = random_forest(seed).fit(window[FEATURES], window.Sales)
        tr_scores.append(mape(window.Sales, m.predict(window[FEATURES])))
        te_scores.append(mape(lc_test.Sales, m.predict(lc_test[FEATURES])))
    sizes.append(len(window))
    train_err.append(tr_scores)
    test_err.append(te_scores)
train_err, test_err = np.array(train_err), np.array(test_err)
seasonal_naive_lc = mape(lc_test.Sales, lc_test.Sales_Lag12)

for s, a, b in zip(sizes, train_err.mean(1), test_err.mean(1)):
    report(f"Model 1  train rows {s:3d}: train MAPE {a:5.1f}% | test MAPE {b:5.1f}%")
report(f"Seasonal-naive test MAPE on same window: {seasonal_naive_lc:.1f}%")

# Model 2: stability and silhouette vs sample size
fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
ari_mean, ari_sd, silhouettes = [], [], []
for f in fractions:
    mu, sd = stability(X_scaled, FINAL_K, n_boot=30, frac=f)
    ari_mean.append(mu)
    ari_sd.append(sd)
    idx = rng.choice(len(X_scaled), size=int(len(X_scaled) * f), replace=False)
    labels = KMeans(FINAL_K, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled[idx])
    silhouettes.append(silhouette_score(X_scaled[idx], labels))
    report(f"Model 2  n={int(len(X_scaled) * f):4d}: ARI {mu:.3f} +/- {sd:.3f} | "
           f"silhouette {silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
for err, name, color in [(train_err, "Training", "tab:blue"),
                         (test_err, "Test (last 12 months)", "tab:red")]:
    ax.plot(sizes, err.mean(1), "o-", color=color, label=name)
    ax.fill_between(sizes, err.mean(1) - err.std(1), err.mean(1) + err.std(1),
                    color=color, alpha=0.15)
ax.axhline(seasonal_naive_lc, ls="--", color="grey", label="Seasonal-naive baseline")
ax.set_xlabel("Training rows (months x 3 categories)")
ax.set_ylabel("MAPE (%), lower is better")
ax.set_title("(a) Model 1: Random Forest sales forecast")
ax.legend()

ax = axes[1]
n_customers = [int(len(X_scaled) * f) for f in fractions]
ax.errorbar(n_customers, ari_mean, yerr=ari_sd, fmt="o-", color="tab:green",
            capsize=3, label="Stability (ARI)")
ax.plot(n_customers, silhouettes, "s--", color="tab:purple", label="Silhouette")
ax.set_xlabel("Number of customers used to fit")
ax.set_ylabel("Score, higher is better")
ax.set_title(f"(b) Model 2: K-Means (k={FINAL_K})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_learning_curves.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Q3. Fairness audit (Fairlearn)
# -----------------------------------------------------------------------------
section("Q3. FAIRNESS AUDIT (Fairlearn)")

# Scenario: a premium promotion targets clusters whose mean spending score
# is above the overall mean.


def targeted(labels, frame):
    cluster_means = frame.groupby(labels)["Spending Score (1-100)"].mean()
    target = cluster_means[cluster_means > frame["Spending Score (1-100)"].mean()].index
    return np.isin(labels, target).astype(int), list(target)


mall["AgeGroup"] = pd.cut(mall.Age, [0, 29, 44, 59, 200],
                          labels=["18-29", "30-44", "45-59", "60+"])
selected, target_clusters = targeted(mall.Cluster.values, mall)
report(f"Targeted clusters (mean spending score above average): {target_clusters}")

y_dummy = np.zeros(len(mall))  # selection_rate only uses y_pred
sensitive = {"AgeGroup": mall.AgeGroup}
if gender_col:
    sensitive["Gender"] = mall[gender_col].astype(str)
else:
    report("No gender column found; only age is audited.")

summary = []
for name, feature in sensitive.items():
    mf = MetricFrame(metrics={"selection_rate": selection_rate, "n": count},
                     y_true=y_dummy, y_pred=selected, sensitive_features=feature)
    dpd = demographic_parity_difference(y_dummy, selected, sensitive_features=feature)
    dpr = demographic_parity_ratio(y_dummy, selected, sensitive_features=feature)
    report(f"\nSelection rate by {name} (Task 1 model, Age included):")
    report(mf.by_group.round(3).to_string())
    report(f"Demographic parity difference: {dpd:.3f} | ratio: {dpr:.3f}")
    summary.append(("With Age", name, dpd, dpr))

report("\nCluster composition by age group (row %):")
report((pd.crosstab(mall.Cluster, mall.AgeGroup, normalize="index") * 100).round(1).to_string())

# Fairness through unawareness: refit without Age
X_no_age = StandardScaler().fit_transform(mall[["Annual Income (k$)", "Spending Score (1-100)"]])
labels_no_age = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=10).fit_predict(X_no_age)
selected_no_age, _ = targeted(labels_no_age, mall)

report("\nK-Means without Age")
report(f"Silhouette without Age: {silhouette_score(X_no_age, labels_no_age):.3f} "
       f"(with Age: {silhouette_score(X_scaled, mall.Cluster):.3f})")
for name, feature in sensitive.items():
    mf = MetricFrame(metrics=selection_rate, y_true=y_dummy, y_pred=selected_no_age,
                     sensitive_features=feature)
    dpd = demographic_parity_difference(y_dummy, selected_no_age, sensitive_features=feature)
    dpr = demographic_parity_ratio(y_dummy, selected_no_age, sensitive_features=feature)
    report(f"Selection rate by {name} (Age removed): {mf.by_group.round(3).to_dict()}")
    report(f"Demographic parity difference: {dpd:.3f} | ratio: {dpr:.3f}")
    summary.append(("Without Age", name, dpd, dpr))

report("\nProxy check: correlation of Age with the remaining features")
report(mall[CLUSTER_FEATURES].corr()["Age"].round(3).to_string())
report("\nSummary:")
report(pd.DataFrame(summary, columns=["model", "attribute", "DP_difference", "DP_ratio"])
         .round(3).to_string(index=False))

# Model 1 has no person-level attribute: report error disparity across categories
mf_sales = MetricFrame(metrics={"MAE": mean_absolute_error, "MAPE_%": mape, "n": count},
                       y_true=test.Sales, y_pred=test.pred_rf, sensitive_features=test.Category)
report("\nModel 1 hold-out error by Category:")
report(mf_sales.by_group.round(1).to_string())
report(f"Largest MAPE gap between categories: {mf_sales.difference()['MAPE_%']:.1f} pp")

rate_with = MetricFrame(metrics=selection_rate, y_true=y_dummy, y_pred=selected,
                        sensitive_features=mall.AgeGroup).by_group
rate_without = MetricFrame(metrics=selection_rate, y_true=y_dummy, y_pred=selected_no_age,
                           sensitive_features=mall.AgeGroup).by_group
x = np.arange(len(rate_with))
plt.figure(figsize=(7, 4))
plt.bar(x - 0.2, rate_with.values, 0.4, label="Task 1 model (Age included)")
plt.bar(x + 0.2, rate_without.values, 0.4, label="Age removed")
plt.xticks(x, rate_with.index)
plt.xlabel("Age group")
plt.ylabel("Share selected for promotion")
plt.ylim(0, 1)
plt.legend()
plt.title("Selection rate by age group")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_fairness_age.png", dpi=200)
plt.close()

_log.close()
print(f"\nDone. Results in {OUT_DIR}/part2_results.txt, figures in {OUT_DIR}/")
