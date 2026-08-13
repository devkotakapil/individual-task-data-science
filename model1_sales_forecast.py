"""
Model 1: Sales Forecasting (Regression)
Dataset: Superstore Sales Dataset (Kaggle)
Algorithm: Random Forest Regressor
Task: Predict monthly sales per Category/Region for the next period
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

RANDOM_STATE = 42

#  Loading & cleaning the data
df = pd.read_csv('data/superstore/train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Aggregating to monthly sales per Category (coarser, less noisy than Category x Region)
monthly = (
    df.groupby(['Year', 'Month', 'Category'])['Sales']
    .sum()
    .reset_index()
    .sort_values(['Category', 'Year', 'Month'])
)

# Feature engineering
# Lag features + rolling mean, per Category series
monthly['Sales_Lag1'] = monthly.groupby('Category')['Sales'].shift(1)
monthly['Sales_Lag2'] = monthly.groupby('Category')['Sales'].shift(2)
monthly['Sales_RollMean3'] = (
    monthly.groupby('Category')['Sales']
    .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
)
monthly['Quarter'] = ((monthly['Month'] - 1) // 3) + 1
monthly = monthly.dropna(subset=['Sales_Lag1', 'Sales_Lag2']).reset_index(drop=True)

le_cat = LabelEncoder()
monthly['Category_enc'] = le_cat.fit_transform(monthly['Category'])

features = ['Year', 'Month', 'Quarter', 'Category_enc', 'Sales_Lag1', 'Sales_Lag2', 'Sales_RollMean3']
X = monthly[features]
y = monthly['Sales']

# Train/test split (time-respecting: last 20% chronologically)
monthly = monthly.sort_values(['Year', 'Month'])
split_idx = int(len(monthly) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Training model
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluating
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
r2 = r2_score(y_test, preds)

print("=== Model 1: Sales Forecasting (Random Forest Regressor) ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R2:   {r2:.3f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# Saving plot: actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Monthly Sales')
plt.ylabel('Predicted Monthly Sales')
plt.title('Model 1: Actual vs Predicted Sales (Random Forest Regressor)')
plt.tight_layout()
plt.savefig('outputs/model1_actual_vs_predicted.png', dpi=150)
plt.close()

# Saving metrics to file for report/appendix 
with open('outputs/model1_metrics.txt', 'w') as f:
    f.write("Model 1: Sales Forecasting (Random Forest Regressor)\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
    f.write(f"R2: {r2:.3f}\n\n")
    f.write("Feature importances:\n")
    f.write(importances.to_string())

print("\nSaved: outputs/model1_actual_vs_predicted.png, outputs/model1_metrics.txt")
