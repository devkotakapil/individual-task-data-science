"""
Model 2: Customer Segmentation (Clustering)
Dataset: Mall Customer Segmentation Dataset (Kaggle)
Algorithm: K-Means Clustering
Task: Segment customers by Age, Annual Income, Spending Score
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# Loading and cleaning the data
df = pd.read_csv('data/mall_customer/store_customers.csv')
df = df.dropna(subset=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']).reset_index(drop=True)
print(f"Rows after dropping missing values: {len(df)}")

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Scale features (for K-Means / distance-based clustering) 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determining optimal k via elbow method + silhouette score 
inertias = []
sil_scores = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(list(k_range), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(list(k_range), sil_scores, marker='o', color='green')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by k')
plt.tight_layout()
plt.savefig('outputs/model2_k_selection.png', dpi=150)
plt.close()

best_k = k_range[int(np.argmax(sil_scores))]
print(f"\nSilhouette scores by k: {dict(zip(k_range, [round(s,3) for s in sil_scores]))}")
print(f"Best k by silhouette score: {best_k}")

# Fit final model
# NOTE: silhouette score is maximised at k=2, but this yields only a coarse split
# with limited business interpretability. We select k=5, which balances a strong
# silhouette score with actionable, commercially meaningful segments
kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, df['Cluster'])

print(f"\n=== Model 2: Customer Segmentation (K-Means, k={final_k}) ===")
print(f"Silhouette Score: {final_silhouette:.3f}")
print(f"Inertia: {kmeans.inertia_:.2f}")

# Cluster profiles
cluster_summary = df.groupby('Cluster')[features].mean().round(1)
cluster_summary['Count'] = df.groupby('Cluster').size()
print("\nCluster profiles (mean values):")
print(cluster_summary)

# Visualization of  clusters (Income vs Spending Score) 
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df['Annual Income (k$)'], df['Spending Score (1-100)'],
    c=df['Cluster'], cmap='tab10', alpha=0.6
)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'Model 2: Customer Segments (K-Means, k={final_k})')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('outputs/model2_clusters.png', dpi=150)
plt.close()

# Saving metrics/summary to file
with open('outputs/model2_metrics.txt', 'w') as f:
    f.write(f"Model 2: Customer Segmentation (K-Means, k={final_k})\n")
    f.write(f"Silhouette Score: {final_silhouette:.3f}\n")
    f.write(f"Inertia: {kmeans.inertia_:.2f}\n\n")
    f.write("Cluster profiles (mean values):\n")
    f.write(cluster_summary.to_string())

print("\nSaved: outputs/model2_k_selection.png, outputs/model2_clusters.png, outputs/model2_metrics.txt")
