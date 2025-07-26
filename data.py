import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 1: Load the dataset
df = pd.read_csv('shopping_behavior_updated.csv')

# Step 2: Preview the dataset
print("First few rows of the dataset:")
print(df.head())

# Step 3: Select only numerical columns for clustering
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Optional: Handle missing values
df_numeric = df_numeric.dropna()

# Step 4: Standardize the numerical data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# Step 5: Elbow Method to find optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, 11), y=inertia, marker="o")
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()

# Step 6: Choose optimal k (e.g., 3) based on elbow plot
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Step 7: Add cluster labels to the original DataFrame
df['Cluster'] = clusters

# Step 8: Visualize the clusters using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=60)
plt.title("Customer Segmentation via PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("cluster_visualization.png")
plt.show()

# Step 9: Save the clustered data
df.to_csv('clustered_shopping_behavior.csv', index=False)
print("✅ Clustering complete. File saved as 'clustered_shopping_behavior.csv'")
