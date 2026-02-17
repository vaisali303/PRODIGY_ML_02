# Customer Segmentation 

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 2. Load dataset
data = pd.read_csv("Mall_Customers.csv")


# 3. Select features for clustering
# Using Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]


# 4. Feature scaling (VERY IMPORTANT for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


# 6. Add cluster labels to dataset
data['Cluster'] = clusters


# 7. Visualize the clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)

plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using K-Means")
plt.show()


# 8. Print first few rows
print(data.head())
