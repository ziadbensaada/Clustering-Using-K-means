import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data (replace 'donnees_clients.csv' with your actual file path)
data = pd.read_csv("donnees_clients.csv")

# Check for missing values
print("Number of missing values before cleaning:")
print(data.isnull().sum())

# Calculate mean for 'revenu' and 'score' columns
mean_revenu = data["revenu"].mean()
mean_score = data["score"].mean()

# Fill missing values with mean values
data["revenu"].fillna(mean_revenu, inplace=True)
data["score"].fillna(mean_score, inplace=True)

# Check for missing values after filling
print("\nNumber of missing values after filling:")
print(data.isnull().sum())

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Check for duplicate rows after removal
print("\nNumber of duplicate rows after removal:")
print(data.duplicated().sum())

# Save the cleaned dataset
data.to_csv('cleaned_donne_clients.csv', index=False)

# Drop non-numeric column
data_numeric = data.drop(columns=['sexe'])

# Normalize the data
scaler = StandardScaler()
data_scale = scaler.fit_transform(data_numeric)

# Define the number of clusters
n_clusters = 3

# Run k-means algorithm
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit(data_scale).labels_

# Display the results
print("Labels des clusters:", labels)
print("Centroides des clusters:", kmeans.cluster_centers_)

# Visualiser les clusters
import matplotlib.pyplot as plt
plt.scatter(data_scale[:, 0], data_scale[:, 1], c=labels, s=50, alpha=0.5)
plt.show()