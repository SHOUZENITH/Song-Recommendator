import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("Data/data.csv")

# Clustering pipeline for song data
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=21, verbose=False, n_init='auto', random_state=42))
])

# Selecting numerical features
X_song = data.select_dtypes(np.number)

# Fit and predict clusters
song_cluster_pipeline.fit(X_song)
song_cluster_labels = song_cluster_pipeline.predict(X_song)
data['cluster_label'] = song_cluster_labels

# Save the clustering model
joblib.dump(song_cluster_pipeline, 'song_cluster_pipeline.pkl')

# PCA pipeline for song embedding
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('PCA', PCA(n_components=2))
])

# Create PCA projection
song_embedding = pca_pipeline.fit_transform(X_song)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

# Save the projection data for visualization
projection.to_csv("song_projection.csv", index=False)

# Plot and save song clusters visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(data=projection, x='x', y='y', hue='cluster', palette='tab20', alpha=0.7, s=100)
plt.title('PCA Visualization of Song Clusters')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("song_clusters_visualization.png")
plt.show()
