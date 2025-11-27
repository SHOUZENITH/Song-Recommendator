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
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("Data/data.csv")

# Clustering pipeline for song data
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])

X_song = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X_song)
song_cluster_labels = song_cluster_pipeline.predict(X_song)
data['cluster_label'] = song_cluster_labels


# PCA pipeline for song embedding
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('PCA', PCA(n_components=2))
])

song_embedding = pca_pipeline.fit_transform(X_song)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

# Save the projection data for visualization
projection.to_csv("song_projection.csv", index=False)

# Plot song clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=projection, x='x', y='y', hue='cluster', palette='tab20', alpha=0.7, s=100)
plt.title('PCA Visualization of Song Clusters')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(title='Cluster')
plt.savefig("song_clusters_visualization.png")
plt.show()
