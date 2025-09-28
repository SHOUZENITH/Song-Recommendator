import os
import numpy as np
import pandas as pd
import joblib
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
    ('kmeans', KMeans(n_clusters=30, verbose=False,n_init='auto', random_state=42))
])

X_song = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X_song)
song_cluster_labels = song_cluster_pipeline.predict(X_song)
data['cluster_label'] = song_cluster_labels

# Save the song clustering model
joblib.dump(song_cluster_pipeline, 'song_cluster_pipeline.pkl')

# PCA pipeline for song embedding
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('PCA', PCA(n_components=2))
])

song_embedding = pca_pipeline.fit_transform(X_song)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

