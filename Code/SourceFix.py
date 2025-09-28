# model_training.py

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
genre_data = pd.read_csv('Data/data_by_genres.csv')
year_data = pd.read_csv('Data/data_by_year.csv')

# Clustering pipeline for genre data
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=10))
])

X_genre = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X_genre)
genre_data['cluster'] = cluster_pipeline.predict(X_genre)

# Save the genre clustering model
joblib.dump(cluster_pipeline, 'genre_cluster_pipeline.pkl')

# TSNE pipeline for genre embedding
tsne_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('tsne', TSNE(n_components=2, verbose=1))
])

genre_embedding = tsne_pipeline.fit_transform(X_genre)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

# Save the projection data for visualization
projection.to_csv("genre_projection.csv", index=False)

# Clustering pipeline for song data
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
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

# Save the projection data for visualization
projection.to_csv("song_projection.csv", index=False)
