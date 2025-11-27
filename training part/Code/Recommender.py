import os
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from scipy.spatial.distance import cdist
import joblib

# Load the song data
data = pd.read_csv("Data/data.csv")

# Spotify authentication setup
os.environ["SPOTIPY_CLIENT_ID"] = "b2378468d44943e2a6f84618e02c3685"
os.environ["SPOTIPY_CLIENT_SECRET"] = "82fa7d3b79804800870df0af77e154c0"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIPY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIPY_CLIENT_SECRET"]))

# Load the AI model for clustering
song_cluster_pipeline = joblib.load('song_cluster_pipeline.pkl')

# Define the columns to be used in the clustering
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Function to find a song on Spotify and return its features
def find_song(name, year):
    song_data = defaultdict()
    try:
        # Search for the song
        results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = sp.audio_features(track_id)[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        # Add the audio features to the song data
        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)
    
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error with track '{name}' from year '{year}': {e}")
        return None

# Function to retrieve song data, either from the database or by querying Spotify
def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                 & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

# Function to calculate the mean vector of a list of songs
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    if song_vectors:
        song_matrix = np.array(song_vectors)
        return np.mean(song_matrix, axis=0)
    else:
        return np.array([])

# Function to flatten a list of dictionaries into a single dictionary
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

# Function to recommend songs based on a list of songs
def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    # Get the mean vector of the input songs
    song_center = get_mean_vector(song_list, spotify_data)
    if song_center.size == 0:
        print("Error: No valid songs to process")
        return []

    # Use the song clustering pipeline to scale the song data
    scaler = song_cluster_pipeline.steps[0][1]  # Extract the scaler from the pipeline
    scaled_data = scaler.transform(spotify_data[number_cols])
    
    # Scale the song center and calculate the cosine distance
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    
    # Get the indices of the closest songs
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    # Filter out the input songs from the recommendations
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    return rec_songs[metadata_cols].to_dict(orient='records')

# Example usage of the recommend_songs function
recommended_songs = recommend_songs([{'name': 'Come As You Are', 'year': 1991},
                                     {'name': 'Smells Like Teen Spirit', 'year': 1991},
                                     {'name': 'Lithium', 'year': 1992},
                                     {'name': 'All Apologies', 'year': 1993},
                                     {'name': 'Stay Away', 'year': 1993}], data)

# Print the recommended songs
for song in recommended_songs:
    print(song)
