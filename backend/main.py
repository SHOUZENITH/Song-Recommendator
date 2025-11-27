import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Spotify Setup ---
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

sp = None
if SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET
        ))
        print("✅ Spotify Connected")
    except Exception as e:
        print(f"⚠️ Spotify Connection Failed: {e}")

# --- Load Model & Data ---
model = None
data = None

@app.on_event("startup")
def load_assets():
    global model, data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Model
    model_path = os.path.join(base_dir, "song_cluster_pipeline_20.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    # 2. Load Data
    data_path = os.path.join(base_dir, "Data", "data.csv")
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            # Create a lowercase name column for faster searching
            data['name_lower'] = data['name'].str.lower().astype(str)
            print(f"✅ Data loaded successfully ({len(data)} rows)")
            
            # Pre-compute clusters if needed
            if model and 'cluster_label' not in data.columns:
                print("⏳ Pre-computing clusters...")
                cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
                        'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
                        'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
                valid_data = data.dropna(subset=cols)
                data.loc[valid_data.index, 'cluster_label'] = model.predict(valid_data[cols])
                print("✅ Clusters computed.")

        except Exception as e:
            print(f"❌ Failed to load data: {e}")

class SongRequest(BaseModel):
    song_name: str
    artist_name: str = ""
    
@app.get("/search")
def search_songs(query: str):
    """
    Autocomplete endpoint: Returns top 5 matching songs as user types.
    """
    if not query:
        return []

    results = []
    query_lower = query.lower().strip()

    # 1. Search Local Data (Fastest & Preferred)
    if data is not None:
        # Check for songs containing the query string
        matches = data[data['name_lower'].str.contains(query_lower, na=False)]
        
        # Sort by popularity if available to show "Blinding Lights" before obscure covers
        if 'popularity' in matches.columns:
            matches = matches.sort_values(by='popularity', ascending=False)
        
        # Take top 5
        top_matches = matches.head(5)
        
        for _, row in top_matches.iterrows():
            results.append({
                "name": row['name'],
                "artist": row['artists'], # Note: CSV column is 'artists'
                "source": "local"
            })

    # 2. If Local gave < 5 results, fill the rest with Spotify (Better for new songs)
    if len(results) < 5 and sp:
        try:
            sp_results = sp.search(q=query, limit=5 - len(results), type='track')
            if sp_results and sp_results.get('tracks'):
                for t in sp_results['tracks']['items']:
                    # Avoid duplicates
                    if not any(r['name'].lower() == t['name'].lower() for r in results):
                        results.append({
                            "name": t['name'],
                            "artist": t['artists'][0]['name'],
                            "image": t['album']['images'][-1]['url'] if t['album']['images'] else None,
                            "source": "spotify"
                        })
        except Exception:
            pass # Ignore Spotify errors for autocomplete to keep it fast

    return results

@app.post("/recommend")
async def recommend(request: SongRequest):
    if model is None or data is None:
        raise HTTPException(status_code=503, detail="Server initializing or Data/Model missing")

    search_query = request.song_name.lower().strip()
    
    # --- STRATEGY 1: Local Search (Primary) ---
    # This avoids Spotify API limits and crashes
    matches = data[data['name_lower'] == search_query]
    
    if matches.empty:
        matches = data[data['name_lower'].str.contains(search_query, regex=False)]

    if not matches.empty:
        # Sort by popularity to get the most famous version
        if 'popularity' in matches.columns:
            matches = matches.sort_values(by='popularity', ascending=False)
        
        song_row = matches.iloc[0]
        
        # Safe extraction from CSV
        input_features = {
            'valence': song_row.get('valence'),
            'year': song_row.get('year'),
            'acousticness': song_row.get('acousticness'),
            'danceability': song_row.get('danceability'),
            'duration_ms': song_row.get('duration_ms'),
            'energy': song_row.get('energy'),
            'explicit': song_row.get('explicit'),
            'instrumentalness': song_row.get('instrumentalness'),
            'key': song_row.get('key'),
            'liveness': song_row.get('liveness'),
            'loudness': song_row.get('loudness'),
            'mode': song_row.get('mode'),
            'popularity': song_row.get('popularity'),
            'speechiness': song_row.get('speechiness'),
            'tempo': song_row.get('tempo')
        }
        
        track_name = song_row['name']
        track_artist = song_row['artists']
        
        # Try to get image from Spotify, but act safely
        track_image = None
        if sp:
            try:
                sp_results = sp.search(q=f"track:{track_name} artist:{track_artist}", limit=1)
                if sp_results and sp_results.get('tracks') and sp_results['tracks'].get('items'):
                    item = sp_results['tracks']['items'][0]
                    # Safe image access
                    images = item.get('album', {}).get('images', [])
                    if images:
                        track_image = images[0].get('url')
            except:
                pass

    else:
        # --- STRATEGY 2: Spotify API (Fallback) ---
        if not sp:
            raise HTTPException(status_code=404, detail="Song not found in local database.")
            
        try:
            results = sp.search(q=f"track:{request.song_name}", type='track', limit=1)
            if not results or not results.get('tracks') or not results['tracks'].get('items'):
                raise HTTPException(status_code=404, detail="Song not found on Spotify")
            
            track = results['tracks']['items'][0]
            if not track:
                raise HTTPException(status_code=404, detail="Invalid Spotify data")

            track_id = track.get('id')
            track_name = track.get('name')
            
            # Safe Artist Access
            artists_list = track.get('artists', [])
            track_artist = artists_list[0].get('name') if artists_list else "Unknown"
            
            # Safe Image Access
            album = track.get('album', {})
            images = album.get('images', [])
            track_image = images[0].get('url') if images else None
            
            # Audio Features
            af_list = sp.audio_features([track_id])
            if not af_list or af_list[0] is None:
                raise HTTPException(status_code=404, detail="Audio features unavailable for this song")
            
            af = af_list[0]
            
            # Safe Date Parsing
            release_date = album.get('release_date', '2020')
            release_year = int(release_date[:4]) if release_date else 2020

            # Safe Feature Access
            input_features = {
                'valence': af.get('valence', 0.5), 
                'year': release_year,
                'acousticness': af.get('acousticness', 0.5), 
                'danceability': af.get('danceability', 0.5),
                'duration_ms': af.get('duration_ms', 200000), 
                'energy': af.get('energy', 0.5),
                'explicit': 1 if track.get('explicit') else 0, 
                'instrumentalness': af.get('instrumentalness', 0.0),
                'key': af.get('key', 5), 
                'liveness': af.get('liveness', 0.1), 
                'loudness': af.get('loudness', -5.0),
                'mode': af.get('mode', 1), 
                'popularity': track.get('popularity', 50),
                'speechiness': af.get('speechiness', 0.05), 
                'tempo': af.get('tempo', 120.0)
            }
        except Exception as e:
            print(f"Spotify Fallback Error: {e}")
            raise HTTPException(status_code=404, detail="Song could not be processed via Spotify")

    # --- PREDICTION ---
    cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
            'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
            'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    
    # Create DataFrame safely
    input_df = pd.DataFrame([input_features], columns=cols)
    
    try:
        predicted_cluster = model.predict(input_df)[0]
    except Exception as e:
        print(f"Model Error: {e}")
        raise HTTPException(status_code=500, detail="Model prediction error")

    # --- RECOMMENDATION ---
    if 'cluster_label' not in data.columns:
        # Emergency calculation
        try:
            valid_data = data.dropna(subset=cols)
            data.loc[valid_data.index, 'cluster_label'] = model.predict(valid_data[cols])
        except:
             raise HTTPException(status_code=500, detail="Could not cluster local data")

    cluster_songs = data[data['cluster_label'] == predicted_cluster]
    
    # Filter out input song
    cluster_songs = cluster_songs[cluster_songs['name_lower'] != str(track_name).lower()]
    
    if len(cluster_songs) > 5:
        recs = cluster_songs.sample(5)
    else:
        recs = cluster_songs

    # --- FETCH IMAGES FOR RECS ---
    final_recommendations = []
    for _, row in recs.iterrows():
        r_name = row['name']
        r_artist = row['artists']
        r_image = None
        r_preview = None
        
        if sp:
            try:
                # Safe Search
                s_res = sp.search(q=f"track:{r_name} artist:{r_artist}", type='track', limit=1)
                if s_res and s_res.get('tracks') and s_res['tracks'].get('items'):
                    t = s_res['tracks']['items'][0]
                    
                    # Safe Image/Preview Access
                    t_album = t.get('album', {})
                    t_images = t_album.get('images', [])
                    if t_images:
                        r_image = t_images[0].get('url')
                    
                    r_preview = t.get('preview_url')
            except:
                pass
        
        final_recommendations.append({
            "name": str(r_name),
            "artists": str(r_artist),
            "year": int(row['year']),
            "image": r_image,
            "preview_url": r_preview
        })

    return {
        "searched_song": {
            "name": track_name,
            "artist": track_artist,
            "image": track_image
        },
        "recommendations": final_recommendations
    }