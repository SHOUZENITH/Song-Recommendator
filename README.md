# 🎵 Vibe Match – AI Song Recommender

A full-stack web application that recommends songs based on their **“vibe”** using audio features.

It uses a **FastAPI (Python)** backend powered by a **Scikit-Learn K-Means Clustering ML model**, along with a modern **Next.js** frontend.

---

## ✨ Features

### 🔮 Hybrid Recommendation Engine
- **Local AI Model**  
  Uses a trained **K-Means model** to cluster songs (from `data.csv`) based on:
  - valence  
  - acousticness  
  - danceability  
  - energy

- **Spotify Fallback**  
  If a song isn’t found in the local dataset:
  - The system fetches it from **Spotify**
  - If Spotify restricts audio analysis, it uses **Spotify’s native recommendation engine**

### ⚡ Smart Search
- Autocomplete search bar  
- Prioritizes local data for speed and accuracy

### 🎨 Modern UI / UX
- Fetches album art and preview audio  
- Clean **dark-mode interface** using **Tailwind CSS**  
- Built with **Next.js 14**

---

## 🛠️ Tech Stack

**Frontend**
- Next.js 14  
- React  
- Tailwind CSS  
- Lucide Icons  

**Backend**
- Python  
- FastAPI  
- Uvicorn  
- Pandas  
- Scikit-Learn  
- Joblib  
- Spotipy  

**Data**
- Spotify Dataset (~170k songs)

---

## 🚀 Getting Started

### **Prerequisites**
- Node.js (v18+)  
- Python (v3.9+)  
- Spotify Developer Account  

---

## 1. Backend Setup

Navigate to the backend folder:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up Spotify Keys:  
Create `.env` inside `backend`:

```env
SPOTIPY_CLIENT_ID="your_spotify_client_id"
SPOTIPY_CLIENT_SECRET="your_spotify_client_secret"
```

Ensure data files are present:
```
backend/song_cluster_pipeline_20.pkl
backend/Data/data.csv
```

Run server:

```bash
uvicorn main:app --reload
```

Server runs at:  
**http://127.0.0.1:8000**

---

## 2. Frontend Setup

Navigate to frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run development server:

```bash
npm run dev
```

App will be live at:  
**http://localhost:3000**

---

## 🧠 How It Works

1. User types a song → `/search` loads autocomplete suggestions.  
2. Backend checks **local CSV first**, then **Spotify**.  
3. After a song is selected:
   - Audio features are analyzed  
   - K-Means predicts the **vibe cluster**  
   - 5 random songs from the same cluster are returned  

---

## 📂 Project Structure

```
/
├── backend/
│   ├── Data/
│   │   └── data.csv
│   ├── main.py
│   ├── song_cluster_pipeline_20.pkl
│   └── requirements.txt
│   
│
├── frontend/
│   ├── app/
│   │   └── page.tsx
│   └── ...
│
└── training part/
    ├── Code/             # Training scripts (Jupyter/py)
    ├── Data/             # Raw datasets for training
    ├── Model/            # Exported models (.pkl)
    ├── Visualization/    # Charts, graphs, EDA outputs
```

---

## 📜 License

This project is open-source.  
Feel free to use, modify, and improve it!
