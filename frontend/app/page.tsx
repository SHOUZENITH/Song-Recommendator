'use client';

import { useState, useEffect, useRef } from 'react';
import { Search, Music, PlayCircle, Loader2 } from 'lucide-react';

// Define types for our data
interface SongResult {
  name: string;
  artist: string;
  image?: string;
  source: string;
}

interface Recommendation {
  name: string;
  artists: string;
  year: number;
  image?: string;
  preview_url?: string;
}

interface ResultData {
  searched_song: {
    name: string;
    artist: string;
    image?: string;
  };
  recommendations: Recommendation[];
}

export default function Home() {
  const [songName, setSongName] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResultData | null>(null);
  
  // New State for Autocomplete
  const [suggestions, setSuggestions] = useState<SongResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const searchTimeout = useRef<NodeJS.Timeout | null>(null);

  // Handle typing with debounce (wait 300ms after stopping typing to search)
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSongName(value);
    setShowSuggestions(true);

    if (searchTimeout.current) clearTimeout(searchTimeout.current);

    if (value.length > 1) {
      searchTimeout.current = setTimeout(async () => {
        try {
          const res = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(value)}`);
          if (res.ok) {
            const data = await res.json();
            setSuggestions(data);
          }
        } catch (err) {
          console.error("Autocomplete error:", err);
        }
      }, 300);
    } else {
      setSuggestions([]);
    }
  };

  const selectSong = (song: SongResult) => {
    setSongName(song.name);
    setSuggestions([]); // Hide suggestions
    setShowSuggestions(false);
    // Optional: Auto-trigger search when selected
    // handleSearch(undefined, song.name); 
  };

  const handleSearch = async (e?: React.FormEvent, overrideName?: string) => {
    if (e) e.preventDefault();
    const query = overrideName || songName;
    if (!query) return;

    setLoading(true);
    setSuggestions([]); // Clear suggestions on search
    setResult(null);

    try {
      const res = await fetch('http://localhost:8000/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song_name: query }),
      });

      if (!res.ok) throw new Error('Failed to fetch');
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert('Error fetching recommendations. Check backend console.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-8 font-sans selection:bg-green-500 selection:text-black">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 text-center">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-green-400 to-emerald-600 text-transparent bg-clip-text">
            Vibe Match
          </h1>
          <p className="text-gray-400 text-lg">Type a song, let AI match the vibe.</p>
        </header>

        {/* Search Section */}
        <div className="relative max-w-xl mx-auto mb-16">
          <form onSubmit={handleSearch} className="relative z-10">
            <input
              type="text"
              value={songName}
              onChange={handleInputChange}
              onFocus={() => songName.length > 1 && setShowSuggestions(true)}
              // Delay blur to allow clicking suggestions
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              placeholder="What are you listening to? (e.g. Blinding Lights)"
              className="w-full bg-neutral-900 border border-neutral-800 rounded-full py-4 px-6 pl-14 text-lg focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500 transition-all placeholder:text-gray-600"
              autoComplete="off"
            />
            <Search className="absolute left-5 top-1/2 transform -translate-y-1/2 text-gray-500" />
            
            <button 
              type="submit"
              disabled={loading}
              className="absolute right-2 top-2 bottom-2 bg-green-500 hover:bg-green-400 text-black font-semibold px-6 rounded-full transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? 'Digging...' : 'Match'}
            </button>
          </form>

          {/* Autocomplete Dropdown */}
          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-2 bg-neutral-900 border border-neutral-800 rounded-2xl shadow-2xl overflow-hidden z-20 animate-in fade-in slide-in-from-top-2">
              <ul>
                {suggestions.map((song, idx) => (
                  <li 
                    key={idx}
                    onClick={() => selectSong(song)}
                    className="flex items-center gap-4 px-6 py-3 hover:bg-neutral-800 cursor-pointer transition-colors border-b border-neutral-800/50 last:border-0"
                  >
                    {song.image ? (
                      <img src={song.image} alt="art" className="w-10 h-10 rounded object-cover" />
                    ) : (
                      <div className="w-10 h-10 bg-neutral-800 rounded flex items-center justify-center">
                        <Music className="w-5 h-5 text-gray-600" />
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-white truncate">{song.name}</p>
                      <p className="text-sm text-gray-400 truncate">{song.artist}</p>
                    </div>
                    {song.source === 'spotify' && (
                      <span className="text-[10px] uppercase tracking-wider text-green-500 font-bold">Spotify</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Results Display */}
        {result && (
          <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
            
            {/* The Song User Searched */}
            <div className="flex flex-col items-center">
              <h2 className="text-xs font-bold tracking-widest text-gray-500 uppercase mb-6">Based on</h2>
              <div className="bg-neutral-900 p-6 rounded-3xl flex items-center gap-6 border border-neutral-800 shadow-2xl">
                {result.searched_song.image ? (
                  <img 
                    src={result.searched_song.image} 
                    alt="Album Art" 
                    className="w-32 h-32 rounded-2xl shadow-lg object-cover" 
                  />
                ) : (
                  <div className="w-32 h-32 bg-neutral-800 rounded-2xl flex items-center justify-center">
                    <Music className="w-12 h-12 text-gray-600" />
                  </div>
                )}
                <div>
                  <h3 className="text-2xl font-bold">{result.searched_song.name}</h3>
                  <p className="text-green-400 text-lg">{result.searched_song.artist}</p>
                </div>
              </div>
            </div>

            {/* Recommendations Grid */}
            <div>
              <h2 className="text-xs font-bold tracking-widest text-gray-500 uppercase mb-6 text-center">We Recommend</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {result.recommendations.map((rec, idx) => (
                  <div key={idx} className="bg-neutral-900 group hover:bg-neutral-800 transition-colors rounded-xl p-4 flex gap-4 items-center border border-neutral-800/50">
                    <div className="relative shrink-0">
                      {rec.image ? (
                        <img 
                          src={rec.image} 
                          alt={rec.name} 
                          className="w-16 h-16 rounded-lg object-cover shadow-md" 
                        />
                      ) : (
                        <div className="w-16 h-16 bg-neutral-800 rounded-lg flex items-center justify-center">
                          <Music className="w-6 h-6 text-gray-600" />
                        </div>
                      )}
                      
                      {rec.preview_url && (
                        <a 
                          href={rec.preview_url} 
                          target="_blank"
                          rel="noreferrer"
                          className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/40 transition-opacity rounded-lg"
                        >
                           <PlayCircle className="w-8 h-8 text-green-400 drop-shadow-lg" />
                        </a>
                      )}
                    </div>
                    
                    <div className="overflow-hidden">
                      <h4 className="font-bold truncate" title={rec.name}>{rec.name}</h4>
                      <p className="text-sm text-gray-400 truncate" title={rec.artists}>{rec.artists}</p>
                      {rec.year > 0 && <span className="text-xs text-neutral-600 mt-1 block">{rec.year}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}