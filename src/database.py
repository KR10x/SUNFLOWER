import pickle
import os
from collections import defaultdict
from .audio_processor import load_audio, compute_spectrogram, get_constellation, generate_hashes

class SongDatabase:
    def __init__(self, database_path="song_db.pkl"):
        self.database_path = database_path
        self.hashes = defaultdict(list)
        self.songs = set()
        
        self.load_db()

    def add_song(self, song_name, hashes_list):
        if song_name in self.songs:
            return
        
        self.songs.add(song_name)
        for hash_val, time_val in hashes_list:
            self.hashes[hash_val].append((song_name, time_val))
            
    def save_db(self):
        with open(self.database_path, 'wb') as file_obj:
            pickle.dump({'hashes': self.hashes, 'songs': self.songs}, file_obj)
            
    def load_db(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as file_obj:
                loaded_data = pickle.load(file_obj)
                self.hashes = loaded_data['hashes']
                self.songs = loaded_data['songs']

def index_dataset(dataset_dir, database_path="song_db.pkl"):
    db = SongDatabase(database_path)
    
    files = [filename for filename in os.listdir(dataset_dir) if filename.endswith('.mp3')]
    
    for i, filename in enumerate(files):
        song_name = os.path.splitext(filename)[0]
        if song_name in db.songs:
            print(f"Skipping {song_name}, already indexed.")
            continue
            
        print(f"Indexing {i+1}/{len(files)}: {song_name}...")
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            audio_signal, sample_rate = load_audio(file_path)
            _, _, log_spectrogram = compute_spectrogram(audio_signal, sample_rate)
            constellation = get_constellation(log_spectrogram)
            hashes = generate_hashes(constellation)
            db.add_song(song_name, hashes)
        except Exception as e:
            print(f"Error indexing {song_name}: {e}")
            
    db.save_db()
    print("Indexing complete!")
    return db
