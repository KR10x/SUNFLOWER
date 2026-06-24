import pickle
import os
from collections import defaultdict
from .audio_processor import fetch_sound, get_spectro, extract_peaks, create_fingerprints

class TrackDB:
    def __init__(self, db_loc="song_db.pkl"):
        self.db_loc = db_loc
        self.stored_hashes = defaultdict(list)
        self.track_names = set()
        
        self.retrieve_db()

    def insert_track(self, title, hash_array):
        if title in self.track_names:
            return
        
        self.track_names.add(title)
        for h_val, t_val in hash_array:
            self.stored_hashes[h_val].append((title, t_val))
            
    def write_db(self):
        with open(self.db_loc, 'wb') as f_obj:
            pickle.dump({'hashes': self.stored_hashes, 'songs': self.track_names}, f_obj)
            
    def retrieve_db(self):
        if os.path.exists(self.db_loc):
            with open(self.db_loc, 'rb') as f_obj:
                data_in = pickle.load(f_obj)
                self.stored_hashes = data_in['hashes']
                self.track_names = data_in['songs']

def populate_db(audio_dir, db_loc="song_db.pkl"):
    main_db = TrackDB(db_loc)
    
    mp3s = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
    
    for idx, f_name in enumerate(mp3s):
        title = os.path.splitext(f_name)[0]
        if title in main_db.track_names:
            print(f"Skipping {title}, already indexed.")
            continue
            
        print(f"Indexing {idx+1}/{len(mp3s)}: {title}...")
        full_path = os.path.join(audio_dir, f_name)
        
        try:
            sig, rate = fetch_sound(full_path)
            _, _, log_mat = get_spectro(sig, rate)
            peaks = extract_peaks(log_mat)
            h_list = create_fingerprints(peaks)
            main_db.insert_track(title, h_list)
        except Exception as err:
            print(f"Error indexing {title}: {err}")
            
    main_db.write_db()
    print("Indexing complete!")
    return main_db
