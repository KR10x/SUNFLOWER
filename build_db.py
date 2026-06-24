import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database import populate_db

if __name__ == "__main__":
    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed")
    if not os.path.exists(audio_dir):
        print(f"Dataset directory not found at: {audio_dir}")
        print("Please ensure the 'seed' folder is in the sunflower directory.")
        sys.exit(1)
        
    db_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "song_db.pkl")
    print("Building database... this may take a few minutes.")
    populate_db(audio_dir, db_loc)
