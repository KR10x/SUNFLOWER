import os
import sys

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database import index_dataset

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed")
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found at: {dataset_dir}")
        print("Please ensure the 'seed' folder is in the sunflower directory.")
        sys.exit(1)
        
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "song_db.pkl")
    print("Building database... this may take a few minutes.")
    index_dataset(dataset_dir, db_path)
