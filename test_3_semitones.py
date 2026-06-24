import os
import sys
import librosa

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import load_audio, compute_spectrogram, get_constellation, generate_hashes
from src.database import SongDatabase
from src.matcher import match_query

def test_3_semitones():
    dataset_dir = "seed"
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.mp3')]
    
    # Pick the first 33 files
    test_files = files[:33]
    
    db = SongDatabase("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        print("Database not found. Please run build_db.py first!")
        return

    print("==================================================")
    print("      PITCH TEST (3 SEMITONES) ON 33 SONGS        ")
    print("==================================================")
    
    overall_passed = 0

    for i, file in enumerate(test_files):
        song_name = os.path.splitext(file)[0]
        file_path = os.path.join(dataset_dir, file)
        
        try:
            y, sr = load_audio(file_path)
            start_sample = 30 * sr
            end_sample = 35 * sr
            if len(y) < end_sample:
                start_sample = 0
                end_sample = 5 * sr
            
            y_snippet = y[start_sample:end_sample]
            
            if len(y_snippet) == 0:
                continue
            
            # Pitch Shift Test (+3 semitones)
            y_shifted = librosa.effects.pitch_shift(y_snippet, sr=sr, n_steps=3)
            _, _, S_shifted = compute_spectrogram(y_shifted, sr)
            hashes_shifted = generate_hashes(get_constellation(S_shifted))
            match_shifted, _ = match_query(hashes_shifted, db)
            
            pass_pitch = match_shifted == song_name
            print(f"[{i+1}/33] {song_name} -> Predicted: {match_shifted} -> {'PASS ✅' if pass_pitch else 'FAIL ❌'}")
            
            overall_passed += int(pass_pitch)
            
        except Exception as e:
            pass

    print("\n==================================================")
    print(f"FINAL SCORE: {overall_passed} / 33 Tests Passed (3 Semitones)")
    print("==================================================")

if __name__ == "__main__":
    test_3_semitones()
