import os
import sys
import numpy as np
import librosa

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import load_audio, compute_spectrogram, get_constellation, generate_hashes
from src.database import SongDatabase
from src.matcher import match_query

def test_on_10_songs():
    dataset_dir = "seed"
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.mp3')]
    
    # Pick the first 33 files
    test_files = files[:33]
    
    db = SongDatabase("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        print("Database not found. Please run build_db.py first!")
        return

    print("==================================================")
    print("      ROBUSTNESS TESTING ON 33 DIFFERENT SONGS     ")
    print("==================================================")
    
    overall_passed = 0
    total_tests = 0

    for i, file in enumerate(test_files):
        song_name = os.path.splitext(file)[0]
        file_path = os.path.join(dataset_dir, file)
        
        print(f"\n[{i+1}/10] Testing: {song_name}")
        
        try:
            # Load the song and extract a 5 second snippet (from 30s to 35s)
            # If the song is shorter, just take what we can
            y, sr = load_audio(file_path)
            start_sample = 30 * sr
            end_sample = 35 * sr
            if len(y) < end_sample:
                start_sample = 0
                end_sample = 5 * sr
            
            y_snippet = y[start_sample:end_sample]
            
            if len(y_snippet) == 0:
                print("  -> ERROR: Audio too short or empty.")
                continue

            # 1. Original Test
            _, _, S_log = compute_spectrogram(y_snippet, sr)
            hashes = generate_hashes(get_constellation(S_log))
            match_orig, _ = match_query(hashes, db)
            
            pass_orig = match_orig == song_name
            print(f"  [Original]    Predicted: {match_orig} -> {'PASS ✅' if pass_orig else 'FAIL ❌'}")
            
            # 2. Noise Test
            noise = np.random.randn(len(y_snippet)) * 0.05
            y_noisy = y_snippet + noise
            _, _, S_noisy = compute_spectrogram(y_noisy, sr)
            hashes_noisy = generate_hashes(get_constellation(S_noisy))
            match_noisy, _ = match_query(hashes_noisy, db)
            
            pass_noise = match_noisy == song_name
            print(f"  [Noise Added] Predicted: {match_noisy} -> {'PASS ✅' if pass_noise else 'FAIL ❌'}")
            
            # 3. Pitch Shift Test (+1 semitone)
            y_shifted = librosa.effects.pitch_shift(y_snippet, sr=sr, n_steps=1)
            _, _, S_shifted = compute_spectrogram(y_shifted, sr)
            hashes_shifted = generate_hashes(get_constellation(S_shifted))
            match_shifted, _ = match_query(hashes_shifted, db)
            
            pass_pitch = match_shifted == song_name
            print(f"  [Pitch Shift] Predicted: {match_shifted} -> {'PASS ✅' if pass_pitch else 'FAIL ❌'}")
            
            # 4. Noise + Pitch Shift Test
            y_noisy_shifted = librosa.effects.pitch_shift(y_noisy, sr=sr, n_steps=1)
            _, _, S_noisy_shifted = compute_spectrogram(y_noisy_shifted, sr)
            hashes_noisy_shifted = generate_hashes(get_constellation(S_noisy_shifted))
            match_noisy_shifted, _ = match_query(hashes_noisy_shifted, db)
            
            pass_both = match_noisy_shifted == song_name
            print(f"  [Noise+Pitch] Predicted: {match_noisy_shifted} -> {'PASS ✅' if pass_both else 'FAIL ❌'}")
            
            total_tests += 4
            overall_passed += int(pass_orig) + int(pass_noise) + int(pass_pitch) + int(pass_both)
            
        except Exception as e:
            print(f"  -> ERROR during processing: {e}")

    print("\n==================================================")
    print(f"FINAL SCORE: {overall_passed} / {total_tests} Tests Passed")
    print("==================================================")

if __name__ == "__main__":
    test_on_10_songs()
