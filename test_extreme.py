import os
import sys
import numpy as np
import librosa

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import load_audio, compute_spectrogram, get_constellation, generate_hashes
from src.database import SongDatabase
from src.matcher import match_query

def run_extreme_tests():
    dataset_dir = "seed"
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.mp3')]
    
    # Evaluate all files in dataset
    test_files = files
    
    db = SongDatabase("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        print("Database not found. Please run build_db.py first!")
        return

    print("==================================================")
    print(f"    EXTREME ROBUSTNESS TESTING ON {len(test_files)} SONGS")
    print("==================================================")
    
    results = {
        "Original": 0,
        "Noise": 0,
        "Pitch (+1)": 0,
        "Pitch (+3)": 0,
        "Noise + Pitch (+1)": 0
    }
    
    total_songs = len(test_files)
    valid_songs = 0

    for i, file in enumerate(test_files):
        song_name = os.path.splitext(file)[0]
        file_path = os.path.join(dataset_dir, file)
        
        print(f"\n[{i+1}/{total_songs}] Testing: {song_name}")
        
        try:
            # Extract a 5 second snippet
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
                
            valid_songs += 1

            # 1. Original Test
            _, _, S_log = compute_spectrogram(y_snippet, sr)
            match_orig, _ = match_query(generate_hashes(get_constellation(S_log)), db)
            pass_orig = match_orig == song_name
            results["Original"] += int(pass_orig)
            print(f"  [Original]         Predicted: {match_orig} -> {'PASS ✅' if pass_orig else 'FAIL ❌'}")
            
            # 2. Noise Test
            noise = np.random.randn(len(y_snippet)) * 0.05
            y_noisy = y_snippet + noise
            _, _, S_noisy = compute_spectrogram(y_noisy, sr)
            match_noisy, _ = match_query(generate_hashes(get_constellation(S_noisy)), db)
            pass_noise = match_noisy == song_name
            results["Noise"] += int(pass_noise)
            print(f"  [Noise]            Predicted: {match_noisy} -> {'PASS ✅' if pass_noise else 'FAIL ❌'}")
            
            # 3. Pitch Shift (+1)
            y_shifted_1 = librosa.effects.pitch_shift(y_snippet, sr=sr, n_steps=1)
            _, _, S_shifted_1 = compute_spectrogram(y_shifted_1, sr)
            match_shifted_1, _ = match_query(generate_hashes(get_constellation(S_shifted_1)), db)
            pass_pitch_1 = match_shifted_1 == song_name
            results["Pitch (+1)"] += int(pass_pitch_1)
            print(f"  [Pitch (+1)]       Predicted: {match_shifted_1} -> {'PASS ✅' if pass_pitch_1 else 'FAIL ❌'}")

            # 4. Pitch Shift (+3)
            y_shifted_3 = librosa.effects.pitch_shift(y_snippet, sr=sr, n_steps=3)
            _, _, S_shifted_3 = compute_spectrogram(y_shifted_3, sr)
            match_shifted_3, _ = match_query(generate_hashes(get_constellation(S_shifted_3)), db)
            pass_pitch_3 = match_shifted_3 == song_name
            results["Pitch (+3)"] += int(pass_pitch_3)
            print(f"  [Pitch (+3)]       Predicted: {match_shifted_3} -> {'PASS ✅' if pass_pitch_3 else 'FAIL ❌'}")
            
            # 5. Noise + Pitch (+1)
            y_noisy_shifted = librosa.effects.pitch_shift(y_noisy, sr=sr, n_steps=1)
            _, _, S_noisy_shifted = compute_spectrogram(y_noisy_shifted, sr)
            match_noisy_shifted, _ = match_query(generate_hashes(get_constellation(S_noisy_shifted)), db)
            pass_both = match_noisy_shifted == song_name
            results["Noise + Pitch (+1)"] += int(pass_both)
            print(f"  [Noise+Pitch (+1)] Predicted: {match_noisy_shifted} -> {'PASS ✅' if pass_both else 'FAIL ❌'}")
            
        except Exception as e:
            print(f"  -> ERROR during processing: {e}")

    print("\n==================================================")
    print("                FINAL REPORT                      ")
    print("==================================================")
    for condition, passed in results.items():
        accuracy = (passed / valid_songs) * 100 if valid_songs > 0 else 0
        print(f"  {condition.ljust(20)}: {passed}/{valid_songs} ({accuracy:.1f}%)")
    print("==================================================")

if __name__ == "__main__":
    run_extreme_tests()
