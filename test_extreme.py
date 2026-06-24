import os
import sys
import numpy as np
import librosa

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import fetch_sound, get_spectro, extract_peaks, create_fingerprints
from src.database import TrackDB
from src.matcher import find_match

def fire_extreme_tests():
    data_folder = "seed"
    track_list = [t for t in os.listdir(data_folder) if t.endswith('.mp3')]
    
    main_db = TrackDB("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        print("Database missing.")
        return

    print("==================================================")
    print(f"    EXTREME ROBUSTNESS TESTING ON {len(track_list)} SONGS")
    print("==================================================")
    
    scores = {
        "Clean": 0,
        "Static": 0,
        "Shifted (+1)": 0,
        "Shifted (+3)": 0,
        "Static + Shift (+1)": 0
    }
    
    total_cnt = len(track_list)
    good_cnt = 0

    for idx, t_name in enumerate(track_list):
        clean_title = os.path.splitext(t_name)[0]
        full_p = os.path.join(data_folder, t_name)
        
        print(f"\n[{idx+1}/{total_cnt}] Testing: {clean_title}")
        
        try:
            sig, rate = fetch_sound(full_p)
            idx_start = 30 * rate
            idx_end = 35 * rate
            if len(sig) < idx_end:
                idx_start = 0
                idx_end = 5 * rate
            
            clip = sig[idx_start:idx_end]
            
            if len(clip) == 0:
                print("  -> ERROR: Clip too short.")
                continue
                
            good_cnt += 1

            _, _, mat_c = get_spectro(clip, rate)
            res_c, _ = find_match(create_fingerprints(extract_peaks(mat_c)), main_db)
            is_c = res_c == clean_title
            scores["Clean"] += int(is_c)
            print(f"  [Clean]            Predicted: {res_c} -> {'PASS ✅' if is_c else 'FAIL ❌'}")
            
            ns = np.random.randn(len(clip)) * 0.05
            clip_ns = clip + ns
            _, _, mat_ns = get_spectro(clip_ns, rate)
            res_ns, _ = find_match(create_fingerprints(extract_peaks(mat_ns)), main_db)
            is_ns = res_ns == clean_title
            scores["Static"] += int(is_ns)
            print(f"  [Static]           Predicted: {res_ns} -> {'PASS ✅' if is_ns else 'FAIL ❌'}")
            
            clip_s1 = librosa.effects.pitch_shift(clip, sr=rate, n_steps=1)
            _, _, mat_s1 = get_spectro(clip_s1, rate)
            res_s1, _ = find_match(create_fingerprints(extract_peaks(mat_s1)), main_db)
            is_s1 = res_s1 == clean_title
            scores["Shifted (+1)"] += int(is_s1)
            print(f"  [Shift (+1)]       Predicted: {res_s1} -> {'PASS ✅' if is_s1 else 'FAIL ❌'}")

            clip_s3 = librosa.effects.pitch_shift(clip, sr=rate, n_steps=3)
            _, _, mat_s3 = get_spectro(clip_s3, rate)
            res_s3, _ = find_match(create_fingerprints(extract_peaks(mat_s3)), main_db)
            is_s3 = res_s3 == clean_title
            scores["Shifted (+3)"] += int(is_s3)
            print(f"  [Shift (+3)]       Predicted: {res_s3} -> {'PASS ✅' if is_s3 else 'FAIL ❌'}")
            
            clip_ns1 = librosa.effects.pitch_shift(clip_ns, sr=rate, n_steps=1)
            _, _, mat_ns1 = get_spectro(clip_ns1, rate)
            res_ns1, _ = find_match(create_fingerprints(extract_peaks(mat_ns1)), main_db)
            is_ns1 = res_ns1 == clean_title
            scores["Static + Shift (+1)"] += int(is_ns1)
            print(f"  [Static+Shift (+1)] Predicted: {res_ns1} -> {'PASS ✅' if is_ns1 else 'FAIL ❌'}")
            
        except Exception as err:
            print(f"  -> ERROR: {err}")

    print("\n==================================================")
    print("                FINAL REPORT                      ")
    print("==================================================")
    for test_cond, passed_cnt in scores.items():
        acc = (passed_cnt / good_cnt) * 100 if good_cnt > 0 else 0
        print(f"  {test_cond.ljust(20)}: {passed_cnt}/{good_cnt} ({acc:.1f}%)")
    print("==================================================")

if __name__ == "__main__":
    fire_extreme_tests()
