import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import load_audio, compute_spectrogram, get_constellation, generate_hashes
from src.database import SongDatabase
from src.matcher import match_query

# Configuration
DATASET_DIR = "seed"
TEST_SONG = "Let It Be.mp3"  # Change this to any song in the dataset
TEST_FILE = os.path.join(DATASET_DIR, TEST_SONG)
OUTPUT_DIR = "report_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_spectrogram(S_log, sr, title, filename, t, f):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_log, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.title(title)
    plt.colorbar(format="%+2.f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_constellation(S_log, sr, constellation, title, filename, t, f):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_log, sr=sr, x_axis='time', y_axis='linear', cmap='magma', alpha=0.5)
    
    if constellation:
        times, freqs = zip(*constellation)
        # map frame indices back to physical units roughly for plotting
        real_times = [t[idx] for idx in times if idx < len(t)]
        real_freqs = [f[idx] for idx in freqs if idx < len(f)]
        plt.scatter(real_times, real_freqs, color='red', s=10, marker='x')
        
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def experiment_window_length():
    print("Running Window Length Experiment...")
    y, sr = load_audio(TEST_FILE)
    # Take just 10 seconds to make plots clear
    y = y[:10 * sr]
    
    # Short Window
    f_short, t_short, S_short = compute_spectrogram(y, sr, nperseg=256, noverlap=128)
    plot_spectrogram(S_short, sr, 'Spectrogram (Short Window - 256)', 'spectrogram_short_window.png', t_short, f_short)
    
    # Long Window
    f_long, t_long, S_long = compute_spectrogram(y, sr, nperseg=4096, noverlap=2048)
    plot_spectrogram(S_long, sr, 'Spectrogram (Long Window - 4096)', 'spectrogram_long_window.png', t_long, f_long)
    print("Saved window length plots.")

def experiment_constellation():
    print("Running Constellation Experiment...")
    y, sr = load_audio(TEST_FILE)
    y = y[:10 * sr] # 10s
    f, t, S_log = compute_spectrogram(y, sr)
    constellation = get_constellation(S_log)
    plot_constellation(S_log, sr, constellation, 'Constellation of Peaks', 'constellation.png', t, f)
    print("Saved constellation plot.")

def experiment_noise_robustness():
    print("Running Noise Robustness Experiment...")
    y, sr = load_audio(TEST_FILE)
    # Take a 5 second snippet
    y_snippet = y[30*sr : 35*sr]
    
    db = SongDatabase("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        print("Database not found. Please run build_db.py first to test matching!")
        return
        
    # Original
    _, _, S_log = compute_spectrogram(y_snippet, sr)
    hashes = generate_hashes(get_constellation(S_log))
    match_orig, _ = match_query(hashes, db)
    print(f"Original 5s snippet matched to: {match_orig}")
    
    # Add noise
    noise = np.random.randn(len(y_snippet)) * 0.05
    y_noisy = y_snippet + noise
    _, _, S_noisy = compute_spectrogram(y_noisy, sr)
    hashes_noisy = generate_hashes(get_constellation(S_noisy))
    match_noisy, _ = match_query(hashes_noisy, db)
    print(f"Noisy 5s snippet matched to: {match_noisy}")

def experiment_pitch_shift():
    print("Running Pitch Shift Experiment...")
    y, sr = load_audio(TEST_FILE)
    y_snippet = y[30*sr : 35*sr]
    
    db = SongDatabase("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        return
        
    # Pitch shift up by 1 semitone
    y_shifted = librosa.effects.pitch_shift(y_snippet, sr=sr, n_steps=1)
    _, _, S_shifted = compute_spectrogram(y_shifted, sr)
    hashes_shifted = generate_hashes(get_constellation(S_shifted))
    match_shifted, _ = match_query(hashes_shifted, db)
    print(f"Pitch shifted (+1 step) matched to: {match_shifted}")

if __name__ == "__main__":
    if not os.path.exists(TEST_FILE):
        print(f"Test file {TEST_FILE} not found. Please check dataset.")
        sys.exit(1)
        
    experiment_window_length()
    experiment_constellation()
    experiment_noise_robustness()
    experiment_pitch_shift()
    print("All experiments done. Check 'report_plots' directory for images.")
