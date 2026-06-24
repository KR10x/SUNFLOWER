import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import fetch_sound, get_spectro, extract_peaks, create_fingerprints
from src.database import TrackDB
from src.matcher import find_match

DATA_FOLDER = "seed"
TARGET_TRACK = "Let It Be.mp3"
FULL_TARGET_PATH = os.path.join(DATA_FOLDER, TARGET_TRACK)
PLOT_FOLDER = "report_plots"

os.makedirs(PLOT_FOLDER, exist_ok=True)

def render_spectro(log_mat, rate, header, out_name, timings, freqs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mat, sr=rate, x_axis='time', y_axis='linear', cmap='magma')
    plt.title(header)
    plt.colorbar(format="%+2.f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, out_name))
    plt.close()

def render_peaks(log_mat, rate, peak_list, header, out_name, timings, freqs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mat, sr=rate, x_axis='time', y_axis='linear', cmap='magma', alpha=0.5)
    
    if peak_list:
        t_vals, f_vals = zip(*peak_list)
        real_t = [timings[idx] for idx in t_vals if idx < len(timings)]
        real_f = [freqs[idx] for idx in f_vals if idx < len(freqs)]
        plt.scatter(real_t, real_f, color='red', s=10, marker='x')
        
    plt.title(header)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, out_name))
    plt.close()

def run_dft_check():
    print("Running DFT Experiment...")
    sig, rate = fetch_sound(FULL_TARGET_PATH)
    sig = sig[:30 * rate]
    
    Y_fft = np.fft.fft(sig)
    f_axis = np.fft.fftfreq(len(Y_fft), 1/rate)
    
    mask_pos = f_axis > 0
    f_axis = f_axis[mask_pos]
    mag_vals = np.abs(Y_fft[mask_pos])
    
    plt.figure(figsize=(10, 4))
    plt.plot(f_axis, mag_vals, color='blue', linewidth=0.5)
    plt.title('Magnitude DFT of the Audio Clip')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, 'dft_entire_song.png'))
    plt.close()
    print("Saved DFT plot.")

def run_win_len_check():
    print("Running Window Length Experiment...")
    sig, rate = fetch_sound(FULL_TARGET_PATH)
    sig = sig[:10 * rate]
    
    f_s, t_s, mat_s = get_spectro(sig, rate, win_len=256, lap=128)
    render_spectro(mat_s, rate, 'Spectrogram (Short Window - 256)', 'spectrogram_short_window.png', t_s, f_s)
    
    f_l, t_l, mat_l = get_spectro(sig, rate, win_len=4096, lap=2048)
    render_spectro(mat_l, rate, 'Spectrogram (Long Window - 4096)', 'spectrogram_long_window.png', t_l, f_l)
    print("Saved window length plots.")

def run_peak_check():
    print("Running Constellation Experiment...")
    sig, rate = fetch_sound(FULL_TARGET_PATH)
    sig = sig[:10 * rate]
    f_arr, t_arr, mat_log = get_spectro(sig, rate)
    p_list = extract_peaks(mat_log)
    render_peaks(mat_log, rate, p_list, 'Constellation of Peaks', 'constellation.png', t_arr, f_arr)
    print("Saved constellation plot.")

def run_noise_check():
    print("Running Noise Robustness Experiment...")
    sig, rate = fetch_sound(FULL_TARGET_PATH)
    clip = sig[30*rate : 35*rate]
    
    main_db = TrackDB("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        print("Database not found.")
        return
        
    _, _, mat_orig = get_spectro(clip, rate)
    h_orig = create_fingerprints(extract_peaks(mat_orig))
    res_orig, _ = find_match(h_orig, main_db)
    print(f"Original matched to: {res_orig}")
    
    static_ns = np.random.randn(len(clip)) * 0.05
    clip_ns = clip + static_ns
    _, _, mat_ns = get_spectro(clip_ns, rate)
    h_ns = create_fingerprints(extract_peaks(mat_ns))
    res_ns, _ = find_match(h_ns, main_db)
    print(f"Noisy matched to: {res_ns}")

def run_pitch_check():
    print("Running Pitch Shift Experiment...")
    sig, rate = fetch_sound(FULL_TARGET_PATH)
    clip = sig[30*rate : 35*rate]
    
    main_db = TrackDB("song_db.pkl")
    if not os.path.exists("song_db.pkl"):
        return
        
    clip_shifted = librosa.effects.pitch_shift(clip, sr=rate, n_steps=1)
    _, _, mat_shifted = get_spectro(clip_shifted, rate)
    h_shifted = create_fingerprints(extract_peaks(mat_shifted))
    res_shifted, _ = find_match(h_shifted, main_db)
    print(f"Shifted matched to: {res_shifted}")

if __name__ == "__main__":
    if not os.path.exists(FULL_TARGET_PATH):
        print("Test file not found.")
        sys.exit(1)
        
    run_dft_check()
    run_win_len_check()
    run_peak_check()
    run_noise_check()
    run_pitch_check()
    print("Experiments done.")
