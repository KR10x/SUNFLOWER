import os
import hashlib
import numpy as np
import librosa
from scipy.signal import stft
from scipy.ndimage import maximum_filter

def fetch_sound(filepath, sr_target=22050):
    sig, current_sr = librosa.load(filepath, sr=sr_target, mono=True)
    return sig, current_sr

def get_spectro(
    sound_wave, 
    rate, 
    win_len=1024, 
    lap=512
):
    freqs, timings, complex_spec = stft(
        sound_wave, 
        fs=rate, 
        nperseg=win_len, 
        noverlap=lap
    )
    mag_spec = np.abs(complex_spec)
    log_spec = 10 * np.log10(mag_spec + 1e-10)
    return freqs, timings, log_spec

def extract_peaks(
    log_matrix, 
    box_size=(15, 15), 
    cutoff_pct=90
):
    max_spots = maximum_filter(log_matrix, size=box_size) == log_matrix
    amp_cutoff = np.percentile(log_matrix, cutoff_pct)
    valid_spots = (log_matrix > amp_cutoff)
    final_peaks = max_spots & valid_spots
    f_bins, t_frames = np.where(final_peaks)
    return list(zip(t_frames, f_bins))

def create_fingerprints(peak_points, fan_out=15):
    ordered_peaks = sorted(peak_points, key=lambda p: p[0])
    hashes = []
    for base_idx in range(len(ordered_peaks) - fan_out):
        base_t, base_f = ordered_peaks[base_idx]
        for offset_val in range(1, fan_out + 1):
            target_t, target_f = ordered_peaks[base_idx + offset_val]
            t_diff = target_t - base_t
            if 0 < t_diff < 200:
                if base_f == 0:
                    continue
                f_ratio = target_f / base_f
                hash_tag = f"{f_ratio:.2f}_{t_diff}"
                hashes.append((hash_tag, base_t))
    return hashes
