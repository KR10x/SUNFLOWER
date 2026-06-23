import os
import hashlib
import numpy as np
import librosa
from scipy.signal import stft
from scipy.ndimage import maximum_filter

def load_audio(file_path, target_sample_rate=22050):
    audio_signal, sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    return audio_signal, sample_rate

def compute_spectrogram(
    audio_signal, 
    sample_rate, 
    window_size=1024, 
    overlap=512
):
    frequencies, times, complex_spectrogram = stft(
        audio_signal, 
        fs=sample_rate, 
        nperseg=window_size, 
        noverlap=overlap
    )
    magnitude_spectrogram = np.abs(complex_spectrogram)
    log_magnitude_spectrogram = 10 * np.log10(magnitude_spectrogram + 1e-10)
    return frequencies, times, log_magnitude_spectrogram

def get_constellation(
    log_magnitude_spectrogram, 
    filter_size=(15, 15), 
    threshold_percentile=90
):
    local_maxima_mask = maximum_filter(log_magnitude_spectrogram, size=filter_size) == log_magnitude_spectrogram
    amplitude_threshold = np.percentile(log_magnitude_spectrogram, threshold_percentile)
    above_threshold_mask = (log_magnitude_spectrogram > amplitude_threshold)
    peak_mask = local_maxima_mask & above_threshold_mask
    freq_bins, time_frames = np.where(peak_mask)
    return list(zip(time_frames, freq_bins))

def generate_hashes(constellation, fan_value=15):
    sorted_points = sorted(constellation, key=lambda point: point[0])
    fingerprint_hashes = []
    for anchor_idx in range(len(sorted_points) - fan_value):
        anchor_time, anchor_freq = sorted_points[anchor_idx]
        for target_offset in range(1, fan_value + 1):
            target_time, target_freq = sorted_points[anchor_idx + target_offset]
            time_difference = target_time - anchor_time
            if 0 < time_difference < 200:
                if anchor_freq == 0:
                    continue
                freq_ratio = target_freq / anchor_freq
                hash_string = f"{freq_ratio:.2f}_{time_difference}"
                fingerprint_hashes.append((hash_string, anchor_time))
    return fingerprint_hashes
