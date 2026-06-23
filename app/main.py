import sys
import os

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import librosa.display

from src.audio_processor import load_audio, compute_spectrogram, get_constellation, generate_hashes
from src.database import SongDatabase
from src.matcher import match_query

st.set_page_config(page_title="Sunflower 🌻", layout="wide")

@st.cache_resource
def load_db():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "song_db.pkl")
    return SongDatabase(db_path)

db = load_db()

st.title("Sunflower 🌻")
st.markdown("Upload a clip to identify the song based on its acoustic fingerprint!")

mode = st.sidebar.radio("Select Mode", ["Single Clip Mode", "Batch Mode"])

if mode == "Single Clip Mode":
    st.header("Single Clip Identification")
    uploaded_file = st.file_uploader("Upload an audio clip", type=['wav', 'mp3', 'ogg', 'm4a'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner("Processing..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                # Process audio
                y, sr = load_audio(tmp_path)
                f, t, S_log = compute_spectrogram(y, sr)
                constellation = get_constellation(S_log)
                query_hashes = generate_hashes(constellation)
                
                match_name, offset_hist = match_query(query_hashes, db)
                
                st.success(f"**Matched Song:** {match_name}")
                
                # Visualizations
                st.subheader("Intermediate Steps")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot Spectrogram
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    img = librosa.display.specshow(S_log, sr=sr, x_axis='time', y_axis='linear', ax=ax1, cmap='magma')
                    ax1.set_title('Spectrogram')
                    fig1.colorbar(img, ax=ax1, format="%+2.f dB")
                    st.pyplot(fig1)
                    
                    # Plot Histogram
                    if offset_hist:
                        fig3, ax3 = plt.subplots(figsize=(10, 4))
                        offsets, counts = zip(*offset_hist.items())
                        ax3.bar(offsets, counts, width=1)
                        ax3.set_title('Offset Histogram (True match = sharp peak)')
                        ax3.set_xlabel('Time Offset (frames)')
                        ax3.set_ylabel('Number of matching hashes')
                        st.pyplot(fig3)
                        
                with col2:
                    # Plot Constellation
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    # Background spectrogram
                    librosa.display.specshow(S_log, sr=sr, x_axis='time', y_axis='linear', ax=ax2, cmap='magma', alpha=0.5)
                    # Peaks
                    times, freqs = zip(*constellation) if constellation else ([], [])
                    # t array is time in seconds, f array is frequency in Hz
                    # Our constellation is in frame indices
                    real_times = [t[idx] for idx in times] if len(t) > max(times, default=-1) else []
                    real_freqs = [f[idx] for idx in freqs] if len(f) > max(freqs, default=-1) else []
                    
                    ax2.scatter(real_times, real_freqs, color='red', s=10, marker='x')
                    ax2.set_title('Constellation Map')
                    st.pyplot(fig2)
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                os.unlink(tmp_path)

elif mode == "Batch Mode":
    st.header("Batch Identification")
    uploaded_files = st.file_uploader("Upload multiple audio clips", type=['wav', 'mp3', 'ogg', 'm4a'], accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Batch"):
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
                
            try:
                y, sr = load_audio(tmp_path)
                f, t, S_log = compute_spectrogram(y, sr)
                constellation = get_constellation(S_log)
                query_hashes = generate_hashes(constellation)
                match_name, _ = match_query(query_hashes, db)
                
                results.append({"filename": file.name, "prediction": match_name})
            except Exception as e:
                results.append({"filename": file.name, "prediction": f"Error: {e}"})
            finally:
                os.unlink(tmp_path)
                
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name="results.csv",
            mime="text/csv",
        )
