import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import librosa.display

from src.audio_processor import fetch_sound, get_spectro, extract_peaks, create_fingerprints
from src.database import TrackDB
from src.matcher import find_match

st.set_page_config(page_title="Sunflower 🌻", layout="wide")

@st.cache_resource
def get_db():
    db_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "song_db.pkl")
    return TrackDB(db_loc)

main_db = get_db()

st.title("Sunflower 🌻")
st.markdown("Upload a clip to identify the song based on its acoustic fingerprint!")

ui_mode = st.sidebar.radio("Select Mode", ["Single Clip Mode", "Batch Mode"])

if ui_mode == "Single Clip Mode":
    st.header("Single Clip Identification")
    up_file = st.file_uploader("Upload an audio clip", type=['wav', 'mp3', 'ogg', 'm4a'])
    
    if up_file is not None:
        st.audio(up_file, format='audio/wav')
        
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_f:
                tmp_f.write(up_file.getvalue())
                tmp_p = tmp_f.name
                
            try:
                sig, rate = fetch_sound(tmp_p)
                freqs, timings, log_mat = get_spectro(sig, rate)
                peaks = extract_peaks(log_mat)
                q_hashes = create_fingerprints(peaks)
                
                track_res, hist_res = find_match(q_hashes, main_db)
                
                st.success(f"**Matched Song:** {track_res}")
                
                st.subheader("Intermediate Steps")
                
                c1, c2 = st.columns(2)
                
                with c1:
                    f1, a1 = plt.subplots(figsize=(10, 4))
                    img_map = librosa.display.specshow(log_mat, sr=rate, x_axis='time', y_axis='linear', ax=a1, cmap='magma')
                    a1.set_title('Spectrogram')
                    f1.colorbar(img_map, ax=a1, format="%+2.f dB")
                    st.pyplot(f1)
                    
                    if hist_res:
                        f3, a3 = plt.subplots(figsize=(10, 4))
                        diffs, counts = zip(*hist_res.items())
                        a3.bar(diffs, counts, width=1)
                        a3.set_title('Offset Histogram (True match = sharp peak)')
                        a3.set_xlabel('Time Offset (frames)')
                        a3.set_ylabel('Number of matching hashes')
                        st.pyplot(f3)
                        
                with c2:
                    f2, a2 = plt.subplots(figsize=(10, 4))
                    librosa.display.specshow(log_mat, sr=rate, x_axis='time', y_axis='linear', ax=a2, cmap='magma', alpha=0.5)
                    t_pts, f_pts = zip(*peaks) if peaks else ([], [])
                    act_t = [timings[idx] for idx in t_pts] if len(timings) > max(t_pts, default=-1) else []
                    act_f = [freqs[idx] for idx in f_pts] if len(freqs) > max(f_pts, default=-1) else []
                    
                    a2.scatter(act_t, act_f, color='red', s=10, marker='x')
                    a2.set_title('Constellation Map')
                    st.pyplot(f2)
                    
            except Exception as err:
                st.error(f"Error processing audio: {err}")
            finally:
                os.unlink(tmp_p)

elif ui_mode == "Batch Mode":
    st.header("Batch Identification")
    up_files = st.file_uploader("Upload multiple audio clips", type=['wav', 'mp3', 'ogg', 'm4a'], accept_multiple_files=True)
    
    if up_files and st.button("Process Batch"):
        out_rows = []
        p_bar = st.progress(0)
        
        for idx, f_obj in enumerate(up_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_f:
                tmp_f.write(f_obj.getvalue())
                tmp_p = tmp_f.name
                
            try:
                sig, rate = fetch_sound(tmp_p)
                _, _, log_mat = get_spectro(sig, rate)
                peaks = extract_peaks(log_mat)
                q_hashes = create_fingerprints(peaks)
                track_res, _ = find_match(q_hashes, main_db)
                
                out_rows.append({"filename": f_obj.name, "prediction": track_res})
            except Exception as err:
                out_rows.append({"filename": f_obj.name, "prediction": f"Error: {err}"})
            finally:
                os.unlink(tmp_p)
                
            p_bar.progress((idx + 1) / len(up_files))
            
        res_df = pd.DataFrame(out_rows)
        st.dataframe(res_df)
        
        csv_data = res_df.to_csv(index=False)
        st.download_button(
            label="Download results.csv",
            data=csv_data,
            file_name="results.csv",
            mime="text/csv",
        )
