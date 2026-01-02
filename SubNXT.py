import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile
import time
import json
import base64
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="SubGEN Pro: AI Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Extended Language Support (ISO-639-1) ---
LANGUAGES = {
    "Auto Detection": None,
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Russian": "ru", "Japanese": "ja", "Chinese": "zh",
    "Korean": "ko", "Hindi": "hi", "Arabic": "ar", "Turkish": "tr", "Polish": "pl",
    "Swedish": "sv", "Norwegian": "no", "Danish": "da", "Finnish": "fi", "Thai": "th",
    "Vietnamese": "vi", "Indonesian": "id", "Ukrainian": "uk", "Greek": "el", "Czech": "cs",
    "Hebrew": "he", "Romanian": "ro", "Hungarian": "hu", "Bengali": "bn", "Malay": "ms",
    "Catalan": "ca", "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Urdu": "ur"
}

# --- Custom CSS (Refined & Fixed) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    :root {
        --primary: #00f2ea;         /* Cyan Neon */
        --secondary: #0b1a3d;       /* Deep Navy */
        --accent: #ff0055;          /* Magenta Neon */
        --bg-dark: #050505;         /* Pure Black */
        --glass-bg: rgba(20, 20, 20, 0.6);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-main: #ffffff;
        --text-muted: #a0a0a0;
    }

    /* Global App Style */
    .stApp {
        background-color: var(--bg-dark);
        background-image: radial-gradient(circle at 50% 0%, #1a2a4a 0%, #050505 60%);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    /* --- Sidebar Styling --- */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.9);
        border-right: 1px solid var(--glass-border);
    }

    /* --- Widget Styling (Fixing White Elements) --- */

    /* Input Fields (Selectbox, Text, Number) */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        background-color: rgba(30, 30, 30, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }

    /* Hover & Focus Effects (Glow) */
    .stSelectbox > div > div:hover,
    .stTextInput > div > div:hover,
    .stNumberInput > div > div:hover {
        border-color: var(--primary) !important;
        box-shadow: 0 0 10px rgba(0, 242, 234, 0.2);
    }

    /* Dropdown Options Text Color */
    .stSelectbox div[data-baseweb="select"] span {
        color: white !important;
    }

    /* File Uploader */
    .stFileUploader > div > div {
        background-color: rgba(20, 20, 25, 0.6) !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: var(--primary) !important;
        background-color: rgba(0, 242, 234, 0.05) !important;
        box-shadow: 0 0 15px rgba(0, 242, 234, 0.1);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00f2ea 0%, #00a8a8 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 242, 234, 0.4) !important;
    }

    /* Custom Cards (CSS Classes for st.markdown) */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .glass-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Typography */
    h1 {
        background: linear-gradient(90deg, #ffffff, #a0a0a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Data Editor / Dataframe */
    .stDataFrame {
        border: 1px solid var(--glass-border) !important;
        border-radius: 8px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }

</style>
""", unsafe_allow_html=True)

# --- Backend Logic ---

def format_timestamp(seconds):
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
    milliseconds = int((seconds % 1) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

@st.cache_resource
def load_model(model_size, device, compute_type):
    """Load and cache the Whisper model"""
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def create_vtt_file(subtitles, position="Bottom"):
    """Create a VTT file content from subtitles"""
    vtt_content = "WEBVTT\n\n"
    settings = ""
    if position == "Top":
        settings = " line:0%"
    elif position == "Middle":
        settings = " line:50%"

    for subtitle in subtitles:
        start_time = format_timestamp(subtitle['start'])
        end_time = format_timestamp(subtitle['end'])
        text = subtitle['text']
        vtt_content += f"{start_time} --> {end_time}{settings}\n{text}\n\n"
    return vtt_content

# --- Main UI Structure ---

# Header
st.title("SubGEN PRO")
st.markdown("Transform your videos with AI-powered subtitle generation. Fast, accurate, and beautifully designed.")

# Stats/Features Row
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown('<div style="padding:10px; border-radius:8px; background:rgba(0,242,234,0.1); border:1px solid rgba(0,242,234,0.3); text-align:center;">⚡ Real-time Processing</div>', unsafe_allow_html=True)
with col_f2:
    st.markdown('<div style="padding:10px; border-radius:8px; background:rgba(255,0,85,0.1); border:1px solid rgba(255,0,85,0.3); text-align:center;">🌍 35+ Languages</div>', unsafe_allow_html=True)
with col_f3:
    st.markdown('<div style="padding:10px; border-radius:8px; background:rgba(159,90,253,0.1); border:1px solid rgba(159,90,253,0.3); text-align:center;">✨ Smart Editor</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### 🛠️ Configuration")

    st.markdown('<p style="font-size:0.8rem; color:#888;">MODEL SETTINGS</p>', unsafe_allow_html=True)
    model_type = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large-v3"], index=2, help="Larger models are more accurate but slower.")
    compute_type = st.selectbox("Compute Type", ["int8", "float16", "float32"], index=0, help="int8 is faster/less memory, float16 is standard.")
    device_type = st.selectbox("Device", ["cpu", "cuda", "auto"], index=0)

    st.markdown("---")
    st.markdown('<p style="font-size:0.8rem; color:#888;">TASK SETTINGS</p>', unsafe_allow_html=True)
    task_type = st.selectbox("Task", ["Transcribe", "Translate to English"], index=0)

    source_lang_name = st.selectbox("Source Language", list(LANGUAGES.keys()), index=0)
    source_lang_code = LANGUAGES[source_lang_name]

# --- Main Content ---
col1, col2 = st.columns([2, 1])

# Initialize Session State
if 'subtitles' not in st.session_state: st.session_state.subtitles = []
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'subtitle_position' not in st.session_state: st.session_state.subtitle_position = "Bottom"
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False

with col1:
    # 1. Upload Section
    st.markdown('<div class="glass-header">1. Upload Media</div>', unsafe_allow_html=True)

    # We don't wrap file_uploader in a div anymore to avoid layout bugs.
    # The CSS class .stFileUploader styles it directly.
    uploaded_file = st.file_uploader("Drop your video/audio file here", type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav'])

    if uploaded_file is not None:
        # Save temp file
        if st.session_state.video_path is None or os.path.basename(st.session_state.video_path) != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.video_path = tmp_file.name
                # Reset subtitles if new file
                st.session_state.subtitles = []
                st.session_state.processing_complete = False

        st.success(f"File loaded: {uploaded_file.name}")

        # 2. Action Button
        if st.button("🚀 Start Generation", type="primary", use_container_width=True):
            st.session_state.subtitles = []
            st.session_state.processing_complete = False

            # --- REAL-TIME STREAMING LOGIC ---
            status_container = st.status("Initializing AI Model...", expanded=True)
            progress_bar = status_container.progress(0)

            try:
                whisper_task = "translate" if task_type == "Translate to English" else "transcribe"

                # Load model
                model = load_model(model_type, device_type, compute_type)
                status_container.write("Model loaded. Transcribing...")

                # Generator
                segments_generator, info = model.transcribe(
                    st.session_state.video_path,
                    beam_size=5,
                    task=whisper_task,
                    language=source_lang_code
                )

                duration = info.duration
                processed_time = 0

                # Placeholder for live updates (optional, but data_editor below updates on rerun)
                # Since we are inside the button callback, we can't update data_editor easily without rerun.
                # We will show a live text preview instead.
                live_preview = st.empty()
                live_text = ""

                result_subs = []

                for segment in segments_generator:
                    processed_time = segment.end
                    progress = min(processed_time / duration, 1.0)
                    progress_bar.progress(progress)

                    seg_data = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip()
                    }
                    result_subs.append(seg_data)

                    # Live Preview Update
                    live_text += f"[{format_timestamp(segment.start)}] {segment.text.strip()}\n"
                    live_preview.code(live_text[-500:], language=None) # Show last 500 chars

                st.session_state.subtitles = result_subs
                st.session_state.processing_complete = True
                status_container.update(label="Processing Complete!", state="complete", expanded=False)
                st.rerun()

            except Exception as e:
                status_container.update(label="Error occurred", state="error")
                st.error(f"Processing Error: {str(e)}")

    # 3. Editor (Only shows after processing)
    if st.session_state.processing_complete and st.session_state.subtitles:
        st.markdown("---")
        st.markdown('<div class="glass-header">2. Editor</div>', unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.subtitles)
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "start": st.column_config.NumberColumn("Start (s)", format="%.2f", width="small"),
                "end": st.column_config.NumberColumn("End (s)", format="%.2f", width="small"),
                "text": st.column_config.TextColumn("Subtitle Text", width="large"),
            },
            hide_index=True,
            key="editor"
        )

        # Sync changes from editor back to session state
        # Note: st.data_editor updates session state automatically if key is set,
        # but we need to ensure our download buttons use the latest data.
        # This explicit assignment ensures compatibility.
        current_data = edited_df.to_dict('records')
        st.session_state.subtitles = current_data

with col2:
    # Right Column: Video Preview & Export
    if st.session_state.video_path:
        st.markdown('<div class="glass-header">Preview</div>', unsafe_allow_html=True)

        # We need to regenerate VTT for the video player every time subtitles change
        vtt_content = create_vtt_file(st.session_state.subtitles, st.session_state.subtitle_position)

        # Video Player with Subtitles
        # Note: Streamlit's st.video doesn't support VTT tracks easily.
        # We use HTML/JS injection for pro feel.
        try:
            with open(st.session_state.video_path, "rb") as vf:
                video_bytes = vf.read()
                video_b64 = base64.b64encode(video_bytes).decode()
            vtt_b64 = base64.b64encode(vtt_content.encode()).decode()

            video_html = f"""
            <div style="border-radius:12px; overflow:hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                <video width="100%" controls style="background: #000;">
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                    <track src="data:text/vtt;base64,{vtt_b64}" kind="subtitles" srclang="en" label="English" default>
                    Your browser does not support the video tag.
                </video>
            </div>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning("Video preview loading...")

        if st.session_state.subtitles:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="glass-header">Export</div>', unsafe_allow_html=True)

            # Export Options
            col_ex1, col_ex2 = st.columns(2)

            vtt_file = create_vtt_file(st.session_state.subtitles, st.session_state.subtitle_position)
            json_file = json.dumps(st.session_state.subtitles, indent=2)

            with col_ex1:
                st.download_button("Download .VTT", vtt_file, "subtitles.vtt", "text/vtt", use_container_width=True)
            with col_ex2:
                st.download_button("Download .JSON", json_file, "subtitles.json", "application/json", use_container_width=True)

            st.markdown("---")
            st.caption("Subtitle Settings")
            pos = st.selectbox("On-Screen Position", ["Bottom", "Middle", "Top"], index=["Bottom", "Middle", "Top"].index(st.session_state.subtitle_position))
            if pos != st.session_state.subtitle_position:
                st.session_state.subtitle_position = pos
                st.rerun()

# Footer
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1rem; border-top:1px solid rgba(255,255,255,0.1); color: #666;">
    <small>SubGEN Pro AI v2.1 • Faster-Whisper • Streamlit</small>
</div>
""", unsafe_allow_html=True)
