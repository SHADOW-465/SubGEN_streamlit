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

# --- Custom CSS (Original Style + Fixes for White Elements) ---
st.markdown("""
<style>
    /* Base styles with recommended color scheme */
    :root {
        --primary: #40e0d0;         /* Turquoise Green */
        --secondary: #0b1a3d;       /* Deep Midnight Blue */
        --accent: #3b9eff;          /* Neon Blue */
        --accent-alt: #9f5afd;      /* Electric Purple */
        --dark: #111111;            /* Charcoal Black */
        --light: #e0ffff;           /* Light Cyan */
        --light-alt: #f1f1f1;       /* Off-White */
        --error: #ff6b6b;           /* Coral Red */
        --warning: #f9c74f;         /* Golden Yellow */
        --gradient: linear-gradient(135deg, var(--primary) 0%, var(--accent-alt) 100%);
        --card-bg: rgba(17, 17, 17, 0.25);
        --title-gradient: linear-gradient(90deg, var(--primary), var(--accent-alt));
        --background-gradient: linear-gradient(135deg, var(--secondary) 0%, #0a142e 50%, #111111 100%);
    }

    /* Overall page styling */
    .stApp {
        background: var(--background-gradient);
        color: var(--light);
        font-family: 'Inter', system-ui, sans-serif;
    }

    /* --- WIDGET FIXES (Prevent White Backgrounds) --- */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stDataFrame {
        background-color: rgba(11, 26, 61, 0.5) !important;
        color: var(--light) !important;
        border: 1px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 8px !important;
    }

    /* Fix for dropdown menus */
    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background-color: var(--secondary) !important;
    }

    /* Hover/Focus states for widgets */
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 10px rgba(64, 224, 208, 0.2) !important;
    }

    /* File Uploader styling */
    .stFileUploader > div > div {
        background-color: rgba(17, 17, 17, 0.25) !important;
        border: 2px dashed rgba(64, 224, 208, 0.5) !important;
        border-radius: 16px !important;
    }

    /* Glass effect classes */
    .glass {
        background: rgba(17, 17, 17, 0.25);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(64, 224, 208, 0.18);
        border-radius: 16px;
    }
    .glass-alt {
        background: rgba(11, 26, 61, 0.25);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(64, 224, 208, 0.18);
        border-radius: 16px;
    }

    /* Custom header with animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: var(--title-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        padding: 0.5rem;
        position: relative;
        animation: floating 3s ease-in-out infinite;
    }
    .subheader {
        text-align: center;
        font-size: 1.2rem;
        max-width: 800px;
        margin: 0 auto 2rem auto;
        color: var(--light);
    }

    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(11, 26, 61, 0.25) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(64, 224, 208, 0.3);
    }
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Cards styling */
    .card {
        background: var(--card-bg) !important;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(64, 224, 208, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .card-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Feature badges */
    .feature-badge {
        background: rgba(64, 224, 208, 0.1);
        border: 1px solid rgba(64, 224, 208, 0.3);
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        color: var(--light);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .feature-badge-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .dot-primary { background: var(--primary); box-shadow: 0 0 5px var(--primary); }
    .dot-accent { background: var(--accent); box-shadow: 0 0 5px var(--accent); }
    .dot-accent-alt { background: var(--accent-alt); box-shadow: 0 0 5px var(--accent-alt); }

    /* Buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent-alt) 100%) !important;
        color: var(--dark) !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(64, 224, 208, 0.4) !important;
    }
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 30px rgba(64, 224, 208, 0.6) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background: rgba(11, 26, 61, 0.25);
        border-radius: 16px;
        color: var(--light);
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

def generate_subtitles(video_path, model_size, device, compute_type, task="transcribe", language=None):
    """Generate subtitles using faster-whisper with cached model"""
    try:
        model = load_model(model_size, device, compute_type)
        segments, info = model.transcribe(video_path, beam_size=5, task=task, language=language)

        subtitles = []
        for segment in segments:
            subtitles.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        return subtitles
    except Exception as e:
        raise e

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

st.markdown('<h1 class="main-header floating">SubGEN PRO: AI-Subtitle Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Transform your videos with AI-powered subtitle generation. Fast, accurate, and beautifully designed.</p>', unsafe_allow_html=True)

# Feature badges
st.markdown("""
<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin-bottom: 3rem;">
    <div class="feature-badge">
        <span class="feature-badge-dot dot-primary"></span>
        Real-time Processing
    </div>
    <div class="feature-badge">
        <span class="feature-badge-dot dot-accent"></span>
        Multi-Language Support
    </div>
    <div class="feature-badge">
        <span class="feature-badge-dot dot-accent-alt"></span>
        Premium Editor
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown('<div class="sidebar-header"><span class="icon">🛠️</span>Configuration</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("**Model Settings**")
        model_type = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large-v3"], index=2)
        compute_type = st.selectbox("Compute Type", ["int8", "float16", "float32"], index=0)
        device_type = st.selectbox("Device", ["cpu", "cuda", "auto"], index=0)

        st.markdown("---")
        st.markdown("**Task Settings**")
        task_type = st.selectbox("Task", ["Transcribe", "Translate to English"], index=0)

        LANGUAGES = {"Auto Detection": None, "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it", "Japanese": "ja", "Hindi": "hi"}
        source_lang_name = st.selectbox("Source Language", list(LANGUAGES.keys()), index=0)
        source_lang_code = LANGUAGES[source_lang_name]

        st.markdown('</div>', unsafe_allow_html=True)

# --- Main Content Grid ---
col1, col2 = st.columns([3, 1])

# Initialize Session State
if 'subtitles' not in st.session_state: st.session_state.subtitles = []
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'subtitle_position' not in st.session_state: st.session_state.subtitle_position = "Bottom"
if 'font_size' not in st.session_state: st.session_state.font_size = 1.0

with col1:
    # Upload Section
    st.markdown('<div class="glass card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span class="icon">📤</span>Upload Media</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video or audio file", type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav'], label_visibility="collapsed")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
            st.session_state.video_path = video_path

        st.markdown(f'<div style="margin-top:10px; color:var(--primary);">✅ Ready: {uploaded_file.name}</div>', unsafe_allow_html=True)

        if st.button("🚀 Generate Subtitles", type="primary", use_container_width=True):
            with st.spinner("Processing media... This may take a moment."):
                whisper_task = "translate" if task_type == "Translate to English" else "transcribe"
                try:
                    subs = generate_subtitles(video_path, model_type, device_type, compute_type, whisper_task, source_lang_code)
                    st.session_state.subtitles = subs
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Preview & Editor Section
    if st.session_state.video_path and st.session_state.subtitles:
        st.markdown('<div class="glass card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="icon">📝</span>Editor & Preview</div>', unsafe_allow_html=True)

        # Video Player
        try:
            vtt_content = create_vtt_file(st.session_state.subtitles, st.session_state.subtitle_position)
            with open(st.session_state.video_path, "rb") as vf:
                video_b64 = base64.b64encode(vf.read()).decode()
            vtt_b64 = base64.b64encode(vtt_content.encode()).decode()

            video_html = f"""
            <div class="video-container" style="border-radius:10px; overflow:hidden; margin-bottom:1rem;">
                <video width="100%" controls style="background: #000;">
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                    <track src="data:text/vtt;base64,{vtt_b64}" kind="subtitles" srclang="en" label="English" default>
                    Your browser does not support the video tag.
                </video>
            </div>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        except Exception:
            st.warning("Video preview unavailable.")

        # Data Editor (New Feature in Old UI)
        st.info("Edit subtitles directly in the table below. Changes update automatically.")
        df = pd.DataFrame(st.session_state.subtitles)
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "start": st.column_config.NumberColumn("Start (s)", format="%.2f"),
                "end": st.column_config.NumberColumn("End (s)", format="%.2f"),
                "text": st.column_config.TextColumn("Subtitle Text", width="large"),
            },
            hide_index=True
        )

        # Sync changes
        current_data = edited_df.to_dict('records')
        if current_data != st.session_state.subtitles:
            st.session_state.subtitles = current_data
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.video_path and st.session_state.subtitles:
        st.markdown('<div class="glass-alt card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="icon">📥</span>Export</div>', unsafe_allow_html=True)

        vtt_dl = create_vtt_file(st.session_state.subtitles, st.session_state.subtitle_position)
        st.download_button("⬇️ Download VTT", vtt_dl, "subtitles.vtt", "text/vtt", use_container_width=True)
        st.download_button("⬇️ Download JSON", json.dumps(st.session_state.subtitles, indent=2), "subtitles.json", "application/json", use_container_width=True)

        st.markdown('<hr style="border-color: rgba(64,224,208,0.2);">', unsafe_allow_html=True)
        st.markdown("<strong>Settings</strong>", unsafe_allow_html=True)

        pos = st.selectbox("Position", ["Bottom", "Middle", "Top"], index=["Bottom", "Middle", "Top"].index(st.session_state.subtitle_position))
        if pos != st.session_state.subtitle_position:
            st.session_state.subtitle_position = pos
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer glass-alt">
    <p>SubGEN Pro v2.0 • Powered by Faster-Whisper & Streamlit</p>
</div>
""", unsafe_allow_html=True)
