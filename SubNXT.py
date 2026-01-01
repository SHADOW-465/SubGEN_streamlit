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

# --- Premium Glassmorphism UI Styling ---
st.markdown("""
<style>
    /* Base Theme */
    :root {
        --primary-color: #40e0d0;
        --secondary-color: #3b9eff;
        --bg-color: #0e1117;
        --glass-bg: rgba(20, 24, 35, 0.75);
        --glass-border: 1px solid rgba(255, 255, 255, 0.08);
        --glow-color: rgba(64, 224, 208, 0.5);
    }

    .stApp {
        background-color: var(--bg-color);
        background-image: radial-gradient(circle at 10% 20%, rgba(64, 224, 208, 0.05) 0%, transparent 20%),
                          radial-gradient(circle at 90% 80%, rgba(59, 158, 255, 0.05) 0%, transparent 20%);
    }

    /* Inputs & Selectboxes */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 10px rgba(64, 224, 208, 0.2) !important;
    }

    /* Buttons with Glow */
    .stButton > button {
        background: linear-gradient(135deg, rgba(64, 224, 208, 0.1) 0%, rgba(59, 158, 255, 0.1) 100%);
        border: 1px solid rgba(64, 224, 208, 0.3);
        color: var(--primary-color);
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(5px);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px var(--glow-color);
        border-color: var(--primary-color);
        color: white;
    }

    /* File Uploader */
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 15px rgba(64, 224, 208, 0.15);
        background-color: rgba(255, 255, 255, 0.05);
    }

    /* Header Gradient & Animation */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        background: linear-gradient(90deg, #40e0d0, #3b9eff, #9f5afd);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        text-shadow: 0 0 30px rgba(64, 224, 208, 0.3);
        margin: 1rem 0;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    /* Subheader */
    .subheader {
        font-family: 'Inter', sans-serif;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        font-size: 1.2rem;
        letter-spacing: 0.5px;
        margin-bottom: 2rem;
    }

    /* Feature Badges */
    .feature-badge-container {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 3rem;
        flex-wrap: wrap;
    }
    .feature-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        color: #40e0d0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        backdrop-filter: blur(5px);
    }

    /* Data Editor */
    .stDataFrame {
        border: var(--glass-border);
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 5rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Logic & Backend ---

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

        # Transcribe
        segments, info = model.transcribe(video_path, beam_size=5, task=task, language=language)

        # Collect results
        subtitles = []
        for segment in segments:
            subtitles.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        return subtitles, info
    except Exception as e:
        raise e

def create_vtt_file(subtitles, position="Bottom"):
    """Create a VTT file content from subtitles with positioning"""
    vtt_content = "WEBVTT\n\n"

    # Map friendly position names to VTT line settings
    # Line 0 is top, Line 100% is bottom (default)
    # Middle is roughly 50%
    settings = ""
    if position == "Top":
        settings = " line:0%"
    elif position == "Middle":
        settings = " line:50%"
    # Bottom is default, no tag needed usually, or line:100%

    for subtitle in subtitles:
        start_time = format_timestamp(subtitle['start'])
        end_time = format_timestamp(subtitle['end'])
        text = subtitle['text']
        vtt_content += f"{start_time} --> {end_time}{settings}\n{text}\n\n"
    return vtt_content

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### 🛠️ Configuration")

    with st.expander("🤖 Model Settings", expanded=True):
        model_type = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=2,
            help="Select 'large-v3' for best accuracy, 'tiny' for speed."
        )

        compute_type = st.selectbox(
            "Compute Type",
            ["int8", "float16", "float32"],
            index=0,
            help="int8: Fastest/Compact. float16: High Perf (GPU). float32: Precision (CPU)."
        )
        st.caption("ℹ️ **int8** is recommended for CPU speed.")

        device_type = st.selectbox(
            "Device",
            ["cpu", "cuda", "auto"],
            index=0,
            help="Select 'cuda' if you have a compatible NVIDIA GPU."
        )

    with st.expander("🌍 Language & Task", expanded=True):
        task_type = st.selectbox(
            "Task",
            ["Transcribe", "Translate to English"],
            index=0,
            help="'Transcribe' keeps original language. 'Translate' converts to English."
        )

        # Common languages map
        LANGUAGES = {
            "Auto Detection": None,
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru",
            "Hindi": "hi",
            "Arabic": "ar"
        }

        source_lang_name = st.selectbox(
            "Source Language",
            list(LANGUAGES.keys()),
            index=0,
            help="Select the language spoken in the video. 'Auto' usually works best."
        )
        source_lang_code = LANGUAGES[source_lang_name]

    st.markdown("---")
    st.info("SubGEN Pro v2.0\nPowered by **Faster-Whisper**")

# --- Main Interface ---

# Header with Glow
st.markdown('<div class="main-header">SubGEN PRO: AI-Subtitle Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Transform your videos with AI-powered subtitle generation. Fast, accurate, and beautifully designed.</div>', unsafe_allow_html=True)

# Feature Badges
st.markdown("""
<div class="feature-badge-container">
    <div class="feature-badge">⚡ Real-time Processing</div>
    <div class="feature-badge">🎯 Multi-Language Support</div>
    <div class="feature-badge">🎬 Embedded Playback</div>
    <div class="feature-badge">✨ Premium Design</div>
</div>
""", unsafe_allow_html=True)

# Session State Initialization
if 'subtitles' not in st.session_state:
    st.session_state.subtitles = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'vtt_content' not in st.session_state:
    st.session_state.vtt_content = None
# Persist settings
if 'subtitle_position' not in st.session_state:
    st.session_state.subtitle_position = "Bottom"

# Layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 1. Upload Media")
    uploaded_file = st.file_uploader("Drop your video/audio file here", type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav'])

    if uploaded_file:
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.video_path = tmp_file.name

        # File Info - Styled Box
        st.markdown(f"""
        <div style="background: rgba(64, 224, 208, 0.1); border: 1px solid rgba(64, 224, 208, 0.3); border-radius: 8px; padding: 10px; margin: 10px 0;">
            <span style="color: #40e0d0; font-weight: bold;">✓ Ready:</span> {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)

        # Generate Button
        if st.button("🚀 Generate Subtitles", type="primary", use_container_width=True):
            with st.status("Processing media...", expanded=True) as status:
                st.write("Initializing Faster-Whisper Engine...")

                # Map task to whisper param
                whisper_task = "translate" if task_type == "Translate to English" else "transcribe"

                try:
                    subs, info = generate_subtitles(
                        st.session_state.video_path,
                        model_type,
                        device_type,
                        compute_type,
                        task=whisper_task,
                        language=source_lang_code
                    )
                    st.session_state.subtitles = subs
                    st.write("Formatting VTT...")
                    st.session_state.vtt_content = create_vtt_file(subs, st.session_state.subtitle_position)
                    status.update(label="Transcription Complete!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="Error Occurred", state="error")
                    st.error(f"Error: {str(e)}")

with col2:
    if st.session_state.video_path and st.session_state.subtitles:
        st.markdown("### 2. Editor & Preview")

        tab1, tab2, tab3 = st.tabs(["📝 Subtitle Editor", "🎥 Video Preview", "⚙️ Export Settings"])

        with tab1:
            st.markdown("Edit your subtitles below. Changes are applied automatically.")

            # Convert to DataFrame for Editor
            df = pd.DataFrame(st.session_state.subtitles)

            # Configure column config
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "start": st.column_config.NumberColumn("Start (s)", format="%.2f"),
                    "end": st.column_config.NumberColumn("End (s)", format="%.2f"),
                    "text": st.column_config.TextColumn("Subtitle Text", width="large")
                },
                hide_index=True
            )

            # Check for changes and update session state
            current_data = edited_df.to_dict('records')
            if current_data != st.session_state.subtitles:
                st.session_state.subtitles = current_data
                st.session_state.vtt_content = create_vtt_file(current_data, st.session_state.subtitle_position)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="⬇️ Download .VTT",
                    data=st.session_state.vtt_content,
                    file_name=f"subtitles.vtt",
                    mime="text/vtt",
                    use_container_width=True
                )
            with col_d2:
                st.download_button(
                    label="⬇️ Download .JSON",
                    data=json.dumps(st.session_state.subtitles, indent=2),
                    file_name=f"subtitles.json",
                    mime="application/json",
                    use_container_width=True
                )

        with tab2:
            # Video Player with Subtitles
            try:
                with open(st.session_state.video_path, "rb") as f:
                    video_bytes = f.read()
                    video_b64 = base64.b64encode(video_bytes).decode()

                vtt_b64 = base64.b64encode(st.session_state.vtt_content.encode()).decode()

                video_html = f"""
                    <video width="100%" controls style="border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);">
                        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                        <track src="data:text/vtt;base64,{vtt_b64}" kind="subtitles" srclang="en" label="English" default>
                        Your browser does not support the video tag.
                    </video>
                """
                st.markdown(video_html, unsafe_allow_html=True)
                st.info("Note: Refresh the page if video player doesn't update immediately after edits.")
            except Exception as e:
                st.error("Could not load video player.")

        with tab3:
            st.markdown("Customize your subtitle appearance (for supported players).")

            new_position = st.selectbox(
                "Subtitle Position",
                ["Bottom", "Middle", "Top"],
                index=["Bottom", "Middle", "Top"].index(st.session_state.subtitle_position),
                help="Adjust where subtitles appear on the video."
            )

            font_size = st.slider(
                "Font Size Scale", 0.5, 3.0, 1.0, 0.1,
                help="Note: This setting is saved but depends on the video player to respect it."
            )

            if new_position != st.session_state.subtitle_position:
                st.session_state.subtitle_position = new_position
                st.session_state.vtt_content = create_vtt_file(st.session_state.subtitles, new_position)
                st.success("Position updated! Check the Preview tab.")
                st.rerun()

    elif not st.session_state.video_path:
        st.info("👈 Start by uploading a video file from the sidebar or left panel.")

# Footer
st.markdown("""
<div class="footer">
    <p>SubGEN Pro v2.0 • Powered by <b style="color: #40e0d0;">Faster-Whisper</b> & Streamlit</p>
</div>
""", unsafe_allow_html=True)
