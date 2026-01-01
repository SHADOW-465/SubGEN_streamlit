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
    page_title="AutoSub Pro - AI Subtitle Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional UI Styling ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #40e0d0, #3b9eff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subheader {
        font-family: 'Inter', sans-serif;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards/Containers */
    .custom-card {
        background-color: #1e212b;
        border: 1px solid #2e313b;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1e212b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2e313b;
    }

    /* Inputs and Selectboxes */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #ffffff;
        border: 1px solid #404040;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Data Editor */
    .stDataFrame {
        border: 1px solid #2e313b;
        border-radius: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid #2e313b;
        color: #666;
        font-size: 0.85rem;
    }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .status-ready { background: rgba(64, 224, 208, 0.15); color: #40e0d0; }
    .status-processing { background: rgba(59, 158, 255, 0.15); color: #3b9eff; }

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

def generate_subtitles(video_path, model_size, device, compute_type):
    """Generate subtitles using faster-whisper with cached model"""
    try:
        model = load_model(model_size, device, compute_type)

        # Transcribe
        segments, info = model.transcribe(video_path, beam_size=5)

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

def create_vtt_file(subtitles):
    """Create a VTT file content from subtitles"""
    vtt_content = "WEBVTT\n\n"
    for subtitle in subtitles:
        start_time = format_timestamp(subtitle['start'])
        end_time = format_timestamp(subtitle['end'])
        text = subtitle['text']
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt_content

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    with st.expander("Model Configuration", expanded=True):
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
            help="int8 is faster/smaller, float16 requires GPU support, float32 is standard CPU."
        )

        device_type = st.selectbox(
            "Device",
            ["cpu", "cuda", "auto"],
            index=0,
            help="Select 'cuda' if you have a compatible NVIDIA GPU."
        )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "AutoSub Pro uses the latest **Faster-Whisper** engine for state-of-the-art transcription accuracy."
    )

# --- Main Interface ---

# Header
st.markdown('<div class="main-header">AutoSub Pro <span style="font-size: 0.5em; vertical-align: middle; color: #40e0d0; border: 1px solid #40e0d0; border-radius: 5px; padding: 2px 6px;">BETA</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Professional AI-powered subtitle generation and editing studio.</div>', unsafe_allow_html=True)

# Session State Initialization
if 'subtitles' not in st.session_state:
    st.session_state.subtitles = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'vtt_content' not in st.session_state:
    st.session_state.vtt_content = None

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

        # File Info
        st.success(f"File loaded: {uploaded_file.name}")

        # Generate Button
        if st.button("✨ Start Transcription", type="primary", use_container_width=True):
            with st.status("Processing media...", expanded=True) as status:
                st.write("Loading Whisper model...")
                start_time = time.time()
                try:
                    subs, info = generate_subtitles(st.session_state.video_path, model_type, device_type, compute_type)
                    st.session_state.subtitles = subs
                    st.write("Generating VTT...")
                    st.session_state.vtt_content = create_vtt_file(subs)
                    status.update(label="Transcription Complete!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="Error Occurred", state="error")
                    st.error(f"Error: {str(e)}")

with col2:
    if st.session_state.video_path and st.session_state.subtitles:
        st.markdown("### 2. Editor & Preview")

        tab1, tab2 = st.tabs(["📝 Subtitle Editor", "🎥 Video Preview"])

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
                st.session_state.vtt_content = create_vtt_file(current_data)
                # We don't rerun here to avoid jarring refreshes while typing,
                # but the VTT download will always be fresh.

            st.download_button(
                label="⬇️ Download .VTT",
                data=st.session_state.vtt_content,
                file_name=f"subtitles.vtt",
                mime="text/vtt",
                use_container_width=True
            )

            st.download_button(
                label="⬇️ Download .JSON",
                data=json.dumps(st.session_state.subtitles, indent=2),
                file_name=f"subtitles.json",
                mime="application/json",
                use_container_width=True
            )

        with tab2:
            # Video Player with Subtitles
            # We need to re-read the video to base64 for the HTML player
            try:
                with open(st.session_state.video_path, "rb") as f:
                    video_bytes = f.read()
                    video_b64 = base64.b64encode(video_bytes).decode()

                vtt_b64 = base64.b64encode(st.session_state.vtt_content.encode()).decode()

                video_html = f"""
                    <video width="100%" controls style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
                        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                        <track src="data:text/vtt;base64,{vtt_b64}" kind="subtitles" srclang="en" label="English" default>
                        Your browser does not support the video tag.
                    </video>
                """
                st.markdown(video_html, unsafe_allow_html=True)
                st.info("Note: If you update subtitles in the Editor tab, the video player needs a page refresh to pick up the new VTT blob in some browsers.")
            except Exception as e:
                st.error("Could not load video player.")

    elif not st.session_state.video_path:
        st.info("👈 Start by uploading a video file from the sidebar or left panel.")

# Footer
st.markdown("""
<div class="footer">
    AutoSub Pro v2.1 • Powered by OpenAI Whisper & Streamlit
</div>
""", unsafe_allow_html=True)
