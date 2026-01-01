import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile
import time
import json
import base64
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
# Configure Streamlit page
st.set_page_config(
    page_title="SubGEN Pro: AI Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS with glass-morphism and modern animations
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
        min-height: 100vh;
    }
    /* Glass effect */
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
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(64, 224, 208, 0.4); }
        50% { box-shadow: 0 0 40px rgba(64, 224, 208, 0.8); }
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(11, 26, 61, 0.25) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(64, 224, 208, 0.3);
        box-shadow: 0 0 20px rgba(64, 224, 208, 0.3);
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
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 30px rgba(64, 224, 208, 0.3);
        border-color: var(--primary);
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
    .download-btn {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-alt) 100%) !important;
        color: white !important;
    }
    /* Generate button special style */
    .generate-btn .stButton>button {
        background: linear-gradient(135deg, #18ed71 0%, #764ba2 100%) !important;
        animation: pulse-glow 2s infinite;
    }
    .generate-btn .stButton>button:hover {
        background: linear-gradient(135deg, #18ed71 0%, #9f5afd 100%) !important;
        box-shadow: 0 0 40px rgba(24, 237, 113, 0.6) !important;
    }
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
        border-radius: 10px;
        height: 12px !important;
    }
    .progress-container {
        background: rgba(17, 17, 17, 0.7);
        border-radius: 10px;
        padding: 1rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(64, 224, 208, 0.3);
    }
    /* Video player container */
    .video-container {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.7);
        margin-bottom: 1.5rem;
        background: #000;
        border: 1px solid rgba(64, 224, 208, 0.3);
    }
    /* Subtitle timeline */
    .subtitle-timeline {
        background: rgba(17, 17, 17, 0.25);
        padding: 15px;
        border-radius: 16px;
        margin-top: 20px;
    }
    .subtitle-item {
        background: rgba(64, 224, 208, 0.1);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 3px solid var(--primary);
        transition: all 0.3s ease;
    }
    .subtitle-item:hover {
        background: rgba(64, 224, 208, 0.2);
        transform: translateX(5px);
    }
    .subtitle-time {
        color: var(--primary);
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 5px;
    }
    .subtitle-text {
        font-size: 1.1rem;
        line-height: 1.5;
        color: var(--light);
    }
    /* File uploader */
    .stFileUploader>div>div {
        background: rgba(17, 17, 17, 0.25) !important;
        border: 2px dashed rgba(64, 224, 208, 0.5) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .stFileUploader>div>div:hover {
        border-color: var(--primary) !important;
        background: rgba(25, 25, 35, 0.3) !important;
        transform: scale(1.01);
    }
    /* Selectbox (Dropdown) styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--secondary) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(64, 224, 208, 0.3) !important;
        color: var(--light) !important;
    }
    /* Text area styling */
    .stTextArea textarea {
        background-color: rgba(17, 17, 17, 0.25) !important;
        color: var(--light) !important;
        border: 1px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 8px !important;
    }
    /* Expander styling */
    .stExpander {
        background: rgba(17, 17, 17, 0.25) !important;
        border: 1px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 16px !important;
        margin-bottom: 0.8rem !important;
    }
    .stExpander summary {
        background: rgba(64, 224, 208, 0.15) !important;
        padding: 1rem !important;
        border-radius: 16px 16px 0 0 !important;
        font-weight: 600 !important;
        color: var(--primary) !important;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background: rgba(11, 26, 61, 0.25);
        border-radius: 16px;
@@ -394,27 +410,27 @@
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
def generate_subtitles(video_path, model_type):
    """Generate subtitles using faster-whisper"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown(
            f'<div class="glass card"><div class="card-header"><span class="icon">⏳</span>Loading Whisper model...</div></div>',
            unsafe_allow_html=True
        )
        progress_bar.progress(20)
        model = WhisperModel(model_type, device="cpu", compute_type="int8")
        status_text.markdown(
            f'<div class="glass card"><div class="card-header"><span class="icon">🎙️</span>Transcribing video...</div></div>',
            unsafe_allow_html=True
        )
        progress_bar.progress(40)
        segments, info = model.transcribe(video_path, beam_size=5)
        segments, _ = model.transcribe(video_path, beam_size=5, task="translate")

        status_text.markdown(
            f'<div class="glass card"><div class="card-header"><span class="icon">✍️</span>Processing subtitles...</div></div>',
            unsafe_allow_html=True
        )
        progress_bar.progress(80)
        # Format subtitles for display
        subtitles = []
        for segment in segments:
            subtitles.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        progress_bar.progress(100)
        status_text.markdown(
            f'<div class="glass card"><div class="card-header"><span class="icon">✅</span>Subtitles generated successfully!</div></div>',
            unsafe_allow_html=True
        )
        time.sleep(1.5)
        progress_bar.empty()
        status_text.empty()
        return subtitles
    except Exception as e:
        st.error(f"Error generating subtitles: {str(e)}")
        return None
def get_subtitle_at_time(subtitles, current_time):
    """Get the subtitle text for the current time"""
    if not subtitles:
        return ""
    for subtitle in subtitles:
        if subtitle['start'] <= current_time <= subtitle['end']:
            return subtitle['text']
    return ""
def create_vtt_file(subtitles):
    """Create a VTT file content from subtitles"""
    vtt_content = "WEBVTT\n\n"
    for subtitle in subtitles:
        start_time = format_timestamp(subtitle['start'])
        end_time = format_timestamp(subtitle['end'])
        text = subtitle['text']
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt_content
def get_base64_encoded_file(file_path):
    """Return base64 encoded file"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
# Main UI
st.markdown('<h1 class="main-header floating">SubGEN PRO:AI-Subtitle Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Transform your videos with AI-powered subtitle generation. Fast, accurate, and beautifully designed.</p>', unsafe_allow_html=True)
# Sidebar for controls
with st.sidebar:
    st.markdown('<div class="sidebar-header"><span class="icon">🛠️</span>Configuration</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        model_type = st.selectbox(
            "**Whisper Model**",
            ["tiny", "base", "small"],
            index=2,
            help="Larger models are more accurate but slower"
        )
        st.markdown("""
        <div class="info-box">
            <h4>Model Comparison:</h4>
            <ul>
                <li><strong>Tiny</strong>: Fastest, basic accuracy</li>
                <li><strong>Base</strong>: Good balance</li>
                <li><strong>Small</strong>: Recommended for most users</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
# Feature badges
st.markdown("""
<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin-bottom: 3rem;">
    <div class="feature-badge">
        <span class="feature-badge-dot dot-primary"></span>
        Real-time Processing
    </div>
    <div class="feature-badge">
        <span class="feature-badge-dot dot-accent"></span>
        Multiple Formats
    </div>
    <div class="feature-badge">
        <span class="feature-badge-dot dot-accent-alt"></span>
        Embedded Playback
    </div>
</div>
""", unsafe_allow_html=True)
# Main content grid
col1, col2 = st.columns([3, 1])
with col1:
    # Upload Section
    with st.container():
        st.markdown('<div class="glass card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="icon">📤</span>Upload Media</div>', unsafe_allow_html=True)
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video or audio file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v', 'mp3', 'wav', 'flac'],
            help="Upload your video or audio file to generate subtitles",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
                st.session_state.video_path = video_path
            # Display file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 1.5rem;">🎥</span>
                        <div>
                            <div style="font-weight: 600;">{uploaded_file.name}</div>
                            <div style="font-size: 0.9rem; color: var(--light-alt);">{file_size:.2f} MB</div>
                        </div>
                    </div>
                    <span style="color: var(--primary); font-size: 1.5rem;">✓</span>
                </div>
            """, unsafe_allow_html=True)
            # Generate subtitles button with custom style
            st.markdown('<div class="generate-btn" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            if st.button("🚀 Generate Subtitles", type="primary", use_container_width=True):
                st.session_state.processing = True
                with st.spinner(""):
                    st.markdown('<div style="text-align: center; font-size: 1.5rem; padding: 2rem; color: var(--primary);">Processing your media... ⚙️</div>', unsafe_allow_html=True)
                    subtitles = generate_subtitles(video_path, model_type)
                    if subtitles:
                        st.session_state.subtitles = subtitles
                        st.session_state.processing = False
                        # Encode video and subtitles for embedding
                        try:
                            # Encode video
                            with open(video_path, "rb") as video_file:
                                video_bytes = video_file.read()
                                st.session_state.video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                            # Encode VTT
                            vtt_content = create_vtt_file(subtitles)
                            st.session_state.vtt_base64 = base64.b64encode(vtt_content.encode('utf-8')).decode('utf-8')
                        except Exception as e:
                            st.error(f"Error preparing video: {str(e)}")
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
        # Video Preview Section
        if st.session_state.video_path and st.session_state.subtitles:
            st.markdown('<div class="glass card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="icon">🎥</span>Video Preview</div>', unsafe_allow_html=True)
            if st.session_state.video_base64 and st.session_state.vtt_base64:
                # Create HTML video player with embedded subtitles
                video_html = f"""
                <div class="video-container">
                    <video width="100%" height="360" controls style="border-radius: 10px; background: #000;">
                        <source src="data:video/mp4;base64,{st.session_state.video_base64}" type="video/mp4">
                        <track src="data:text/vtt;base64,{st.session_state.vtt_base64}" kind="subtitles"
                               srclang="en" label="English" default>
                        Your browser does not support the video tag.
                    </video>
                </div>
                """
                st.markdown(video_html, unsafe_allow_html=True)
            else:
                st.warning("Video content not available. Please regenerate subtitles.")
            st.markdown('</div>', unsafe_allow_html=True)  # Close card
            # Subtitle Editor Section
            st.markdown('<div class="glass card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="icon">📝</span>Subtitle Timeline</div>', unsafe_allow_html=True)
            st.info("Click on any subtitle to edit its text. Changes will be reflected in the video player.")
            # Create a scrollable container for subtitles
            subtitle_container = st.container()
            with subtitle_container:
                for i, subtitle in enumerate(st.session_state.subtitles):
                    start_time = subtitle['start']
                    end_time = subtitle['end']
                    text = subtitle['text']
                    # Format time display
                    start_formatted = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    end_formatted = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                    with st.expander(f"🕒 {start_formatted} - {end_formatted}", expanded=False):
                        # Display current subtitle
                        st.markdown(f'<div class="subtitle-text">{text}</div>', unsafe_allow_html=True)
                        # Edit subtitle option
                        edited_text = st.text_area(
                            "Edit subtitle:",
                            value=text,
                            key=f"edit_{i}",
                            height=100
                        )
                        if st.button(f"Update Subtitle {i+1}", key=f"update_{i}"):
                            st.session_state.subtitles[i]['text'] = edited_text
                            # Update VTT content
                            vtt_content = create_vtt_file(st.session_state.subtitles)
                            st.session_state.vtt_base64 = base64.b64encode(vtt_content.encode('utf-8')).decode('utf-8')
                            st.success("Subtitle updated! Refresh the page to see changes in the video player.")
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)  # Close card
with col2:
    # Progress Card
    if st.session_state.processing:
        with st.container():
            st.markdown('<div class="glass-alt card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="icon">⏳</span>Processing</div>', unsafe_allow_html=True)
            # Simulated progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            # Simulate progress
            for percent_complete in range(100):
                time.sleep(0.05)
                progress_bar.progress(percent_complete + 1)
                progress_text.markdown(f'<div style="text-align: center; color: var(--light);">{percent_complete + 1}% complete</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close card
    # Download Card
    if st.session_state.video_path and st.session_state.subtitles:
        with st.container():
            st.markdown('<div class="glass-alt card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="icon">📥</span>Export Subtitles</div>', unsafe_allow_html=True)
            # Download VTT file
            vtt_content = create_vtt_file(st.session_state.subtitles)
            st.download_button(
                label="⬇️ Download VTT File",
                data=vtt_content,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.vtt",
                mime="text/vtt",
                use_container_width=True
            )
            # Download JSON file
            json_content = json.dumps(st.session_state.subtitles, indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_content,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitles.json",
                mime="application/json",
                use_container_width=True
            )
            # Subtitle settings
            st.markdown('<div class="glass" style="padding: 1rem; border-radius: 12px; margin-top: 1.5rem;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: var(--primary);"><span class="icon">⚙️</span>Subtitle Settings</h4>', unsafe_allow_html=True)
            # Font size slider
            font_size = st.slider("Font Size", 1.0, 3.0, st.session_state.font_size, 0.1,
                                 help="Adjust subtitle font size",
                                 key="font_size_slider")
            st.session_state.font_size = font_size
            # Position selector
            position = st.selectbox("Position",
                                  ["Bottom (Default)", "Middle", "Top"],
                                  index=0,
                                  help="Position of subtitles on video",
                                  key="position_select")
            st.session_state.position = position
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close card
    # Features Card
    with st.container():
        st.markdown('<div class="glass-alt card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="icon">✨</span>Features</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="padding: 0.5rem 0;">
            <div style="display: flex; align-items: center; margin: 15px 0;">
                <span style="color: var(--primary); font-size: 1.5rem; margin-right: 10px;">✓</span>
                <span>AI-powered transcription</span>
            </div>
            <div style="display: flex; align-items: center; margin: 15px 0;">
                <span style="color: var(--accent); font-size: 1.5rem; margin-right: 10px;">✓</span>
                <span>Embedded video playback</span>
            </div>
            <div style="display: flex; align-items: center; margin: 15px 0;">
                <span style="color: var(--accent-alt); font-size: 1.5rem; margin-right: 10px;">✓</span>
                <span>Multiple export formats</span>
            </div>
            <div style="display: flex; align-items: center; margin: 15px 0;">
                <span style="color: var(--primary); font-size: 1.5rem; margin-right: 10px;">✓</span>
                <span>Real-time preview</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
# Footer
st.markdown("""
<div class="footer glass-alt">
    <p style="color: var(--light);">Built with ❤️ using Streamlit and OpenAI Whisper • SubNXT Pro v2.0</p>
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
        <a href="#" style="color: var(--primary);">🌐 Website</a>
        <a href="#" style="color: var(--accent);">🐦 Twitter</a>
        <a href="#" style="color: var(--accent-alt);">💼 LinkedIn</a>
        <a href="#" style="color: var(--primary);">📧 Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)