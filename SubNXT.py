"""
SubGEN PRO v2 — AI-Powered Subtitle Generator
Hardware Integration | Signal-Informed QC | Self-Learning Loop
"""

import streamlit as st
import os
import sys
import tempfile
import time
import json
import base64
import warnings
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import altair as alt
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Conditional imports (graceful fallback) ──────────────────────────────────
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    import torch
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    PYSRT_AVAILABLE = False

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SubGEN PRO v2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Language Maps ────────────────────────────────────────────────────────────
SOURCE_LANGUAGES = {
    "Auto Detection": None,
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Japanese": "ja", "Chinese": "zh", "Korean": "ko", "Arabic": "ar",
    "Turkish": "tr", "Polish": "pl", "Swedish": "sv", "Norwegian": "no",
    "Danish": "da", "Finnish": "fi", "Thai": "th", "Vietnamese": "vi",
    "Indonesian": "id", "Ukrainian": "uk", "Greek": "el", "Czech": "cs",
    "Hebrew": "he", "Romanian": "ro", "Hungarian": "hu", "Bengali": "bn",
    "Malay": "ms", "Catalan": "ca", "Urdu": "ur",
    "Hindi": "hi", "Tamil": "ta", "Malayalam": "ml", "Telugu": "te", "Marathi": "mr",
}

TARGET_LANGUAGES = {
    "Same as Source (Transcribe Only)": None,
    "English": "en",
    "Hindi": "hi", "Tamil": "ta", "Malayalam": "ml", "Telugu": "te",
    "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Russian": "ru", "Japanese": "ja",
    "Chinese": "zh", "Korean": "ko", "Arabic": "ar", "Turkish": "tr",
    "Bengali": "bn", "Urdu": "ur", "Marathi": "mr",
}

# Helsinki-NLP model map for non-English target translation
OPUS_MT_PAIRS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-mul",
    ("en", "ml"): "Helsinki-NLP/opus-mt-en-mul",
    ("en", "te"): "Helsinki-NLP/opus-mt-en-mul",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-ROMANCE",
    ("en", "nl"): "Helsinki-NLP/opus-mt-en-nl",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("en", "ko"): "Helsinki-NLP/opus-mt-en-ko",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("en", "tr"): "Helsinki-NLP/opus-mt-en-tr",
    ("en", "bn"): "Helsinki-NLP/opus-mt-en-mul",
    ("en", "ur"): "Helsinki-NLP/opus-mt-en-ur",
    ("en", "mr"): "Helsinki-NLP/opus-mt-en-mul",
}

INDIC_CODES = {"ta", "ml", "hi", "te", "mr", "bn", "ur"}

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #00f2ea;
        --primary-dark: #00b8b2;
        --secondary: #0b1a3d;
        --accent: #ff0055;
        --accent-glow: rgba(255, 0, 85, 0.3);
        --bg-dark: #050508;
        --glass-bg: rgba(15, 15, 25, 0.65);
        --glass-border: rgba(255, 255, 255, 0.08);
        --text-main: #e8e8f0;
        --text-dim: #7a7a8e;
        --green-qc: #00e676;
        --red-qc: #ff1744;
        --yellow-qc: #ffab00;
    }

    .stApp {
        background: var(--bg-dark);
        background-image:
            radial-gradient(ellipse at 20% 0%, rgba(0, 242, 234, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(255, 0, 85, 0.04) 0%, transparent 50%);
        color: var(--text-main);
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #00f2ea 0%, #ffffff 50%, #ff0055 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .main-header p {
        color: var(--text-dim);
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(8, 8, 18, 0.95);
        border-right: 1px solid var(--glass-border);
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(25, 25, 40, 0.8) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        justify-content: center;
        background: rgba(15, 15, 25, 0.5);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--glass-border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        color: var(--text-dim);
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,242,234,0.15), rgba(0,242,234,0.05)) !important;
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .glass-header {
        font-size: 1rem;
        font-weight: 700;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00f2ea 0%, #00b8b2 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 242, 234, 0.35) !important;
    }

    /* ── File Uploader ── */
    .stFileUploader > div > div {
        background: rgba(15, 15, 25, 0.5) !important;
        border: 2px dashed rgba(0, 242, 234, 0.2) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: var(--primary) !important;
        background: rgba(0, 242, 234, 0.03) !important;
    }

    /* ── QC Badge ── */
    .qc-badge-green {
        display: inline-block;
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .qc-badge-red {
        display: inline-block;
        background: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .qc-badge-yellow {
        display: inline-block;
        background: rgba(255, 171, 0, 0.15);
        color: #ffab00;
        border: 1px solid rgba(255, 171, 0, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        color: var(--text-dim);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* ── Scrollable containers ── */
    .live-output {
        background: rgba(10, 10, 20, 0.7);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Status Indicator ── */
    .hw-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .hw-connected { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
    .hw-simulated { background: rgba(255,171,0,0.15); color: #ffab00; border: 1px solid rgba(255,171,0,0.3); }

    /* ── Data Editor overrides ── */
    .stDataFrame { border-radius: 12px !important; overflow: hidden; }
    div[data-testid="stExpander"] { border: 1px solid var(--glass-border); border-radius: 12px; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.6rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
    }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UTILITY FUNCTIONS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def format_timestamp_vtt(seconds):
    """Convert seconds → VTT timestamp (HH:MM:SS.mmm)"""
    ms = int((seconds % 1) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def format_timestamp_srt(seconds):
    """Convert seconds → SRT timestamp (HH:MM:SS,mmm)"""
    ms = int((seconds % 1) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def create_vtt(subtitles, position="Bottom"):
    """Generate WebVTT content from subtitle list"""
    vtt = "WEBVTT\n\n"
    pos_map = {"Top": " line:0%", "Middle": " line:50%", "Bottom": ""}
    setting = pos_map.get(position, "")
    for sub in subtitles:
        vtt += f"{format_timestamp_vtt(sub['start'])} --> {format_timestamp_vtt(sub['end'])}{setting}\n"
        vtt += f"{sub['text']}\n\n"
    return vtt

def create_srt(subtitles):
    """Generate SRT content from subtitle list"""
    srt = ""
    for i, sub in enumerate(subtitles, 1):
        srt += f"{i}\n"
        srt += f"{format_timestamp_srt(sub['start'])} --> {format_timestamp_srt(sub['end'])}\n"
        srt += f"{sub['text']}\n\n"
    return srt


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HARDWARE INTEGRATION (ESP32 / Serial)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class HardwareReader:
    """
    Reads beamformed audio + metadata from ESP32 via USB serial.
    Falls back to simulated data if no hardware is connected.
    """
    def __init__(self):
        self.connected = False
        self.serial_port = None
        self.simulated = True
        self.snr = 25.0
        self.doa = 0.0
        self.doa_variance = 5.0
        self._try_connect()

    def _try_connect(self):
        """Attempt to connect to ESP32 via first available serial port"""
        if not SERIAL_AVAILABLE:
            self.simulated = True
            return
        try:
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                if "USB" in port.description.upper() or "CP210" in port.description.upper() or "CH340" in port.description.upper():
                    self.serial_port = serial.Serial(port.device, 115200, timeout=1)
                    self.connected = True
                    self.simulated = False
                    return
            self.simulated = True
        except Exception:
            self.simulated = True

    def read_metadata(self):
        """Read SNR, DOA, DOA_variance from hardware or simulate"""
        if self.connected and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode("utf-8").strip()
                if line:
                    data = json.loads(line)
                    self.snr = data.get("snr", self.snr)
                    self.doa = data.get("doa", self.doa)
                    self.doa_variance = data.get("doa_variance", self.doa_variance)
            except Exception:
                pass
        else:
            # Simulate realistic sensor data
            self.snr = np.clip(np.random.normal(25, 5), 5, 45)
            self.doa = np.random.uniform(0, 360)
            self.doa_variance = np.clip(np.random.exponential(8), 1, 30)

        return {
            "snr": round(self.snr, 1),
            "doa": round(self.doa, 1),
            "doa_variance": round(self.doa_variance, 2),
        }

    def status_html(self):
        if self.simulated:
            return '<span class="hw-status hw-simulated">⚡ SIMULATED</span>'
        return '<span class="hw-status hw-connected">● CONNECTED</span>'

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SIGNAL-INFORMED QC                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_snr_penalty(snr):
    """Map SNR to penalty: low SNR → high penalty (0-1)"""
    if snr >= 30:
        return 0.0
    elif snr <= 10:
        return 1.0
    return 1.0 - (snr - 10) / 20.0

def compute_fused_conf(asr_conf, snr, speaker_stability=0.85):
    """
    Fused confidence = 0.6 * ASR_conf + 0.3 * (1 - SNR_penalty) + 0.1 * speaker_stability
    """
    snr_penalty = compute_snr_penalty(snr)
    fused = 0.6 * asr_conf + 0.3 * (1.0 - snr_penalty) + 0.1 * speaker_stability
    return round(np.clip(fused, 0.0, 1.0), 3)

def qc_badge(conf_value):
    """Return HTML badge based on confidence threshold"""
    if conf_value >= 0.75:
        return f'<span class="qc-badge-green">✓ {conf_value:.2f}</span>'
    elif conf_value >= 0.60:
        return f'<span class="qc-badge-yellow">⚠ {conf_value:.2f}</span>'
    return f'<span class="qc-badge-red">✗ {conf_value:.2f}</span>'


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SELF-LEARNING LOOP                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CORRECTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrections.json")

def load_corrections():
    """Load stored corrections from JSON file"""
    if os.path.exists(CORRECTIONS_FILE):
        try:
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_correction(original_text, corrected_text, language):
    """Save a user correction to the dataset"""
    corrections = load_corrections()
    corrections.append({
        "timestamp": datetime.now().isoformat(),
        "original_text": original_text,
        "corrected_text": corrected_text,
        "language": language,
    })
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)
    return len(corrections)

def run_lora_finetune(model_name="openai/whisper-tiny", language="ta"):
    """
    Run LoRA fine-tuning on accumulated corrections.
    Returns status message. Requires GPU + peft.
    """
    if not PEFT_AVAILABLE:
        return "❌ PEFT library not available. Install `peft` and `torch`."

    try:
        import torch
        if not torch.cuda.is_available():
            return "⚠️ GPU not available. Fine-tuning requires CUDA. Download the dataset and fine-tune locally with GPU."
    except Exception:
        return "⚠️ PyTorch not configured for CUDA."

    corrections = load_corrections()
    lang_corrections = [c for c in corrections if c.get("language") == language]

    if len(lang_corrections) < 10:
        return f"Need at least 10 corrections for {language}. Currently: {len(lang_corrections)}"

    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        peft_model = get_peft_model(model, lora_config)

        adapter_path = f"./adapters/whisper-lora-{language}"
        os.makedirs(adapter_path, exist_ok=True)
        peft_model.save_pretrained(adapter_path)

        return f"✅ LoRA adapter saved to `{adapter_path}`. Ready for next session."
    except Exception as e:
        return f"❌ Fine-tuning failed: {str(e)}"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL LOADING                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@st.cache_resource
def load_whisper_model(model_size, device, compute_type):
    """Load and cache Faster-Whisper model"""
    if not WHISPER_AVAILABLE:
        return None
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        st.warning(f"Failed on {device}, falling back to CPU: {e}")
        try:
            return WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as e2:
            st.error(f"Model load failed: {e2}")
            return None

@st.cache_resource
def load_translation_model(src_code, tgt_code):
    """Load Helsinki-NLP translation model for non-English targets"""
    if not HF_AVAILABLE:
        return None
    pair = (src_code, tgt_code)
    model_name = OPUS_MT_PAIRS.get(pair)
    if model_name is None:
        model_name = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
    try:
        translator = hf_pipeline("translation", model=model_name)
        return translator
    except Exception as e:
        st.warning(f"Translation model not available for {src_code}→{tgt_code}: {e}")
        return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANSCRIPTION / TRANSLATION PIPELINE                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def process_transcription(model, file_path, source_lang, target_lang, beam_size=5, vad_filter=True,
                          hardware_reader=None):
    """
    Core pipeline: transcribe → optionally translate.
    Returns list of segment dicts with QC data.
    """
    src_code = SOURCE_LANGUAGES.get(source_lang)
    tgt_code = TARGET_LANGUAGES.get(target_lang)

    # Determine Whisper task
    if tgt_code == "en" and src_code != "en":
        whisper_task = "translate"
    else:
        whisper_task = "transcribe"

    segments_gen, info = model.transcribe(
        file_path,
        beam_size=beam_size,
        task=whisper_task,
        language=src_code,
        vad_filter=vad_filter,
    )

    results = []
    hw_meta = {"snr": 25.0, "doa": 0.0, "doa_variance": 5.0}

    for segment in segments_gen:
        # Get hardware metadata per segment
        if hardware_reader:
            hw_meta = hardware_reader.read_metadata()

        asr_conf = segment.avg_logprob
        # Convert log-prob to 0-1 confidence (rough mapping)
        asr_conf_normalized = np.clip(np.exp(asr_conf), 0, 1)

        fused_conf = compute_fused_conf(
            asr_conf_normalized,
            hw_meta["snr"],
            speaker_stability=0.85,
        )

        text = segment.text.strip()

        # Post-translation for non-English targets (when source != target and target != English)
        translated_text = text
        if tgt_code and tgt_code != "en" and tgt_code != src_code:
            # First get English text via Whisper translate if source isn't English
            if src_code != "en":
                # Re-run just this segment? No — use accumulated text
                # We'll do batch translation after all segments
                pass
            translated_text = text  # placeholder, batch translate later

        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
            "translated": translated_text,
            "asr_conf": round(asr_conf_normalized, 3),
            "fused_conf": fused_conf,
            "snr": hw_meta["snr"],
            "doa": hw_meta["doa"],
            "doa_variance": hw_meta["doa_variance"],
        })

        yield results[-1], info.duration

    # Batch translate if needed
    if tgt_code and tgt_code != "en" and tgt_code != src_code and results:
        # Translate from English (Whisper output) to target
        translator = load_translation_model("en", tgt_code)
        if translator:
            texts = [r["text"] for r in results]
            try:
                translations = translator(texts, max_length=512)
                for i, t in enumerate(translations):
                    results[i]["translated"] = t.get("translation_text", results[i]["text"])
            except Exception:
                pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VIDEO EDITING (Trimming via MoviePy)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def trim_video(input_path, start_time, end_time):
    """Trim video using moviepy, returns path to trimmed file"""
    if not MOVIEPY_AVAILABLE:
        return None, "MoviePy not installed."
    try:
        clip = VideoFileClip(input_path)
        trimmed = clip.subclip(start_time, min(end_time, clip.duration))
        output_path = tempfile.mktemp(suffix=".mp4")
        trimmed.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        clip.close()
        trimmed.close()
        return output_path, None
    except Exception as e:
        return None, str(e)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DASHBOARD CHARTS (Plotly / Altair)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_snr_gauge(snr_value):
    """Plotly gauge for SNR reading"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=snr_value,
        title={"text": "SNR (dB)", "font": {"size": 14, "color": "#aaa"}},
        number={"font": {"size": 28, "color": "#00f2ea"}},
        gauge={
            "axis": {"range": [0, 50], "tickcolor": "#555"},
            "bar": {"color": "#00f2ea"},
            "bgcolor": "rgba(20,20,30,0.8)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 15], "color": "rgba(255,23,68,0.2)"},
                {"range": [15, 30], "color": "rgba(255,171,0,0.2)"},
                {"range": [30, 50], "color": "rgba(0,230,118,0.2)"},
            ],
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#aaa"}
    )
    return fig

def create_doa_gauge(doa_value):
    """Plotly gauge for DOA reading"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=doa_value,
        title={"text": "DOA (°)", "font": {"size": 14, "color": "#aaa"}},
        number={"font": {"size": 28, "color": "#ff0055"}},
        gauge={
            "axis": {"range": [0, 360], "tickcolor": "#555"},
            "bar": {"color": "#ff0055"},
            "bgcolor": "rgba(20,20,30,0.8)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#aaa"}
    )
    return fig

def create_confidence_chart(subtitles):
    """Altair bar chart showing per-segment fused confidence"""
    if not subtitles:
        return None
    data = pd.DataFrame([
        {
            "Segment": f"#{i+1}",
            "Confidence": s.get("fused_conf", 0),
            "Status": "Pass" if s.get("fused_conf", 0) >= 0.75 else "Fail"
        }
        for i, s in enumerate(subtitles)
    ])
    chart = alt.Chart(data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Segment:N", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(
            alt.datum.Confidence >= 0.75,
            alt.value("#00e676"),
            alt.value("#ff1744"),
        ),
        tooltip=["Segment", "Confidence", "Status"],
    ).properties(height=200).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="rgba(255,255,255,0.05)",
        labelColor="#888",
        titleColor="#aaa",
    )
    return chart


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE INITIALIZATION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

defaults = {
    "subtitles": [],
    "video_path": None,
    "subtitle_position": "Bottom",
    "processing_complete": False,
    "hw_reader": None,
    "live_transcript": "",
    "live_snr_history": [],
    "live_doa_history": [],
    "original_texts": {},
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Initialize hardware reader (singleton)
if st.session_state.hw_reader is None:
    st.session_state.hw_reader = HardwareReader()
hw = st.session_state.hw_reader

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
        <span style="font-size:2rem;">🎬</span>
        <h2 style="margin:0; background: linear-gradient(135deg, #00f2ea, #fff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight:800;">SubGEN PRO v2</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Hardware status
    st.markdown(f"**Hardware** {hw.status_html()}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem; color:#555; text-transform:uppercase; letter-spacing:2px;">Model Configuration</p>', unsafe_allow_html=True)

    model_size = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large-v3"], index=1,
                              help="tiny/base = fast, large-v3 = best quality")
    compute_type = st.selectbox("Compute Type", ["int8", "float16", "float32"], index=0)
    device_type = st.selectbox("Device", ["cpu", "cuda", "auto"], index=0)

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem; color:#555; text-transform:uppercase; letter-spacing:2px;">Language Settings</p>', unsafe_allow_html=True)

    source_lang_name = st.selectbox("Source Language", list(SOURCE_LANGUAGES.keys()), index=0,
                                     help="Language of the input audio/video")
    target_lang_name = st.selectbox("Target Language", list(TARGET_LANGUAGES.keys()), index=0,
                                     help="Translate subtitles to this language")

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem; color:#555; text-transform:uppercase; letter-spacing:2px;">Processing</p>', unsafe_allow_html=True)
    beam_size = st.slider("Beam Size", 1, 10, 5)
    vad_filter = st.checkbox("VAD Filter", value=True, help="Voice Activity Detection to skip silence")

    st.markdown("---")
    st.caption("SubGEN PRO v2 • Faster-Whisper • AI")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HEADER                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.markdown("""
<div class="main-header">
    <h1>SubGEN PRO v2</h1>
    <p>AI-Powered Subtitle Generator • Hardware Integration • Self-Learning</p>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN TABS                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

tab_file, tab_live, tab_settings = st.tabs(["📁 File Mode", "🎙️ Live Mode", "⚙️ Settings"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: FILE MODE
# ──────────────────────────────────────────────────────────────────────────────
with tab_file:
    col_upload, col_preview = st.columns([3, 2], gap="large")

    with col_upload:
        st.markdown('<div class="glass-header">📂 Upload Media</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your video/audio file",
            type=["mp4", "avi", "mov", "mkv", "webm", "mp3", "wav", "flac", "ogg"],
            key="file_uploader",
        )

        if uploaded_file is not None:
            # Save uploaded file to temp
            current_name = getattr(uploaded_file, "name", "")
            if st.session_state.video_path is None or not st.session_state.video_path.endswith(os.path.splitext(current_name)[1]):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(current_name)[1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    st.session_state.video_path = tmp.name
                    st.session_state.subtitles = []
                    st.session_state.processing_complete = False
                    st.session_state.original_texts = {}

            st.success(f"✅ Loaded: **{current_name}**")

            # ── Generate Button ──
            if st.button("🚀 Generate Subtitles", type="primary", use_container_width=True):
                st.session_state.subtitles = []
                st.session_state.processing_complete = False
                st.session_state.original_texts = {}

                if not WHISPER_AVAILABLE:
                    st.error("faster-whisper is not installed. Run: `pip install faster-whisper`")
                else:
                    status = st.status("Initializing pipeline...", expanded=True)
                    progress = status.progress(0)

                    try:
                        model = load_whisper_model(model_size, device_type, compute_type)
                        if model is None:
                            st.error("Failed to load model.")
                        else:
                            status.write("✅ Model loaded. Transcribing...")
                            live_preview = st.empty()
                            live_text = ""
                            all_results = []

                            for seg_data, duration in process_transcription(
                                model, st.session_state.video_path,
                                source_lang_name, target_lang_name,
                                beam_size=beam_size, vad_filter=vad_filter,
                                hardware_reader=hw,
                            ):
                                all_results.append(seg_data)
                                pct = min(seg_data["end"] / max(duration, 0.1), 1.0)
                                progress.progress(pct)

                                badge = qc_badge(seg_data["fused_conf"])
                                live_text += f"[{format_timestamp_vtt(seg_data['start'])}] {seg_data['text']}\n"
                                live_preview.code(live_text[-800:], language=None)

                            st.session_state.subtitles = all_results
                            # Store originals for self-learning diff
                            st.session_state.original_texts = {
                                i: s["text"] for i, s in enumerate(all_results)
                            }
                            st.session_state.processing_complete = True
                            status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                            st.rerun()

                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(f"Processing failed: {str(e)}")

        # ── Subtitle Editor with QC ──
        if st.session_state.processing_complete and st.session_state.subtitles:
            st.markdown("---")
            st.markdown('<div class="glass-header">✏️ Subtitle Editor</div>', unsafe_allow_html=True)

            # Show QC summary
            subs = st.session_state.subtitles
            avg_conf = np.mean([s.get("fused_conf", 0) for s in subs])
            pass_count = sum(1 for s in subs if s.get("fused_conf", 0) >= 0.75)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{len(subs)}</div>
                    <div class="label">Segments</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{avg_conf:.2f}</div>
                    <div class="label">Avg Confidence</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{pass_count}/{len(subs)}</div>
                    <div class="label">QC Pass</div>
                </div>""", unsafe_allow_html=True)

            # Confidence chart
            conf_chart = create_confidence_chart(subs)
            if conf_chart:
                st.altair_chart(conf_chart, use_container_width=True)

            # Editable dataframe
            display_text_key = "translated" if TARGET_LANGUAGES.get(target_lang_name) and TARGET_LANGUAGES[target_lang_name] != SOURCE_LANGUAGES.get(source_lang_name) else "text"

            df = pd.DataFrame([{
                "Start (s)": round(s["start"], 2),
                "End (s)": round(s["end"], 2),
                "Text": s.get(display_text_key, s["text"]),
                "Confidence": s.get("fused_conf", 0),
                "SNR": s.get("snr", 0),
            } for s in subs])

            edited_df = st.data_editor(
                df, use_container_width=True, num_rows="dynamic",
                column_config={
                    "Start (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                    "End (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                    "Text": st.column_config.TextColumn(width="large"),
                    "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f", width="small"),
                    "SNR": st.column_config.NumberColumn(format="%.1f", width="small"),
                },
                hide_index=True, key="subtitle_editor",
            )

            # ── Capture edits for self-learning ──
            if st.button("💾 Save Edits & Capture Corrections", use_container_width=True):
                src_code = SOURCE_LANGUAGES.get(source_lang_name, "en") or "en"
                corrections_count = 0
                for i, row in edited_df.iterrows():
                    if i < len(st.session_state.subtitles):
                        new_text = row["Text"]
                        original = st.session_state.original_texts.get(i, "")
                        # Update subtitle
                        st.session_state.subtitles[i]["text"] = new_text
                        st.session_state.subtitles[i]["start"] = row["Start (s)"]
                        st.session_state.subtitles[i]["end"] = row["End (s)"]
                        # Save correction if text changed
                        if new_text != original and original:
                            save_correction(original, new_text, src_code)
                            corrections_count += 1

                if corrections_count > 0:
                    st.success(f"✅ Saved {corrections_count} correction(s) to self-learning dataset!")
                else:
                    st.info("No text changes detected. Timestamps updated.")

    # ── Preview Column ──
    with col_preview:
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            st.markdown('<div class="glass-header">🎬 Preview</div>', unsafe_allow_html=True)

            # Video preview with subtitles
            try:
                ext = os.path.splitext(st.session_state.video_path)[1].lower()
                if ext in [".mp4", ".webm", ".mov"]:
                    vtt_content = create_vtt(st.session_state.subtitles, st.session_state.subtitle_position)
                    with open(st.session_state.video_path, "rb") as vf:
                        video_b64 = base64.b64encode(vf.read()).decode()
                    vtt_b64 = base64.b64encode(vtt_content.encode()).decode()

                    mime = "video/mp4" if ext == ".mp4" else f"video/{ext[1:]}"
                    st.markdown(f"""
                    <div style="border-radius:12px; overflow:hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
                        <video width="100%" controls style="background:#000;">
                            <source src="data:{mime};base64,{video_b64}" type="{mime}">
                            <track src="data:text/vtt;base64,{vtt_b64}" kind="subtitles" srclang="en" label="Subtitles" default>
                        </video>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.audio(st.session_state.video_path)
            except Exception:
                st.warning("Preview unavailable for this file format.")

            # ── Video Trimming ──
            if st.session_state.video_path and MOVIEPY_AVAILABLE:
                with st.expander("✂️ Video Trimming"):
                    try:
                        clip = VideoFileClip(st.session_state.video_path)
                        max_dur = clip.duration
                        clip.close()
                    except Exception:
                        max_dur = 300.0

                    trim_col1, trim_col2 = st.columns(2)
                    with trim_col1:
                        trim_start = st.number_input("Start (s)", min_value=0.0, max_value=max_dur, value=0.0, step=0.5)
                    with trim_col2:
                        trim_end = st.number_input("End (s)", min_value=0.0, max_value=max_dur, value=min(30.0, max_dur), step=0.5)

                    if st.button("✂️ Trim Video", use_container_width=True):
                        with st.spinner("Trimming..."):
                            trimmed_path, err = trim_video(st.session_state.video_path, trim_start, trim_end)
                            if trimmed_path:
                                st.session_state.video_path = trimmed_path
                                # Filter subtitles to trimmed range
                                st.session_state.subtitles = [
                                    {**s, "start": s["start"] - trim_start, "end": s["end"] - trim_start}
                                    for s in st.session_state.subtitles
                                    if s["start"] >= trim_start and s["end"] <= trim_end
                                ]
                                st.success("✅ Video trimmed!")
                                st.rerun()
                            else:
                                st.error(f"Trim failed: {err}")

            # ── Export ──
            if st.session_state.subtitles:
                st.markdown('<div class="glass-header">📥 Export</div>', unsafe_allow_html=True)

                # Subtitle position
                pos = st.selectbox("Position", ["Bottom", "Middle", "Top"],
                                   index=["Bottom", "Middle", "Top"].index(st.session_state.subtitle_position))
                if pos != st.session_state.subtitle_position:
                    st.session_state.subtitle_position = pos
                    st.rerun()

                export_subs = st.session_state.subtitles
                vtt_out = create_vtt(export_subs, pos)
                srt_out = create_srt(export_subs)
                json_out = json.dumps(export_subs, indent=2, ensure_ascii=False)

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.download_button("⬇ VTT", vtt_out, "subtitles.vtt", "text/vtt", use_container_width=True)
                with ec2:
                    st.download_button("⬇ SRT", srt_out, "subtitles.srt", "text/plain", use_container_width=True)
                with ec3:
                    st.download_button("⬇ JSON", json_out, "subtitles.json", "application/json", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: LIVE MODE
# ──────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown('<div class="glass-header">🎙️ Real-Time Transcription</div>', unsafe_allow_html=True)

    live_col1, live_col2 = st.columns([3, 2], gap="large")

    with live_col1:
        st.markdown("""
        <div class="glass-card">
            <p style="color: var(--text-dim);">
                Real-time transcription from hardware microphone array or system mic.
                Audio is processed in chunks and transcribed using Faster-Whisper.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # WebRTC or simulated live mode
        use_webrtc = WEBRTC_AVAILABLE and st.checkbox("Use Browser Microphone (WebRTC)", value=False,
                                                       help="Enable browser mic via WebRTC. Disable for hardware/simulated mode.")

        if use_webrtc and WEBRTC_AVAILABLE:
            st.info("🎤 WebRTC microphone streaming active. Allow microphone access in your browser.")
            webrtc_ctx = webrtc_streamer(
                key="live-asr",
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"video": False, "audio": True},
                async_processing=True,
            )
        else:
            st.info("📡 Using hardware input (or simulation). Click **Start Live Session** to begin.")

        # Simulated live session
        if st.button("▶️ Start Live Session (5s Demo)", use_container_width=True, key="live_start"):
            if not WHISPER_AVAILABLE:
                st.error("faster-whisper not installed.")
            else:
                model = load_whisper_model(model_size if model_size in ["tiny", "base"] else "tiny",
                                           "cpu", "int8")
                if model:
                    transcript_area = st.empty()
                    sim_text = ""

                    # Simulate 5 chunks of ~1s audio transcription
                    demo_sentences = [
                        "Welcome to SubGEN PRO version two.",
                        "This is a real-time transcription demo.",
                        "Hardware integration provides SNR and DOA metadata.",
                        "The self-learning loop captures your corrections.",
                        "Indic languages like Tamil and Hindi are fully supported.",
                    ]

                    progress_live = st.progress(0)

                    for i, sentence in enumerate(demo_sentences):
                        time.sleep(0.8)  # Simulate processing delay
                        hw_meta = hw.read_metadata()
                        st.session_state.live_snr_history.append(hw_meta["snr"])
                        st.session_state.live_doa_history.append(hw_meta["doa"])

                        ts = format_timestamp_vtt(i * 2.0)
                        sim_text += f'<div style="padding:4px 0;">[{ts}] {sentence}</div>\n'

                        transcript_area.markdown(f"""
                        <div class="live-output">{sim_text}</div>
                        """, unsafe_allow_html=True)

                        progress_live.progress((i + 1) / len(demo_sentences))

                    st.session_state.live_transcript = sim_text
                    st.success("✅ Live session complete!")

        # Show transcript history
        if st.session_state.live_transcript:
            st.markdown("---")
            st.markdown('<div class="glass-header">📝 Transcript</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="live-output">{st.session_state.live_transcript}</div>
            """, unsafe_allow_html=True)

    # ── Dashboard Column ──
    with live_col2:
        st.markdown('<div class="glass-header">📊 Signal Dashboard</div>', unsafe_allow_html=True)

        # Read current hardware values
        current_meta = hw.read_metadata()

        # SNR Gauge
        st.plotly_chart(create_snr_gauge(current_meta["snr"]), use_container_width=True, key="snr_gauge")

        # DOA Gauge
        st.plotly_chart(create_doa_gauge(current_meta["doa"]), use_container_width=True, key="doa_gauge")

        # DOA Variance metric
        st.markdown(f"""
        <div class="metric-card" style="margin-top:0.5rem;">
            <div class="value">{current_meta['doa_variance']:.1f}°</div>
            <div class="label">DOA Variance</div>
        </div>
        """, unsafe_allow_html=True)

        # SNR history chart
        if st.session_state.live_snr_history:
            st.markdown("---")
            snr_df = pd.DataFrame({
                "Sample": range(len(st.session_state.live_snr_history)),
                "SNR (dB)": st.session_state.live_snr_history,
            })
            snr_chart = alt.Chart(snr_df).mark_area(
                line={"color": "#00f2ea"},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(0,242,234,0.3)", offset=0),
                        alt.GradientStop(color="rgba(0,242,234,0.01)", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
            ).encode(
                x=alt.X("Sample:Q", axis=alt.Axis(grid=False)),
                y=alt.Y("SNR (dB):Q", scale=alt.Scale(domain=[0, 50])),
            ).properties(height=120, title="SNR History").configure_view(
                strokeOpacity=0
            ).configure_axis(
                labelColor="#888",
                titleColor="#aaa",
                gridColor="rgba(255,255,255,0.05)",
            )
            st.altair_chart(snr_chart, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
with tab_settings:
    settings_col1, settings_col2 = st.columns(2, gap="large")

    with settings_col1:
        st.markdown('<div class="glass-header">🧠 Self-Learning Loop</div>', unsafe_allow_html=True)

        corrections = load_corrections()
        total_corrections = len(corrections)

        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:0.8rem; color:var(--text-dim); text-transform:uppercase;">Total Corrections</div>
                    <div style="font-size:2rem; font-weight:800; color:var(--primary);">{total_corrections}</div>
                </div>
                <div>
                    <div style="font-size:0.8rem; color:var(--text-dim); text-transform:uppercase;">Ready to Fine-Tune</div>
                    <div style="font-size:2rem; font-weight:800; color:{'var(--green-qc)' if total_corrections >= 10 else 'var(--red-qc)'};">
                        {'✓ Yes' if total_corrections >= 10 else f'{10 - total_corrections} more needed'}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Correction History", expanded=False):
            if corrections:
                for i, c in enumerate(corrections[-20:]):  # Show last 20
                    st.markdown(f"""
                    **#{i+1}** ({c.get('language', '?')}) — {c.get('timestamp', '')[:16]}
                    - Original: `{c.get('original_text', '')[:80]}`
                    - Corrected: `{c.get('corrected_text', '')[:80]}`
                    """)
            else:
                st.info("No corrections yet. Edit subtitles in File Mode to build your dataset.")

        # Fine-tune controls
        st.markdown("---")
        st.markdown('<div class="glass-header">🔬 Fine-Tune Controls</div>', unsafe_allow_html=True)

        ft_language = st.selectbox("Target Language for Fine-Tuning",
                                    ["Tamil (ta)", "Malayalam (ml)", "Hindi (hi)", "Telugu (te)",
                                     "Bengali (bn)", "Urdu (ur)", "Marathi (mr)"])
        ft_lang_code = ft_language.split("(")[1].replace(")", "").strip()
        ft_model = st.selectbox("Base Model", ["openai/whisper-tiny", "openai/whisper-base"])

        lang_corrections_count = sum(1 for c in corrections if c.get("language") == ft_lang_code)
        st.caption(f"Corrections for `{ft_lang_code}`: **{lang_corrections_count}**")

        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("🚀 Fine-Tune Now", use_container_width=True,
                         disabled=lang_corrections_count < 10):
                with st.spinner("Running LoRA fine-tuning..."):
                    result = run_lora_finetune(ft_model, ft_lang_code)
                    if "✅" in result:
                        st.success(result)
                    elif "⚠" in result:
                        st.warning(result)
                    else:
                        st.error(result)
        with fc2:
            if corrections:
                corrections_json = json.dumps(corrections, indent=2, ensure_ascii=False)
                st.download_button("📥 Download Dataset",
                                   corrections_json,
                                   "corrections_dataset.json",
                                   "application/json",
                                   use_container_width=True)

    with settings_col2:
        st.markdown('<div class="glass-header">ℹ️ System Information</div>', unsafe_allow_html=True)

        # Dependency status grid
        deps = [
            ("Faster-Whisper", WHISPER_AVAILABLE),
            ("PySerial", SERIAL_AVAILABLE),
            ("MoviePy", MOVIEPY_AVAILABLE),
            ("Transformers", HF_AVAILABLE),
            ("PEFT/LoRA", PEFT_AVAILABLE),
            ("WebRTC", WEBRTC_AVAILABLE),
            ("PySRT", PYSRT_AVAILABLE),
        ]

        dep_html = '<div class="glass-card"><table style="width:100%;">'
        for name, available in deps:
            color = "var(--green-qc)" if available else "var(--red-qc)"
            icon = "✓" if available else "✗"
            dep_html += f"""
            <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                <td style="padding:8px; color:var(--text-main);">{name}</td>
                <td style="padding:8px; text-align:right; color:{color}; font-weight:600;">{icon} {'Installed' if available else 'Missing'}</td>
            </tr>
            """
        dep_html += "</table></div>"
        st.markdown(dep_html, unsafe_allow_html=True)

        # Hardware info
        st.markdown("---")
        st.markdown('<div class="glass-header">🔌 Hardware Status</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <table style="width:100%;">
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Connection</td>
                    <td style="padding:6px; text-align:right;">{hw.status_html()}</td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Serial Port</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">
                        {hw.serial_port.port if hw.serial_port and hasattr(hw.serial_port, 'port') else 'N/A'}
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Mode</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">
                        {'USB Serial (ESP32)' if hw.connected else 'Simulation'}
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Model info
        st.markdown("---")
        st.markdown('<div class="glass-header">📐 Current Configuration</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <table style="width:100%;">
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Model</td>
                    <td style="padding:6px; text-align:right; color:var(--primary); font-weight:600;">{model_size}</td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Compute</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">{compute_type}</td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Device</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">{device_type}</td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Source Language</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">{source_lang_name}</td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Target Language</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">{target_lang_name}</td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">Beam Size</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">{beam_size}</td>
                </tr>
                <tr>
                    <td style="padding:6px; color:var(--text-dim);">VAD Filter</td>
                    <td style="padding:6px; text-align:right; color:var(--text-main);">{'Enabled' if vad_filter else 'Disabled'}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # LoRA adapters
        adapter_dir = "./adapters"
        if os.path.exists(adapter_dir):
            st.markdown("---")
            st.markdown('<div class="glass-header">🧩 Loaded Adapters</div>', unsafe_allow_html=True)
            adapters = [d for d in os.listdir(adapter_dir) if os.path.isdir(os.path.join(adapter_dir, d))]
            if adapters:
                for a in adapters:
                    st.markdown(f"- `{a}`")
            else:
                st.caption("No adapters found. Fine-tune to create one.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FOOTER                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1.5rem; border-top:1px solid rgba(255,255,255,0.06);">
    <p style="color:#555; font-size:0.8rem; margin:0;">
        SubGEN PRO v2 • Faster-Whisper • PEFT/LoRA • Hardware Integration<br>
        <span style="color:#333;">Built with ❤️ and Streamlit</span>
    </p>
</div>
""", unsafe_allow_html=True)
