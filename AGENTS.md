# SubGEN PRO v2: AI Agent Context

This file provides context for AI assistants and developers working on the **SubGEN PRO v2** project.

## Project Overview
**SubGEN PRO v2** is a high-performance, AI-powered subtitle generation application built with Streamlit. It supports **file-based** media processing, **real-time** transcription, **hardware sensor integration** (ESP32 beamformed audio), **signal-informed QC**, and a **self-learning loop** for fine-tuning on user corrections via PEFT/LoRA.

## Tech Stack
- **UI Framework**: Streamlit (wide layout, glassmorphism dark theme)
- **Transcription**: `faster-whisper` (Whisper ASR with CTranslate2)
- **Translation**:
  - **Whisper Native**: Any language → English (`task="translate"`)
  - **Helsinki-NLP/opus-mt**: English → other languages via Hugging Face `transformers`
- **Hardware Integration**: `pyserial` for ESP32 USB serial (SNR, DOA, DOA_variance metadata)
- **Signal QC**: Fused confidence = `0.6*ASR + 0.3*(1-SNR_penalty) + 0.1*speaker_stability`
- **Self-Learning**: PEFT/LoRA fine-tuning on user corrections (stored as JSON dataset)
- **Video Editing**: `moviepy` for trimming
- **Visualization**: Plotly gauges (SNR/DOA), Altair charts (confidence distribution)
- **Audio Processing**: `ffmpeg-python`
- **Real-Time**: `streamlit-webrtc` (optional browser mic)

## Key Files
- [SubNXT.py](file:///c:/Users/acer/Documents/projects/SubGEN_streamlit/SubNXT.py): Main application (single-file architecture for Streamlit deployment)
- [subgen_pro_v2.py](file:///c:/Users/acer/Documents/projects/SubGEN_streamlit/subgen_pro_v2.py): Backup/Development file
- [requirements.txt](file:///c:/Users/acer/Documents/projects/SubGEN_streamlit/requirements.txt): Python dependencies
- [packages.txt](file:///c:/Users/acer/Documents/projects/SubGEN_streamlit/packages.txt): System-level dependencies (`ffmpeg`)
- [corrections.json](file:///c:/Users/acer/Documents/projects/SubGEN_streamlit/corrections.json): Self-learning dataset (auto-created)

## Running the App
```bash
streamlit run subgen_pro_v2.py
```

## Architecture
The app is structured as tabs:
1. **File Mode**: Upload → Transcribe → Edit (with QC badges) → Export (VTT/SRT/JSON)
2. **Live Mode**: Real-time transcription with SNR/DOA dashboard gauges
3. **Settings**: Self-learning controls, fine-tuning, system info, dependency status

## Translation Logic
1. If Target == Source (or "Same as Source") → `whisper(task="transcribe")`
2. If Target == English → `whisper(task="translate")`
3. Otherwise → `whisper(task="transcribe")` → `Helsinki-NLP/opus-mt(text)`

## Indic Language Support
Full support for: Tamil (ta), Malayalam (ml), Hindi (hi), Telugu (te), Bengali (bn), Urdu (ur), Marathi (mr). Low-confidence Indic segments prompt user corrections which feed the self-learning loop.

## Hardware Integration
- **ESP32**: Beamformed audio + JSON metadata over USB serial (115200 baud)
- **Fallback**: Simulated SNR/DOA data via numpy random
- **Detection**: Auto-scans for USB/CP210/CH340 serial ports
