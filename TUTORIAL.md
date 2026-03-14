# SubGEN AI — Step-by-Step Tutorial Guide

> **Project:** SubGEN AI | PSVPEC ECE | Batch A25
> **Authors:** R. Showmik Kumaar & Niraikulanathan P
> **Guide:** Ms. G. Kalanandhini, Asst. Prof. ECE

---

## Table of Contents

1. [What is SubGEN AI?](#1-what-is-subgen-ai)
2. [What You Need Before Starting](#2-what-you-need-before-starting)
3. [Step 1 — Install Python](#step-1--install-python)
4. [Step 2 — Install ffmpeg](#step-2--install-ffmpeg)
5. [Step 3 — Download the Project](#step-3--download-the-project)
6. [Step 4 — Create a Virtual Environment](#step-4--create-a-virtual-environment)
7. [Step 5 — Install Python Dependencies](#step-5--install-python-dependencies)
8. [Step 6 — Run the App](#step-6--run-the-app)
9. [Step 7 — Using the App (Feature Walkthrough)](#step-7--using-the-app-feature-walkthrough)
   - [7.1 Upload & Transcribe](#71-upload--transcribe)
   - [7.2 Review & Edit](#72-review--edit)
   - [7.3 Export](#73-export)
   - [7.4 The Sidebar Controls](#74-the-sidebar-controls)
10. [Step 8 — (Optional) ESP32 Hardware Setup](#step-8--optional-esp32-hardware-setup)
11. [Understanding RED and GREEN Labels](#understanding-red-and-green-labels)
12. [Understanding the Correction System (Self-Learning)](#understanding-the-correction-system-self-learning)
13. [Supported Languages](#supported-languages)
14. [Troubleshooting](#troubleshooting)
15. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## 1. What is SubGEN AI?

SubGEN AI is a **completely offline** subtitle generator. You give it a video or audio file, and it uses the Faster-Whisper AI model to produce subtitles. It then grades each subtitle segment RED or GREEN based on how confident it is, lets you correct mistakes, and learns from those corrections so it gets better over time.

**Key features:**
- Works 100% on your PC — no internet, no cloud, no API keys needed
- Supports Tamil, Telugu, Hindi, Malayalam, and many more languages
- Grades each subtitle with a confidence score (GREEN = confident, RED = needs review)
- Saves corrections in a local database and auto-applies them next time
- Optional: connects to an ESP32 microcontroller for hardware-accelerated audio fingerprinting
- Exports to `.srt`, `.vtt`, and `.json` formats

---

## 2. What You Need Before Starting

| Requirement | Minimum Version | Where to Get It |
|---|---|---|
| Windows / macOS / Linux | Any modern version | — |
| Python | 3.10 or newer | https://www.python.org/downloads/ |
| ffmpeg | Any recent version | See Step 2 |
| ~2 GB free disk space | — | For Whisper model download |
| Internet (one-time only) | — | To install packages + download model |

> ⚠ **Internet is only needed once** — to install packages and download the Whisper model the first time you run it. After that, everything runs offline.

---

## Step 1 — Install Python

### Windows
1. Go to https://www.python.org/downloads/
2. Click **"Download Python 3.12.x"** (or any 3.10+ version)
3. Run the installer
4. **IMPORTANT:** On the first screen, tick ✅ **"Add python.exe to PATH"**
5. Click **Install Now**
6. When done, open **Command Prompt** and verify:
   ```
   python --version
   ```
   You should see something like `Python 3.12.3`

### macOS
```bash
# Install Homebrew first if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install Python:
brew install python@3.12

# Verify:
python3 --version
```

### Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
python3 --version
```

---

## Step 2 — Install ffmpeg

ffmpeg is a free tool that SubGEN AI uses to extract audio from video files. You must install it separately.

### Windows

**Option A — winget (recommended, built into Windows 10/11):**
```
winget install ffmpeg
```
Then close and reopen Command Prompt.

**Option B — Manual:**
1. Go to https://www.gyan.dev/ffmpeg/builds/
2. Download `ffmpeg-release-essentials.zip`
3. Extract it to `C:\ffmpeg`
4. Open **Start → Search → "Environment Variables"**
5. Click **"Environment Variables"**
6. Under **System Variables**, find `Path` → click **Edit**
7. Click **New** → type `C:\ffmpeg\bin`
8. Click **OK** on all dialogs
9. Restart Command Prompt and verify:
   ```
   ffmpeg -version
   ```

### macOS
```bash
brew install ffmpeg
ffmpeg -version   # verify
```

### Linux (Ubuntu / Debian)
```bash
sudo apt install ffmpeg
ffmpeg -version   # verify
```

---

## Step 3 — Download the Project

### Option A — Using Git (recommended)
```bash
git clone https://github.com/SHADOW-465/SubGEN_streamlit.git
cd SubGEN_streamlit
```

### Option B — Download ZIP
1. Go to https://github.com/SHADOW-465/SubGEN_streamlit
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP somewhere (e.g., `C:\Projects\SubGEN_streamlit`)
4. Open Command Prompt and navigate there:
   ```
   cd C:\Projects\SubGEN_streamlit
   ```

---

## Step 4 — Create a Virtual Environment

A virtual environment keeps SubGEN AI's packages separate from the rest of your system — this avoids conflicts.

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
You'll know it's active when you see `(venv)` at the start of your prompt.

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 **Every time you open a new terminal to use SubGEN AI, you need to activate the venv again** using the same command above.

---

## Step 5 — Install Python Dependencies

Make sure your virtual environment is active (you see `(venv)` in the prompt), then run:

```bash
pip install -r subgen_ai/requirements.txt
```

This will install:
- `streamlit` — the web UI framework
- `faster-whisper` — the AI speech recognition engine
- `pyserial` — for ESP32 communication
- `numpy`, `scipy` — for audio processing
- `ffmpeg-python` — Python wrapper for ffmpeg
- `pandas` — for displaying DB tables

**Expected output:** Lots of download messages ending with `Successfully installed ...`

> ⏳ This step may take 3–10 minutes depending on your internet speed. Be patient.

---

## Step 6 — Run the App

Make sure:
- ✅ Virtual environment is **active** (`(venv)` in prompt)
- ✅ You are in the `SubGEN_streamlit` project folder

Then run:

```bash
streamlit run subgen_ai/app.py
```

You should see output like:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser should open automatically. If not, open your browser and go to:
**http://localhost:8501**

> 🔑 **First run only:** When you first click "Generate Subtitles", the app will download the Whisper model (e.g., `small` = ~480 MB). This is automatic and only happens once. Subsequent runs use the cached model.

---

## Step 7 — Using the App (Feature Walkthrough)

The app has three tabs across the top. Work through them left to right.

---

### 7.1 Upload & Transcribe

This is where you upload your video and generate subtitles.

**Step-by-step:**

1. **Open the app** at http://localhost:8501
2. Click on the **"📁 Upload & Transcribe"** tab (it's the default)
3. Click **"Browse files"** and select your video or audio file
   - Supported: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wav`, `.mp3`, `.m4a`
4. The filename and size appear below the uploader
5. (Optional) Change model/language settings in the **left sidebar** before transcribing — see [Section 7.4](#74-the-sidebar-controls)
6. Click the big **"▶ Generate Subtitles"** button
7. Watch the progress bar — it shows the current segment being processed
8. When done, the tab will show **"✅ Transcription done — N segments — lang: XX"**
9. The app switches you to the **Review & Edit** tab automatically on the next page refresh

> ⏳ **How long does it take?**
> - `tiny` model: ~1× realtime (a 10-min video takes ~10 min)
> - `small` model: ~2–3× realtime
> - `medium`/`large` models: slower but more accurate
> For most use cases, **`small`** gives a good balance.

---

### 7.2 Review & Edit

This is where you check and correct the subtitles.

**What you'll see at the top:**

| Metric | Meaning |
|---|---|
| Total segments | How many subtitle lines were detected |
| 🔴 RED (review needed) | Segments the AI is unsure about |
| 🟢 GREEN (confident) | Segments the AI is confident about |
| ✅ Corrections saved | How many you've corrected this session |

**How to review segments:**

1. Each subtitle appears as a collapsible card showing:
   - Time range: `[00:01:23 → 00:01:27]`
   - Confidence badge: 🔴 or 🟢
   - The `fused_conf` score (0.0–1.0, higher = better)
2. Click any card to expand it and see the full text
3. At the bottom of each card:
   - **ASR conf** — raw AI confidence
   - **SNR** — audio signal quality in decibels
   - **HW-MFCC / SW-MFCC** — whether hardware (ESP32) or software computed the fingerprint

**Correcting a RED segment:**

1. Expand the 🔴 RED segment
2. The text area is **editable** — type the correct transcript
3. Click **"💾 Validate & Save Correction"**
4. The system compares the audio fingerprint of the original recording to confirm this is the right segment
5. You'll see one of three results:
   - ✅ **HIGH** — great match, correction saved
   - ⚠ **MEDIUM** — acceptable match, correction saved but flagged
   - ❌ **MISMATCH** — audio doesn't match; you can still click **"Save anyway (override)"** if you're sure

**Tip:** Use the **"🔴 Show RED segments only"** toggle to focus only on segments that need attention.

---

### 7.3 Export

Once you're happy with the subtitles, download them here.

1. Click the **"📥 Export"** tab
2. You'll see three download buttons:
   - **⬇ Download .srt** — Standard subtitle format, works with VLC, YouTube, etc.
   - **⬇ Download .vtt** — WebVTT format, used on websites and HTML5 video
   - **⬇ Download .json (with QC metadata)** — Full data including confidence scores, useful for further processing
3. Click any button — the file downloads immediately
4. Below the buttons, a **preview** of the SRT content is shown

**Where are the downloaded files?** Check your browser's default Downloads folder.

---

### 7.4 The Sidebar Controls

The left sidebar lets you configure everything before transcribing.

#### ⚙ Hardware Section
- **ESP32 Port** — If you have an ESP32 connected via USB, it auto-detects and appears here. Otherwise select **"Software only"** (default)
- **Status indicator:**
  - 🟢 Connected — ESP32 found and active
  - 🔴 Not found — ESP32 expected but not responding
  - 🔵 Software mode — Running without hardware (this is fine!)
- **🔌 Test Connection** — Sends a dummy signal to the ESP32 and shows the response time

#### 🧠 AI Settings
- **Whisper Model** — Choose the AI model size:

  | Model | Size | Speed | Accuracy |
  |---|---|---|---|
  | `tiny` | ~75 MB | Fastest | Basic |
  | `base` | ~145 MB | Fast | OK |
  | `small` | ~480 MB | Medium | **Good (recommended)** |
  | `medium` | ~1.5 GB | Slow | Better |
  | `large-v2` | ~3 GB | Very slow | Best |
  | `large-v3` | ~3 GB | Very slow | Best |

- **Task:**
  - **Transcribe** — Keep the original language
  - **Translate to English** — Automatically translate any language to English subtitles

- **Language:**
  - **Auto-detect** — Let the AI figure it out (recommended unless you know the language)
  - Or pick from: `ta` (Tamil), `te` (Telugu), `hi` (Hindi), `ml` (Malayalam), `en` (English), and more

#### 📊 Correction DB
- Shows how many corrections are stored in your local database
- Shows a breakdown by language
- **🗑 Clear all corrections** — Deletes the database (use with caution)

---

## Step 8 — (Optional) ESP32 Hardware Setup

The ESP32 is an optional microcontroller that computes audio fingerprints in hardware (faster and more power-efficient). The app works perfectly without it using software fallback.

### What you need
- ESP32 DevKit v1 board (~₹400–600 on Amazon/Robu)
- USB cable (Micro-USB or USB-C depending on your board)
- Arduino IDE 2.x (https://www.arduino.cc/en/software)
- ArduinoJson library v6

### Flashing the firmware

1. **Install Arduino IDE** from https://www.arduino.cc/en/software

2. **Install ESP32 board support:**
   - Open Arduino IDE → **File → Preferences**
   - In "Additional boards manager URLs" add:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Go to **Tools → Board → Boards Manager**
   - Search for `esp32` → Install **"esp32 by Espressif Systems"**

3. **Install ArduinoJson library:**
   - Go to **Tools → Manage Libraries**
   - Search for `ArduinoJson` → Install **version 6.x** by Benoît Blanchon

4. **Open the firmware:**
   - In this project, find `subgen_ai/firmware/esp32_firmware.ino`
   - Open it in Arduino IDE

5. **Configure board settings:**
   - **Tools → Board → ESP32 Arduino → ESP32 Dev Module**
   - **Tools → Port → COM3** (or whatever port your ESP32 is on — check Device Manager on Windows)
   - **Tools → Upload Speed → 460800**

6. **Flash it:**
   - Click the **→ Upload** button
   - Wait for "Done uploading."

7. **Connect to SubGEN AI:**
   - Keep the ESP32 plugged in via USB
   - Restart the SubGEN AI app (`streamlit run subgen_ai/app.py`)
   - The sidebar should now show the COM port in the dropdown and status 🟢 Connected

---

## Understanding RED and GREEN Labels

Every subtitle segment gets a **fused confidence score** between 0.0 and 1.0:

```
fused_conf = 0.6 × ASR_confidence
           + 0.3 × Signal_quality
           + 0.1 × Speaker_stability
```

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.75 | 🟢 GREEN | The AI is confident — likely correct |
| < 0.75 | 🔴 RED | The AI is unsure — please review |

**What affects the score?**
- **ASR confidence** — how sure Whisper was about the words (based on its internal probability)
- **Signal quality (SNR)** — was there background noise? Low SNR = more penalty
- **Speaker stability** — consistency across the segment (currently fixed at 1.0)

**Tip:** In very noisy recordings, many segments may be RED even if the words are correct. That's normal — just correct the wrong ones.

---

## Understanding the Correction System (Self-Learning)

SubGEN AI has a built-in learning loop. Here's how it works:

### Saving a correction
When you correct a RED segment and click **Validate & Save**:
1. The system computes an **audio fingerprint** (MFCC — Mel-frequency cepstral coefficients) of that exact audio moment
2. It stores the fingerprint + your corrected text in a local SQLite database at:
   - `C:\Users\YourName\.subgen_ai\corrections.db` (Windows)
   - `/home/yourname/.subgen_ai/corrections.db` (Linux/macOS)

### Auto-applying corrections next time
When you transcribe a **new video** that contains similar audio:
1. The system computes MFCC fingerprints for each segment
2. It searches the database for stored fingerprints with ≥ 80% cosine similarity
3. If a match is found, it **automatically applies your saved correction** before you even see the segment
4. Auto-corrected segments show a **🔄 Auto-corrected from DB** badge

**This means:** the more you correct, the smarter the app gets for your specific content (speaker, accent, vocabulary).

### The correction database
The database is stored **only on your PC** — it never uploads anywhere. You can:
- View stats in the sidebar (total corrections, by language)
- Clear it with the 🗑 button if needed

---

## Supported Languages

| Code | Language |
|---|---|
| `ta` | Tamil (தமிழ்) |
| `te` | Telugu (తెలుగు) |
| `hi` | Hindi (हिन्दी) |
| `ml` | Malayalam (മലയാളം) |
| `kn` | Kannada (ಕನ್ನಡ) |
| `mr` | Marathi (मराठी) |
| `en` | English |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `ar` | Arabic |
| … | 90+ more via Auto-detect |

> 💡 For **Indic scripts** (Tamil, Telugu, etc.), use the `small` or `medium` model for best results. The `tiny` model sometimes struggles with Indic languages.

---

## Troubleshooting

### ❌ "streamlit: command not found" or "streamlit is not recognized"
**Cause:** Virtual environment is not activated, or streamlit wasn't installed.
**Fix:**
```bash
# Activate venv first:
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Then re-install:
pip install streamlit
```

---

### ❌ "ffmpeg: command not found" or ffmpeg error during transcription
**Cause:** ffmpeg binary is not installed or not in PATH.
**Fix:**
1. Install ffmpeg (see Step 2)
2. Close and reopen your terminal
3. Test: `ffmpeg -version`

---

### ❌ App opens but clicking "Generate Subtitles" does nothing or crashes
**Cause 1:** File format not supported — check it's one of: mp4, avi, mov, mkv, wav, mp3, m4a
**Cause 2:** ffmpeg not found
**Cause 3:** Not enough RAM for the model size (try `tiny` or `base` instead)
Check the terminal window where you ran `streamlit run` — the error message will be there.

---

### ❌ Whisper model download hangs or fails
**Cause:** Slow internet or firewall blocking downloads.
**Fix:** Models download from HuggingFace. Try:
```bash
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-small
```
Or connect to a different network.

---

### ❌ All segments are RED even for clear audio
**Cause:** This can happen with the `tiny` model on Indic languages, or with audio that has consistent background noise.
**Fix:** Switch to `small` or `medium` model in the sidebar. Also check that the audio file has clear speech.

---

### ❌ ESP32 not detected in sidebar
**Cause 1:** Wrong USB cable (some cables are power-only, not data)
**Cause 2:** Driver not installed (CP2102 or CH340 driver needed on Windows)
**Fix:**
- Try a different USB cable
- Install the CP2102 driver: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
- Or CH340 driver: https://sparks.gogo.co.nz/ch340.html
- Check Device Manager (Windows) to confirm the COM port appears

---

### ❌ SRT file has wrong encoding (garbled Tamil/Telugu text)
**Fix:** Open the SRT file in **Notepad++** or **VS Code** and make sure encoding is set to **UTF-8**. VLC and most modern players support UTF-8 SRT files natively.

---

### ❌ "No module named 'subgen_ai'" error
**Cause:** Running the app from the wrong directory.
**Fix:** Make sure you're in the `SubGEN_streamlit` root folder (not inside `subgen_ai/`):
```bash
cd SubGEN_streamlit       # correct
streamlit run subgen_ai/app.py
```

---

## Quick Reference Cheat Sheet

```
EVERY SESSION:
  1. Open terminal
  2. cd SubGEN_streamlit
  3. venv\Scripts\activate         (Windows)
     source venv/bin/activate      (macOS/Linux)
  4. streamlit run subgen_ai/app.py
  5. Open http://localhost:8501

TRANSCRIBE A VIDEO:
  → Tab 1: Upload file → click ▶ Generate Subtitles

REVIEW SUBTITLES:
  → Tab 2: Expand RED segments → edit text → Validate & Save

EXPORT:
  → Tab 3: Download .srt / .vtt / .json

CHANGE AI MODEL:
  → Sidebar → 🧠 AI Settings → Whisper Model

CHANGE LANGUAGE:
  → Sidebar → 🧠 AI Settings → Language

CHECK DB:
  → Sidebar → 📊 Correction DB

STOP THE APP:
  → Press Ctrl+C in the terminal

DEACTIVATE VENV:
  → deactivate
```

---

*SubGEN AI | PSVPEC ECE Department | Batch A25*
*R. Showmik Kumaar & Niraikulanathan P | Guide: Ms. G. Kalanandhini*
