"""
SubGEN AI — ESP32 Hardware Validator + Software MFCC Fallback.

Implements:
  - compute_mfcc_software(): Python-side MFCC identical to ESP32 C firmware algorithm
  - find_esp32_port(): auto-detect ESP32 USB serial port
  - send_audio_to_esp32(): PCM transfer and JSON response parsing
  - get_fingerprint(): unified entry point; tries HW first, falls back to SW
"""
import json
import struct
from typing import Optional

import numpy as np
from scipy.fft import dct

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

# MFCC Parameters (must match ESP32 firmware)
N_MFCC      = 12
N_MELS      = 26
SAMPLE_RATE = 16000
N_FFT       = 512
HOP_LENGTH  = 160    # 10 ms
WIN_LENGTH  = 400    # 25 ms
FMIN        = 0.0
FMAX        = 8000.0

# Serial Protocol
BAUD_RATE = 460800
HEADER    = bytes([0xAA, 0x55])
TIMEOUT_S = 3.0

# Cached Mel Filterbank
_MEL_FILTERBANK: Optional[np.ndarray] = None


def hz_to_mel(hz: float) -> float:
    """Convert Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(n_mels: int, n_fft: int,
                          sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Build (n_mels, n_fft//2 + 1) triangular filterbank on mel scale."""
    mel_min    = hz_to_mel(fmin)
    mel_max    = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filterbank


def get_filterbank() -> np.ndarray:
    """Return cached mel filterbank (built on first call)."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = build_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX)
    return _MEL_FILTERBANK


def compute_mfcc_software(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Compute MFCC fingerprint using the same algorithm as the ESP32 firmware.

    Steps: Hanning window → 512-pt FFT → power spectrum → 26-band mel filterbank
           → log compression → DCT-II → 12 MFCC coefficients per frame
           → mean and variance across all frames.

    Args:
        audio: float32 array, normalised [-1, 1] or int16 range.
        sr: sample rate (default 16000 Hz).

    Returns:
        dict with keys: ok (bool), hw (bool), mfcc_mean (list[float]),
                        mfcc_var (list[float]), rms (float), frames (int).
    """
    if len(audio) == 0:
        return {"ok": False, "hw": False,
                "mfcc_mean": [0.0] * N_MFCC, "mfcc_var": [0.0] * N_MFCC,
                "rms": 0.0, "frames": 0}

    # Normalise: detect int16 range
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    rms        = float(np.sqrt(np.mean(audio ** 2)))
    filterbank = get_filterbank()
    coefficients_per_frame: list = []

    for start in range(0, len(audio) - WIN_LENGTH, HOP_LENGTH):
        frame = audio[start:start + WIN_LENGTH]

        # Hanning window (matches ESP32 firmware)
        window          = np.hanning(len(frame))
        frame_windowed  = frame * window

        # Zero-pad to N_FFT
        padded          = np.zeros(N_FFT, dtype=np.float32)
        padded[:len(frame_windowed)] = frame_windowed

        # FFT power spectrum
        spectrum = np.fft.rfft(padded)
        power    = (np.abs(spectrum) ** 2) / N_FFT

        # Mel filterbank energies
        mel_energies = np.dot(filterbank, power)

        # Log compression
        log_mel = np.log10(mel_energies + 1e-9)

        # DCT-II → first 12 coefficients
        cepstrum = dct(log_mel, type=2, norm='ortho')
        coefficients_per_frame.append(cepstrum[:N_MFCC])

    if not coefficients_per_frame:
        return {"ok": False, "hw": False,
                "mfcc_mean": [0.0] * N_MFCC, "mfcc_var": [0.0] * N_MFCC,
                "rms": rms, "frames": 0}

    frames_arr = np.array(coefficients_per_frame)
    mfcc_mean  = frames_arr.mean(axis=0).tolist()
    mfcc_var   = frames_arr.var(axis=0).tolist()

    return {
        "ok": True, "hw": False,
        "mfcc_mean": mfcc_mean, "mfcc_var": mfcc_var,
        "rms": rms, "frames": len(coefficients_per_frame)
    }


def find_esp32_port() -> Optional[str]:
    """
    Auto-detect ESP32 by checking for CP2102 or CH340 USB descriptors.
    Returns the device path (e.g. 'COM3' or '/dev/ttyUSB0') or None.
    """
    if not _SERIAL_AVAILABLE:
        return None
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        if any(k in desc for k in ["cp210", "ch340", "esp32", "uart"]):
            return port.device
    return None


def send_audio_to_esp32(audio_int16: np.ndarray, port: str) -> dict:
    """
    Send PCM int16 audio to ESP32 over USB serial.

    Protocol: [0xAA][0x55][N_high][N_low][PCM int16 LE bytes...]
    ESP32 responds with a single JSON line containing the MFCC fingerprint.

    Caps input at 32000 samples (2 seconds at 16 kHz).

    Returns parsed fingerprint dict or an error dict with ok=False.
    """
    if not _SERIAL_AVAILABLE:
        return {"ok": False, "hw": False, "error": "pyserial not installed"}

    n = len(audio_int16)
    if n > 32000:
        audio_int16 = audio_int16[:32000]
        n = 32000

    payload = HEADER + struct.pack(">H", n) + audio_int16.tobytes()

    try:
        with serial.Serial(port, BAUD_RATE, timeout=TIMEOUT_S) as ser:
            ser.reset_input_buffer()
            ser.write(payload)
            response_bytes = ser.readline()   # ESP32 sends one JSON line
        response_str = response_bytes.decode("utf-8", errors="replace").strip()
        result = json.loads(response_str)
        result["hw"] = True
        return result
    except Exception as e:
        return {"ok": False, "hw": False, "error": str(e)}


def get_fingerprint(audio: np.ndarray, sr: int,
                    esp32_port: Optional[str] = None) -> dict:
    """
    Obtain MFCC fingerprint for an audio clip.

    Tries ESP32 hardware first (if port provided), falls back to software.

    Args:
        audio:      float32 array normalised [-1, 1] at sr Hz.
        sr:         sample rate.
        esp32_port: serial device path or None for software-only mode.

    Returns:
        Standard fingerprint dict: ok, hw, mfcc_mean, mfcc_var, rms, frames.
    """
    if esp32_port is not None:
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        result = send_audio_to_esp32(audio_int16, esp32_port)
        if result.get("ok"):
            return result

    return compute_mfcc_software(audio, sr)
