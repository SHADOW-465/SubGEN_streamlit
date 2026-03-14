import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from subgen_ai.core.esp32_validator import (
    compute_mfcc_software, get_fingerprint, find_esp32_port,
    N_MFCC, SAMPLE_RATE
)


def test_mfcc_software_empty_audio():
    """Empty audio returns ok=False."""
    result = compute_mfcc_software(np.array([], dtype=np.float32))
    assert result["ok"] is False
    assert result["hw"] is False


def test_mfcc_software_sine_wave():
    """1 second sine wave at 440 Hz → ok=True, 12 coefficients."""
    t = np.linspace(0, 1, SAMPLE_RATE)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = compute_mfcc_software(audio)
    assert result["ok"] is True
    assert result["hw"] is False
    assert len(result["mfcc_mean"]) == N_MFCC
    assert len(result["mfcc_var"]) == N_MFCC
    assert result["frames"] > 0
    assert isinstance(result["rms"], float)


def test_mfcc_software_int16_normalisation():
    """int16-range audio (> 1.0 max) is normalised before processing."""
    audio = np.ones(SAMPLE_RATE, dtype=np.float32) * 10000.0  # int16 range
    result = compute_mfcc_software(audio)
    assert result["ok"] is True


def test_mfcc_mean_length():
    """mfcc_mean must always have exactly N_MFCC=12 elements."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
    result = compute_mfcc_software(audio)
    if result["ok"]:
        assert len(result["mfcc_mean"]) == 12
        assert len(result["mfcc_var"]) == 12


def test_get_fingerprint_no_port_uses_software():
    """With no ESP32 port, get_fingerprint falls back to software."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
    result = get_fingerprint(audio, SAMPLE_RATE, esp32_port=None)
    assert result["hw"] is False


def test_find_esp32_port_returns_str_or_none():
    """find_esp32_port must return a string or None."""
    result = find_esp32_port()
    assert result is None or isinstance(result, str)
