import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from subgen_ai.core.qc_engine import (
    compute_asr_conf, compute_snr_penalty, compute_fused_conf,
    label_segment, cosine_similarity, euclidean_similarity,
    compute_match_score, FUSED_CONF_THRESHOLD,
    THRESHOLD_HIGH, THRESHOLD_MEDIUM
)


def test_asr_conf_logprob_zero():
    """exp(0) == 1.0"""
    assert compute_asr_conf(0.0) == 1.0


def test_asr_conf_negative():
    """exp(-0.5) ≈ 0.6065"""
    result = compute_asr_conf(-0.5)
    assert abs(result - 0.6065) < 0.001


def test_snr_penalty_all_silence():
    """All-zero audio → no speech windows → penalty = 1.0"""
    silence = np.zeros(16000, dtype=np.float32)
    assert compute_snr_penalty(silence) == 1.0


def test_snr_penalty_pure_speech():
    """Loud sine wave → all speech windows → penalty = 0.0"""
    t = np.linspace(0, 1, 16000)
    speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert compute_snr_penalty(speech) == 0.0


def test_fused_conf_range():
    """fused_conf must be in [0, 1] for any reasonable inputs."""
    fc = compute_fused_conf(0.8, 0.2, 1.0)
    assert 0.0 <= fc <= 1.0


def test_fused_conf_formula():
    """0.6*0.8 + 0.3*(1-0.2) + 0.1*1.0 = 0.48 + 0.24 + 0.10 = 0.82"""
    fc = compute_fused_conf(0.8, 0.2, 1.0)
    assert abs(fc - 0.82) < 1e-6


def test_label_green():
    assert label_segment(FUSED_CONF_THRESHOLD) == "GREEN"
    assert label_segment(1.0) == "GREEN"


def test_label_red():
    assert label_segment(FUSED_CONF_THRESHOLD - 0.01) == "RED"
    assert label_segment(0.0) == "RED"


def test_cosine_similarity_identical():
    """Identical vectors → 1.0"""
    v = [1.0, 2.0, 3.0] + [0.0] * 9
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_zero_vector():
    """Zero vector → 0.0"""
    z = [0.0] * 12
    assert cosine_similarity(z, z) == 0.0


def test_compute_match_score_identical():
    """Identical fingerprints → match score close to 1.0"""
    v = [1.0] * 12
    score = compute_match_score(v, v)
    assert score > 0.99


def test_threshold_constants():
    assert THRESHOLD_HIGH > THRESHOLD_MEDIUM
    assert THRESHOLD_MEDIUM > 0
