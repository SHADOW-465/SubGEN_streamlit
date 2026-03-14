import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from subgen_ai.core.models import SubtitleSegment, CorrectionRecord, ValidationResult


def test_subtitle_segment_defaults():
    seg = SubtitleSegment(
        index=0, start=0.0, end=2.0, text="hello", language="en",
        asr_conf=0.9, snr_db=20.0, fused_conf=0.85, label="GREEN"
    )
    assert seg.mfcc_mean == []
    assert seg.mfcc_var == []
    assert seg.hw_fingerprint is False
    assert seg.corrected is False
    assert seg.correction_text == ""


def test_correction_record_fields():
    rec = CorrectionRecord(
        id=None, segment_start=1.0, segment_end=3.0,
        original_text="wrong", corrected_text="right", language="ta",
        mfcc_mean=[0.1]*12, mfcc_var=[0.01]*12,
        match_score=0.85, hw_used=False, created_at="2026-01-01T00:00:00"
    )
    assert len(rec.mfcc_mean) == 12
    assert rec.hw_used is False


def test_validation_result_fields():
    vr = ValidationResult(score=0.9, tier="HIGH", accepted=True,
                          hw_used=False, message="OK")
    assert vr.accepted is True
    assert vr.tier == "HIGH"


def test_transcriber_imports():
    """Transcriber module must import without errors (lazy model loading)."""
    from subgen_ai.core import transcriber
    assert hasattr(transcriber, "transcribe")
    assert hasattr(transcriber, "extract_clip")
    assert hasattr(transcriber, "load_model")


def test_extract_clip_bounds():
    """extract_clip must clamp to array boundaries."""
    import numpy as np
    from subgen_ai.core.transcriber import extract_clip
    audio = np.ones(16000, dtype=np.float32)
    clip = extract_clip(audio, 16000, start_s=-1.0, end_s=100.0)
    assert len(clip) == len(audio)   # clamped to full array
