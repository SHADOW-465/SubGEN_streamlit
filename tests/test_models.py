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
