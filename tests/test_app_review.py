# tests/test_app_review.py
"""
Tests for the Review tab — specifically that GREEN segments can be saved
through the same _do_save_correction path as RED ones.
"""
from subgen_ai.core.models import SubtitleSegment, ValidationResult


def _green_seg() -> SubtitleSegment:
    return SubtitleSegment(
        index=0, start=0.0, end=1.0, text="Original text", language="en",
        asr_conf=0.95, snr_db=25.0, fused_conf=0.95, label="GREEN",
        mfcc_mean=[], mfcc_var=[], hw_fingerprint=False,
        corrected=False, correction_text="",
    )


def test_do_save_correction_works_for_green_segment(monkeypatch):
    """_do_save_correction must work regardless of segment label."""
    import streamlit as st
    from subgen_ai import app   # safe to import — conftest stubs set_page_config

    seg = _green_seg()

    # Patch session_state with the minimum keys _do_save_correction touches
    monkeypatch.setattr(
        st, "session_state",
        {"segments": [seg], "correction_count": 0},
    )

    saved = []
    # Patch at the app module level (from-import binding — NOT the db module)
    monkeypatch.setattr("subgen_ai.app.save_correction", lambda r: saved.append(r))
    monkeypatch.setattr(st, "rerun", lambda: None)

    vr = ValidationResult(
        accepted=True, score=0.98, tier="HIGH",
        message="High similarity", hw_used=False,
    )
    app._do_save_correction(seg, "Corrected text", vr, override=False)

    assert len(saved) == 1, "Expected exactly one correction to be saved"
    assert saved[0].corrected_text == "Corrected text"
    assert st.session_state["correction_count"] == 1
