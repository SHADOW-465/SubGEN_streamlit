import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
from subgen_ai.export.formatters import to_srt, to_vtt, to_json, _format_ts_srt
from subgen_ai.core.models import SubtitleSegment


def _seg(index=0, start=0.0, end=2.5, text="Hello world", label="GREEN",
         fused_conf=0.85, asr_conf=0.9, snr_db=20.0):
    return SubtitleSegment(
        index=index, start=start, end=end, text=text, language="en",
        asr_conf=asr_conf, snr_db=snr_db, fused_conf=fused_conf, label=label
    )


def test_format_ts_srt_zero():
    assert _format_ts_srt(0.0) == "00:00:00,000"


def test_format_ts_srt_complex():
    # 1h 2m 3s 456ms
    ts = 3600 + 120 + 3 + 0.456
    assert _format_ts_srt(ts) == "01:02:03,456"


def test_to_srt_structure():
    segs = [_seg(0, 0.0, 2.0, "Hello"), _seg(1, 3.0, 5.0, "World")]
    srt = to_srt(segs)
    lines = srt.strip().splitlines()
    assert lines[0] == "1"
    assert "-->" in lines[1]
    assert "Hello" in lines[2]
    assert lines[3] == ""
    assert lines[4] == "2"


def test_to_vtt_header():
    srt = to_vtt([_seg()])
    assert srt.startswith("WEBVTT")


def test_to_vtt_uses_dot_separator():
    srt = to_vtt([_seg(start=1.5, end=3.7)])
    assert "." in srt and "," not in srt.split("WEBVTT")[1]


def test_to_json_structure():
    segs = [_seg(label="RED", fused_conf=0.6)]
    result = json.loads(to_json(segs))
    assert "segments" in result
    seg_data = result["segments"][0]
    assert seg_data["text"] == "Hello world"
    assert seg_data["quality"]["label"] == "RED"
    assert "fused_conf" in seg_data["quality"]
    assert "snr_db" in seg_data["quality"]
    assert "hw_fingerprint" in seg_data["quality"]


def test_to_json_ensure_ascii_false():
    """Tamil/Indic text must not be escaped."""
    seg = _seg(text="வணக்கம்")
    result = to_json([seg])
    assert "வணக்கம்" in result


# ── to_burn_in tests ────────────────────────────────────────────────────────
import os
import tempfile
from unittest.mock import patch, MagicMock, call
import pytest


def _popen_mock():
    """Return a MagicMock that looks like a successful subprocess.Popen result."""
    m = MagicMock()
    m.communicate.return_value = (b"", b"")
    m.returncode = 0
    return m


def _seg_for_burn():
    """Minimal GREEN SubtitleSegment for burn-in tests."""
    from subgen_ai.core.models import SubtitleSegment
    return SubtitleSegment(
        index=0, start=0.0, end=1.0, text="Hello", language="en",
        asr_conf=0.9, snr_db=20.0, fused_conf=0.9, label="GREEN",
        mfcc_mean=[], mfcc_var=[], hw_fingerprint=False,
        corrected=False, correction_text="",
    )


def test_to_burn_in_calls_ffmpeg():
    """to_burn_in must invoke subprocess.Popen (used internally by ffmpeg-python)."""
    from subgen_ai.export.formatters import to_burn_in
    with patch("subprocess.Popen", return_value=_popen_mock()) as mock_popen:
        try:
            to_burn_in(b"FAKEVIDEO", [_seg_for_burn()])
        except Exception:
            pass  # output file won't exist in test; we only check Popen was called
    assert mock_popen.called


def test_to_burn_in_cleans_up_temp_files():
    """All temp files created by to_burn_in must be deleted after the call.

    Tracks both NamedTemporaryFile paths AND the mkstemp output path.
    """
    from subgen_ai.export.formatters import to_burn_in

    ntf_paths = []
    real_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(**kw):
        f = real_ntf(**kw)
        ntf_paths.append(f.name)
        return f

    mkstemp_paths = []
    real_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(**kw):
        fd, path = real_mkstemp(**kw)
        mkstemp_paths.append(path)
        return fd, path

    with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf), \
         patch("tempfile.mkstemp", side_effect=tracking_mkstemp), \
         patch("subprocess.Popen", return_value=_popen_mock()):
        try:
            to_burn_in(b"FAKEVIDEO", [_seg_for_burn()])
        except Exception:
            pass

    all_paths = ntf_paths + mkstemp_paths
    assert all_paths, "Expected at least one temp file to be created"
    for p in all_paths:
        assert not os.path.exists(p), f"Temp file not cleaned up: {p}"


def test_to_burn_in_raises_runtime_error_on_ffmpeg_failure():
    """ffmpeg.Error raised inside the ffmpeg call must be re-raised as RuntimeError.

    We patch ffmpeg._run.run — the internal entry point that ffmpeg-python's
    fluent .run() delegates to — so the error is raised inside the try block
    where the except ffmpeg.Error guard lives.
    """
    import ffmpeg as ffmpeg_mod
    from subgen_ai.export.formatters import to_burn_in

    err = ffmpeg_mod.Error("ffmpeg", b"", b"something went wrong")
    # Patch the internal run function that ffmpeg-python's .run() calls
    with patch("ffmpeg._run.run", side_effect=err):
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            to_burn_in(b"FAKEVIDEO", [_seg_for_burn()])
