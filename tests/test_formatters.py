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
