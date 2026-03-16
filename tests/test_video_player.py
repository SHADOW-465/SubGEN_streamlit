# tests/test_video_player.py
import base64
import re
import pytest

# Import will fail until we create the module — that's the point
from subgen_ai.components.video_player import render_video_player, MAX_PLAYER_BYTES

FAKE_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n"


def _capture_html(monkeypatch):
    """Monkeypatch st.components.v1.html and return the call list."""
    calls = []
    monkeypatch.setattr(
        "streamlit.components.v1.html", lambda h, **kw: calls.append(h)
    )
    return calls


def test_html_contains_video_tag(monkeypatch):
    calls = _capture_html(monkeypatch)
    render_video_player(b"FAKEVIDEO", "video/mp4", FAKE_VTT)
    assert "<video" in calls[0]


def test_html_contains_track_tag(monkeypatch):
    calls = _capture_html(monkeypatch)
    render_video_player(b"FAKEVIDEO", "video/mp4", FAKE_VTT)
    assert "<track" in calls[0]


def test_track_src_is_valid_base64_vtt_data_uri(monkeypatch):
    calls = _capture_html(monkeypatch)
    render_video_player(b"FAKEVIDEO", "video/mp4", FAKE_VTT)
    html = calls[0]
    match = re.search(r'src="(data:text/vtt;base64,[^"]+)"', html)
    assert match, "No data:text/vtt;base64 URI found in <track src>"
    b64_part = match.group(1).split(",", 1)[1]
    decoded = base64.b64decode(b64_part).decode("utf-8")
    assert decoded == FAKE_VTT


def test_large_file_warns_and_skips(monkeypatch):
    warns = []
    html_calls = []
    monkeypatch.setattr("streamlit.warning", lambda m: warns.append(m))
    monkeypatch.setattr(
        "streamlit.components.v1.html", lambda h, **kw: html_calls.append(h)
    )
    big = b"x" * (MAX_PLAYER_BYTES + 1)
    render_video_player(big, "video/mp4", FAKE_VTT)
    assert warns, "Expected a st.warning for oversized file"
    assert not html_calls, "Expected no HTML injection for oversized file"


def test_invalid_mime_shows_error_and_skips(monkeypatch):
    errors = []
    html_calls = []
    monkeypatch.setattr("streamlit.error", lambda m: errors.append(m))
    monkeypatch.setattr("streamlit.components.v1.html", lambda h, **kw: html_calls.append(h))
    render_video_player(b"FAKEVIDEO", 'video/mp4" onerror="alert(1)', FAKE_VTT)
    assert errors, "Expected st.error for invalid MIME"
    assert not html_calls, "Expected no HTML for invalid MIME"


def test_track_has_label_attribute(monkeypatch):
    calls = _capture_html(monkeypatch)
    render_video_player(b"FAKEVIDEO", "video/mp4", FAKE_VTT)
    assert 'label="Subtitles"' in calls[0]


def test_empty_video_bytes_warns_and_skips(monkeypatch):
    warns = []
    html_calls = []
    monkeypatch.setattr("streamlit.warning", lambda m: warns.append(m))
    monkeypatch.setattr("streamlit.components.v1.html", lambda h, **kw: html_calls.append(h))
    render_video_player(b"", "video/mp4", FAKE_VTT)
    assert warns
    assert not html_calls
