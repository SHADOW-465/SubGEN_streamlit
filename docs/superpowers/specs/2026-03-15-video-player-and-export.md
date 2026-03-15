# Video Player, Universal Edit & Burn-in Export — Design Spec

**Date:** 2026-03-15
**Status:** Approved (rev 4 — final)

---

## Goal

Add a browser-native HTML video player with live subtitle overlay to the Review & Edit tab, make every subtitle segment (RED and GREEN) editable with full DB persistence, and add a burn-in `.mp4` download alongside the existing SRT / VTT / JSON exports.

---

## Architecture

Three focused changes to the existing `subgen_ai/` package:

1. **New component** — `subgen_ai/components/video_player.py` renders a `<video>` + `<track>` via data URIs, injected through `st.components.v1.html()`.
2. **Extended formatter** — `subgen_ai/export/formatters.py` gets `to_burn_in()` using `ffmpeg-python`.
3. **App wiring** — `subgen_ai/app.py`: capture video bytes at upload, add player to Review tab, remove GREEN-only display branch, add Burn-in button to Export tab.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `subgen_ai/components/` | **Create directory** | Package directory — mkdir before any files |
| `subgen_ai/components/__init__.py` | Create | Empty package init |
| `subgen_ai/components/video_player.py` | Create | `render_video_player(video_bytes, mime, vtt_str, height)` |
| `subgen_ai/export/formatters.py` | Modify | Add `to_burn_in(video_bytes, segments, ext) → bytes` |
| `subgen_ai/app.py` | Modify | Capture bytes; player; universal edit; burn-in button |
| `tests/test_video_player.py` | Create | 4 unit tests for player HTML generation |
| `tests/test_formatters.py` | Modify | 3 `to_burn_in` tests (mocked ffmpeg) |
| `tests/test_app_review.py` | Create | GREEN segment edit/save test |

---

## Session State Changes

Add both keys to the **`defaults` dict inside `_init_state()`**:

```python
defaults = {
    # ... existing keys ...
    "video_bytes": None,   # bytes | None
    "video_ext":   "",     # str e.g. ".mp4"
}
```

---

## Video Bytes Capture in `_run_transcription()`

Replace the existing `suffix = ...` + `NamedTemporaryFile` block with:

```python
# Capture bytes BEFORE try block so they're saved even if transcription fails
raw_bytes = uploaded_file.getvalue()          # getvalue() ignores stream position
suffix    = Path(uploaded_file.name).suffix or ".mp4"
st.session_state["video_bytes"] = raw_bytes
st.session_state["video_ext"]   = suffix

with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(raw_bytes)                       # use raw_bytes, NOT uploaded_file.read()
    tmp_path = tmp.name
```

---

## Component: `video_player.py`

```python
import base64
import streamlit as st
import streamlit.components.v1 as components

MAX_PLAYER_BYTES = 80 * 1024 * 1024  # 80 MB

def render_video_player(
    video_bytes: bytes,
    mime: str,
    vtt_str: str,
    height: int = 360,
) -> None:
    if len(video_bytes) > MAX_PLAYER_BYTES:
        st.warning("⚠ File too large for in-browser player (>80 MB). Download subtitles below.")
        return
    video_b64 = base64.b64encode(video_bytes).decode()
    vtt_b64   = base64.b64encode(vtt_str.encode("utf-8")).decode()
    html = f"""
<video controls width="100%"
       style="max-height:{height}px; background:#000; border-radius:8px;">
  <source src="data:{mime};base64,{video_b64}" type="{mime}">
  <track default kind="subtitles" srclang="en"
         src="data:text/vtt;base64,{vtt_b64}">
  Your browser does not support the video tag.
</video>
"""
    components.html(html, height=height + 20)
```

---

## Formatter Addition: `to_burn_in()`

Signature includes `ext` so the input temp file gets the correct container extension:

```python
def to_burn_in(
    video_bytes: bytes,
    segments: List[SubtitleSegment],
    ext: str = ".mp4",
) -> bytes:
    import os, tempfile, ffmpeg

    srt_path = vid_path = out_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, encoding="utf-8"
        ) as f:
            f.write(to_srt(segments))
            srt_path = f.name

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(video_bytes)
            vid_path = f.name

        out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_fd)

        try:
            (
                ffmpeg
                .input(vid_path)
                .output(out_path, vf=f"subtitles='{srt_path}'", acodec="copy")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='replace')}")

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        for p in (srt_path, vid_path, out_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
```

---

## MIME Helper

Add to `app.py` (module level):

```python
import mimetypes

def _get_video_mime() -> str:
    ext = st.session_state.get("video_ext", ".mp4")
    mime, _ = mimetypes.guess_type(f"file{ext}")
    return mime or "video/mp4"
```

---

## Review Tab Changes

### Render order in `render_tab_review()`

```python
def render_tab_review() -> None:
    if not st.session_state.get("done"):
        st.info("ℹ Transcription not yet run...")
        return

    segments  = st.session_state["segments"]
    red_segs  = [s for s in segments if s.label == "RED"]    # still needed for metrics
    green_segs = [s for s in segments if s.label == "GREEN"] # still needed for metrics

    # 1. Player — top, before header
    vb = st.session_state.get("video_bytes")
    if vb:
        from subgen_ai.components.video_player import render_video_player
        render_video_player(vb, _get_video_mime(), to_vtt(segments))
        st.markdown("---")

    # 2. Header (move existing st.header line to here — do NOT add a second call)
    st.header("✏ Review & Edit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total segments",        len(segments))
    c2.metric("🔴 RED (review needed)", len(red_segs))
    c3.metric("🟢 GREEN (confident)",   len(green_segs))
    c4.metric("✅ Corrections saved",   st.session_state.get("correction_count", 0))

    # 3. Filter toggle
    st.session_state["filter_red_only"] = st.toggle(...)
    display_segs = red_segs if st.session_state["filter_red_only"] else segments

    # 4. Segments
    st.markdown("---")
    for seg in display_segs:
        _render_segment_card(seg)
```

### `_render_segment_card()` — remove if/else branch entirely

Replace the current body with (applies to ALL segments regardless of label):

```python
def _render_segment_card(seg: SubtitleSegment) -> None:
    label_icon = "🟢" if seg.label == "GREEN" else "🔴"
    hw_badge   = "🔧 HW-MFCC" if seg.hw_fingerprint else "💻 SW-MFCC"
    auto_badge = " 🔄 Auto-corrected from DB" if seg.corrected else ""
    header = (
        f"{label_icon}  [{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        f"  fused_conf: {seg.fused_conf:.3f}{auto_badge}"
    )
    with st.expander(header, expanded=(seg.label == "RED")):
        # ALL segments: editable text area
        edit_key = f"edit_{seg.index}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = seg.text
        edited_text = st.text_area("✏ Edit transcript", key=edit_key, height=80)

        m1, m2, m3 = st.columns(3)
        m1.caption(f"ASR conf: **{seg.asr_conf:.3f}**")
        m2.caption(f"SNR: **{seg.snr_db:.1f} dB**")
        m3.caption(hw_badge)

        # ALL segments: same Save button and validation flow
        col_btn, col_msg = st.columns([2, 3])
        with col_btn:
            validate_clicked = st.button(
                "💾 Validate & Save Correction",   # keep existing label
                key=f"validate_{seg.index}",
                use_container_width=True,
            )
        with col_msg:
            vr_key = f"vr_{seg.index}"
            if vr_key in st.session_state:
                vr = st.session_state[vr_key]
                _show_validation_result(vr)
                if vr.tier == "MISMATCH":
                    if st.button("💾 Save anyway (override)", key=f"override_{seg.index}"):
                        _do_save_correction(seg, edited_text, vr, override=True)
        if validate_clicked:
            _handle_validate_correction(seg, edited_text)
```

---

## Export Tab Changes

1. Update the top-level formatter import in `app.py`:
   ```python
   from subgen_ai.export.formatters import to_srt, to_vtt, to_json, to_burn_in
   ```
2. Change `col1, col2, col3 = st.columns(3)` → `col1, col2, col3, col4 = st.columns(4)`.
2. Re-assign columns: `col1` = burn-in, `col2` = SRT, `col3` = VTT, `col4` = JSON.

### Burn-in button (`col1`)

```python
with col1:
    video_bytes = st.session_state.get("video_bytes")
    video_ext   = st.session_state.get("video_ext", "")
    is_video    = video_ext in (".mp4", ".avi", ".mov", ".mkv")
    if not video_bytes or not is_video:
        st.info("🔥 Burn-in requires a video file (not audio-only).")
    else:
        if st.button("🔥 Generate Burn-in .mp4", use_container_width=True):
            try:
                with st.spinner("🔥 Encoding…"):
                    burned = to_burn_in(video_bytes, segments, ext=video_ext)
                st.download_button(
                    "⬇ Download burned-in .mp4",
                    data=burned,
                    file_name=f"{base_name}_burned.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            except RuntimeError as exc:
                st.error(f"❌ Burn-in failed: {exc}")
```

---

## Testing

### `tests/test_video_player.py` (new file, 4 tests)

```python
import base64, re
import pytest
from subgen_ai.components.video_player import render_video_player, MAX_PLAYER_BYTES

FAKE_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n"

def _capture(monkeypatch):
    calls = []
    monkeypatch.setattr("streamlit.components.v1.html", lambda h, **kw: calls.append(h))
    return calls

def test_html_contains_video_tag(monkeypatch):
    c = _capture(monkeypatch)
    render_video_player(b"V", "video/mp4", FAKE_VTT)
    assert "<video" in c[0]

def test_html_contains_track_tag(monkeypatch):
    c = _capture(monkeypatch)
    render_video_player(b"V", "video/mp4", FAKE_VTT)
    assert "<track" in c[0]

def test_track_src_is_valid_base64_vtt_data_uri(monkeypatch):
    c = _capture(monkeypatch)
    render_video_player(b"V", "video/mp4", FAKE_VTT)
    m = re.search(r'src="(data:text/vtt;base64,[^"]+)"', c[0])
    assert m, "No data:text/vtt;base64 URI in <track src>"
    decoded = base64.b64decode(m.group(1).split(",", 1)[1]).decode()
    assert decoded == FAKE_VTT

def test_large_file_warns_and_skips(monkeypatch):
    warns = []
    html_calls = []
    monkeypatch.setattr("streamlit.warning", lambda m: warns.append(m))
    monkeypatch.setattr("streamlit.components.v1.html", lambda h, **kw: html_calls.append(h))
    render_video_player(b"x" * (MAX_PLAYER_BYTES + 1), "video/mp4", FAKE_VTT)
    assert warns and not html_calls
```

### `tests/test_formatters.py` additions (3 tests)

Mock target: **`subprocess.Popen`** (what `ffmpeg-python` calls internally — NOT `subprocess.run`).

```python
from unittest.mock import patch, MagicMock
import pytest
from subgen_ai.export.formatters import to_burn_in
from subgen_ai.core.models import SubtitleSegment

def _seg():
    return SubtitleSegment(
        index=0, start=0.0, end=1.0, text="Hello", language="en",
        asr_conf=0.9, snr_db=20.0, fused_conf=0.9, label="GREEN",
        mfcc_mean=[], mfcc_var=[], hw_fingerprint=False,
        corrected=False, correction_text="",
    )

def _popen_mock():
    m = MagicMock()
    m.communicate.return_value = (b"", b"")
    m.returncode = 0
    return m

def test_to_burn_in_calls_ffmpeg(tmp_path):
    with patch("subprocess.Popen", return_value=_popen_mock()) as mock_popen:
        try:
            to_burn_in(b"FAKEVIDEO", [_seg()])
        except Exception:
            pass
    assert mock_popen.called

def test_to_burn_in_cleans_up_temp_files():
    import os, tempfile
    created = []
    real_ntf = tempfile.NamedTemporaryFile
    def track(**kw):
        f = real_ntf(**kw); created.append(f.name); return f
    with patch("tempfile.NamedTemporaryFile", side_effect=track), \
         patch("subprocess.Popen", return_value=_popen_mock()):
        try:
            to_burn_in(b"FAKEVIDEO", [_seg()])
        except Exception:
            pass
    for p in created:
        assert not os.path.exists(p)

def test_to_burn_in_raises_on_ffmpeg_error():
    import ffmpeg as ffmpeg_mod
    err = ffmpeg_mod.Error("cmd", b"", b"bad error")
    with patch("subprocess.Popen", side_effect=err):
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            to_burn_in(b"FAKEVIDEO", [_seg()])
```

### `tests/test_app_review.py` (new file, 1 test)

```python
from subgen_ai.core.models import SubtitleSegment, ValidationResult

def _green_seg():
    return SubtitleSegment(
        index=0, start=0.0, end=1.0, text="Original", language="en",
        asr_conf=0.95, snr_db=25.0, fused_conf=0.95, label="GREEN",
        mfcc_mean=[], mfcc_var=[], hw_fingerprint=False,
        corrected=False, correction_text="",   # empty string, NOT None
    )

def test_do_save_correction_works_for_green_segment(monkeypatch):
    from subgen_ai import app
    import streamlit as st
    seg = _green_seg()
    monkeypatch.setattr(st, "session_state", {"segments": [seg], "correction_count": 0})
    saved = []
    monkeypatch.setattr("subgen_ai.app.save_correction", lambda r: saved.append(r))
    monkeypatch.setattr(st, "rerun", lambda: None)
    vr = ValidationResult(accepted=True, score=0.98, tier="HIGH",
                          message="High similarity", hw_used=False)
    app._do_save_correction(seg, "Corrected text", vr, override=False)
    assert len(saved) == 1
    assert saved[0].corrected_text == "Corrected text"
    assert st.session_state["correction_count"] == 1
```

---

## Out of Scope

- Subtitle timing editor
- Multi-track / multi-language subtitle support
- Progress reporting during burn-in encoding
- Files >80 MB in the browser player (warn and skip)
