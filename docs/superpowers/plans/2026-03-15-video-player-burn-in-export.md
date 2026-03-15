# Video Player, Universal Edit & Burn-in Export — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a browser-native HTML video player with subtitle overlay to the Review tab, make all segments editable (GREEN and RED), and add a burn-in `.mp4` download alongside the existing SRT/VTT/JSON exports.

**Architecture:** A new `subgen_ai/components/video_player.py` injects a `<video>+<track>` element via `st.components.v1.html()` using base64 data URIs (no file server needed). `to_burn_in()` is added to the existing `formatters.py` and uses `ffmpeg-python` to re-encode the video with subtitles baked into pixels. `app.py` is modified in four places: `_init_state`, `_run_transcription`, `_render_segment_card`, and `render_tab_export`.

**Tech Stack:** Python 3.10+, Streamlit, `ffmpeg-python` (already in requirements), `base64` (stdlib), `mimetypes` (stdlib), `tempfile` (stdlib), `pytest`, `unittest.mock`

**Spec:** `docs/superpowers/specs/2026-03-15-video-player-and-export.md`

---

## Chunk 1: `video_player` component (TDD)

**Files:**
- Create: `subgen_ai/components/__init__.py`
- Create: `subgen_ai/components/video_player.py`
- Create: `tests/test_video_player.py`

---

### Task 1: Scaffold the components package

- [ ] **Step 1: Create the directory and empty init**

```bash
mkdir "subgen_ai/components"
echo "" > "subgen_ai/components/__init__.py"
```

- [ ] **Step 2: Verify it exists**

```bash
ls subgen_ai/components/
```
Expected output: `__init__.py`

- [ ] **Step 3: Commit**

```bash
git add subgen_ai/components/__init__.py
git commit -m "feat: scaffold subgen_ai/components package"
```

---

### Task 2: Write failing tests for `render_video_player`

- [ ] **Step 1: Create `tests/test_video_player.py`**

```python
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
```

- [ ] **Step 2: Run tests to confirm they all FAIL**

```bash
cd "C:\Users\acer\Documents\projects\SubGEN_streamlit\.claude\worktrees\eloquent-benz"
python -m pytest tests/test_video_player.py -v
```
Expected: 4 errors — `ModuleNotFoundError: No module named 'subgen_ai.components.video_player'`

---

### Task 3: Implement `video_player.py`

- [ ] **Step 1: Create `subgen_ai/components/video_player.py`**

```python
# subgen_ai/components/video_player.py
"""
SubGEN AI — Browser Video Player Component.

Renders a native HTML <video> element with a WebVTT subtitle <track>
injected via st.components.v1.html(). Both video and VTT are embedded
as base64 data URIs — no file server required.
"""
import base64

import streamlit as st
import streamlit.components.v1 as components

MAX_PLAYER_BYTES = 80 * 1024 * 1024  # 80 MB — browser limit for data URIs


def render_video_player(
    video_bytes: bytes,
    mime: str,
    vtt_str: str,
    height: int = 360,
) -> None:
    """
    Render an HTML5 video player with embedded subtitle track.

    Args:
        video_bytes: Raw video file bytes.
        mime:        MIME type string, e.g. "video/mp4".
        vtt_str:     Full WebVTT string (must start with "WEBVTT").
        height:      Player height in pixels (default 360).
    """
    if len(video_bytes) > MAX_PLAYER_BYTES:
        st.warning(
            "⚠ File too large for in-browser player (>80 MB). "
            "Download subtitles below."
        )
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

- [ ] **Step 2: Run tests — all 4 should PASS**

```bash
python -m pytest tests/test_video_player.py -v
```
Expected:
```
PASSED tests/test_video_player.py::test_html_contains_video_tag
PASSED tests/test_video_player.py::test_html_contains_track_tag
PASSED tests/test_video_player.py::test_track_src_is_valid_base64_vtt_data_uri
PASSED tests/test_video_player.py::test_large_file_warns_and_skips
```

- [ ] **Step 3: Commit**

```bash
git add subgen_ai/components/video_player.py tests/test_video_player.py
git commit -m "feat: add HTML video player component with WebVTT subtitle track"
```

---

## Chunk 2: `to_burn_in` formatter (TDD)

**Files:**
- Modify: `subgen_ai/export/formatters.py` (add `to_burn_in`)
- Modify: `tests/test_formatters.py` (add 3 tests)

---

### Task 4: Write failing tests for `to_burn_in`

- [ ] **Step 1: Add these tests to the END of `tests/test_formatters.py`**

Open `tests/test_formatters.py` and append:

```python
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
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
python -m pytest tests/test_formatters.py::test_to_burn_in_calls_ffmpeg tests/test_formatters.py::test_to_burn_in_cleans_up_temp_files tests/test_formatters.py::test_to_burn_in_raises_runtime_error_on_ffmpeg_failure -v
```
Expected: 3 failures — `ImportError: cannot import name 'to_burn_in'`

---

### Task 5: Implement `to_burn_in` in `formatters.py`

- [ ] **Step 1: Add `to_burn_in` to `subgen_ai/export/formatters.py`**

Add these imports at the top of the file (after existing imports):

```python
import os
import tempfile
```

Then append `to_burn_in` at the bottom of the file:

```python
def to_burn_in(
    video_bytes: bytes,
    segments: List[SubtitleSegment],
    ext: str = ".mp4",
) -> bytes:
    """
    Re-encode video with subtitles permanently baked into the pixels.

    Uses ffmpeg's ``subtitles`` video filter. Audio is copied without
    re-encoding (``acodec=copy``) for speed.

    Args:
        video_bytes: Raw bytes of the source video file.
        segments:    List of SubtitleSegment (text is used as-is).
        ext:         File extension of the source video (e.g. ".mp4", ".avi").
                     Used so ffmpeg can probe the correct container format.

    Returns:
        Raw bytes of the burned-in MP4.

    Raises:
        RuntimeError: If ffmpeg exits with an error.
    """
    import ffmpeg  # local import — only needed when this function runs

    srt_path = vid_path = out_path = None
    try:
        # 1. Write SRT subtitles to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, encoding="utf-8"
        ) as f:
            f.write(to_srt(segments))
            srt_path = f.name

        # 2. Write the source video bytes to a temp file with correct extension
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(video_bytes)
            vid_path = f.name

        # 3. Prepare output path
        out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_fd)

        # 4. Run ffmpeg: burn subtitles, copy audio track
        try:
            (
                ffmpeg
                .input(vid_path)
                .output(out_path, vf=f"subtitles='{srt_path}'", acodec="copy")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"ffmpeg failed: {e.stderr.decode(errors='replace')}"
            )

        # 5. Return the output bytes
        with open(out_path, "rb") as f:
            return f.read()

    finally:
        # 6. Always clean up temp files
        for p in (srt_path, vid_path, out_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
```

- [ ] **Step 2: Run burn-in tests — all 3 should PASS**

```bash
python -m pytest tests/test_formatters.py::test_to_burn_in_calls_ffmpeg tests/test_formatters.py::test_to_burn_in_cleans_up_temp_files tests/test_formatters.py::test_to_burn_in_raises_runtime_error_on_ffmpeg_failure -v
```
Expected: 3 PASSED

- [ ] **Step 3: Run the full formatter test suite to check no regressions**

```bash
python -m pytest tests/test_formatters.py -v
```
Expected: All existing tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add subgen_ai/export/formatters.py tests/test_formatters.py
git commit -m "feat: add to_burn_in() — ffmpeg subtitle burn-in export"
```

---

## Chunk 3: App wiring — session state, bytes capture, MIME helper

**Files:**
- Modify: `subgen_ai/app.py` (3 targeted changes)

---

### Task 6: Stub Streamlit's page config in conftest so app.py can be imported in tests

`subgen_ai/app.py` calls `st.set_page_config()` and `st.markdown()` at module level when imported.
Importing it in pytest without a live Streamlit runtime raises `StreamlitAPIException`.
We fix this once in `tests/conftest.py` so every test file that imports `app` works.

- [ ] **Step 1: Check whether `tests/conftest.py` already exists**

```bash
ls tests/conftest.py 2>/dev/null && echo EXISTS || echo MISSING
```

- [ ] **Step 2: If MISSING, create it. If it EXISTS, append the fixture below.**

Full content of `tests/conftest.py` (create or add to existing):

```python
# tests/conftest.py
"""
Shared pytest fixtures.

Stubs Streamlit's runtime-only calls (set_page_config, markdown) so that
`subgen_ai.app` can be imported in unit tests without a live Streamlit server.
"""
import pytest
import unittest.mock as mock


@pytest.fixture(autouse=True)
def stub_streamlit_runtime(monkeypatch):
    """
    Prevent st.set_page_config / st.markdown from raising StreamlitAPIException
    when app.py is imported outside a running Streamlit server.
    """
    import streamlit as st
    monkeypatch.setattr(st, "set_page_config", lambda **kw: None)
    monkeypatch.setattr(st, "markdown",        lambda *a, **kw: None)
```

- [ ] **Step 3: Run the existing test suite to confirm nothing breaks**

```bash
python -m pytest tests/ -v --ignore=tests/test_app_review.py -x
```
Expected: All previously passing tests still PASS.

---

### Task 7: Write the GREEN segment save test

- [ ] **Step 1: Create `tests/test_app_review.py`**

```python
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
```

- [ ] **Step 2: Run to confirm it PASSES**

```bash
python -m pytest tests/test_app_review.py -v
```
Expected: PASS (the save function itself is already label-agnostic)

---

### Task 8: Add session state keys and video bytes capture to `app.py`

- [ ] **Step 1: Add `video_bytes` and `video_ext` to `_init_state()`**

In `subgen_ai/app.py`, find the `defaults = {` dict inside `_init_state()` (around line 56). Add two new keys at the end:

```python
    defaults = {
        "segments": [],
        "audio": None,
        "sr": 16000,
        "port": None,
        "done": False,
        "filename": "",
        "esp32_ports": [],
        "hw_status": "software",
        "correction_count": 0,
        "filter_red_only": False,
        "video_bytes": None,   # bytes | None — raw uploaded file for the player
        "video_ext":   "",     # str — e.g. ".mp4", ".avi"
    }
```

- [ ] **Step 2: Replace the bytes capture in `_run_transcription()`**

Find this block (around lines 290–295):

```python
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
```

Replace it with:

```python
    # Capture raw bytes via getvalue() — works regardless of stream position.
    # Done BEFORE the try block so bytes are saved even if transcription fails.
    raw_bytes = uploaded_file.getvalue()
    suffix    = Path(uploaded_file.name).suffix or ".mp4"
    st.session_state["video_bytes"] = raw_bytes
    st.session_state["video_ext"]   = suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)          # use raw_bytes, NOT uploaded_file.read()
        tmp_path = tmp.name
```

- [ ] **Step 3: Add `_get_video_mime()` helper**

Add this function after `_get_audio_clip()` (around line 97), before the sidebar section:

```python
def _get_video_mime() -> str:
    """Derive a MIME type from the stored video extension."""
    import mimetypes
    ext = st.session_state.get("video_ext", ".mp4")
    mime, _ = mimetypes.guess_type(f"file{ext}")
    return mime or "video/mp4"
```

- [ ] **Step 4: Update the formatter import line**

Find (around line 40):

```python
from subgen_ai.export.formatters import to_srt, to_vtt, to_json
```

Replace with:

```python
from subgen_ai.export.formatters import to_srt, to_vtt, to_json, to_burn_in
```

- [ ] **Step 5: Run existing tests to verify no regressions**

```bash
python -m pytest tests/ -v --ignore=tests/test_video_player.py -x
```
Expected: All previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add subgen_ai/app.py tests/test_app_review.py
git commit -m "feat: capture video bytes in session state and add MIME helper"
```

---

## Chunk 4: Review tab — player + universal editing

**Files:**
- Modify: `subgen_ai/app.py` — `render_tab_review()` and `_render_segment_card()`

---

### Task 9: Add video player to the top of the Review tab

- [ ] **Step 1: Add the player block to `render_tab_review()`**

Find `render_tab_review()` (around line 349). After the guard block:

```python
def render_tab_review() -> None:
    if not st.session_state.get("done"):
        st.info("ℹ Transcription not yet run. Go to **Upload & Transcribe** first.")
        return
```

Add the player block immediately after, before the `segments = ...` line:

```python
    segments: list[SubtitleSegment] = st.session_state["segments"]
    red_segs   = [s for s in segments if s.label == "RED"]
    green_segs = [s for s in segments if s.label == "GREEN"]

    # ── Video player (top of tab) ─────────────────────────────────────────────
    vb = st.session_state.get("video_bytes")
    if vb:
        from subgen_ai.components.video_player import render_video_player
        render_video_player(vb, _get_video_mime(), to_vtt(segments))
        st.markdown("---")

    # ── Header + summary metrics ─────────────────────────────────────────────
    st.header("✏ Review & Edit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total segments",        len(segments))
    c2.metric("🔴 RED (review needed)", len(red_segs))
    c3.metric("🟢 GREEN (confident)",   len(green_segs))
    c4.metric("✅ Corrections saved",   st.session_state.get("correction_count", 0))
```

Make sure to **remove** the original `st.header("✏ Review & Edit")` call and the metrics block that currently follows it — do not duplicate them.

- [ ] **Step 2: Verify the app launches without error**

```bash
streamlit run subgen_ai/app.py --server.headless true &
sleep 5 && kill %1
```
Expected: No Python traceback in output.

---

### Task 10: Make all segments editable — remove the `if/else` branch

- [ ] **Step 1: Replace `_render_segment_card()` body**

Find `_render_segment_card()` (around line 385). The current body has an `if seg.label == "RED": ... else: st.markdown(...)` branch. Replace the **entire function body** (everything inside `with st.expander(...):`) with the universal version:

```python
def _render_segment_card(seg: SubtitleSegment) -> None:
    """Render a single segment — all labels get an editable text area."""
    label_icon = "🟢" if seg.label == "GREEN" else "🔴"
    hw_badge   = "🔧 HW-MFCC" if seg.hw_fingerprint else "💻 SW-MFCC"
    auto_badge = " 🔄 Auto-corrected from DB" if seg.corrected else ""

    header = (
        f"{label_icon}  [{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        f"  fused_conf: {seg.fused_conf:.3f}{auto_badge}"
    )

    with st.expander(header, expanded=(seg.label == "RED")):
        # ── Editable transcript — ALL segments ───────────────────────────────
        edit_key = f"edit_{seg.index}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = seg.text
        edited_text = st.text_area("✏ Edit transcript", key=edit_key, height=80)

        # ── Footer metrics ────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.caption(f"ASR conf: **{seg.asr_conf:.3f}**")
        m2.caption(f"SNR: **{seg.snr_db:.1f} dB**")
        m3.caption(hw_badge)

        # ── Save button — ALL segments (same flow as before) ─────────────────
        col_btn, col_msg = st.columns([2, 3])
        with col_btn:
            validate_clicked = st.button(
                "💾 Validate & Save Correction",
                key=f"validate_{seg.index}",
                use_container_width=True,
            )
        with col_msg:
            vr_key = f"vr_{seg.index}"
            if vr_key in st.session_state:
                vr: ValidationResult = st.session_state[vr_key]
                _show_validation_result(vr)
                if vr.tier == "MISMATCH":
                    if st.button(
                        "💾 Save anyway (override)",
                        key=f"override_{seg.index}",
                    ):
                        _do_save_correction(seg, edited_text, vr, override=True)

        if validate_clicked:
            _handle_validate_correction(seg, edited_text)
```

- [ ] **Step 2: Run the GREEN save test**

```bash
python -m pytest tests/test_app_review.py -v
```
Expected: PASS

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add subgen_ai/app.py
git commit -m "feat: add video player to review tab and make all segments editable"
```

---

## Chunk 5: Export tab — 4-column layout + burn-in button

**Files:**
- Modify: `subgen_ai/app.py` — `render_tab_export()`

---

### Task 11: Update the Export tab

- [ ] **Step 1: Find `render_tab_export()` and update the columns**

Find (around line 550):

```python
    col1, col2, col3 = st.columns(3)
```

Replace with:

```python
    col1, col2, col3, col4 = st.columns(4)
```

- [ ] **Step 2: Add the burn-in button in `col1` and shift existing buttons**

The current column assignments are `col1`=SRT, `col2`=VTT, `col3`=JSON. Re-assign so burn-in takes `col1`, and shift the rest right:

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
                    with st.spinner("🔥 Encoding burn-in subtitles…"):
                        burned = to_burn_in(video_bytes, segments, ext=video_ext)
                    st.download_button(
                        label="⬇ Download burned-in .mp4",
                        data=burned,
                        file_name=f"{base_name}_burned.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                except RuntimeError as exc:
                    st.error(f"❌ Burn-in failed: {exc}")

    with col2:
        st.download_button(
            label="⬇ Download .srt",
            data=srt_str.encode("utf-8"),
            file_name=f"{base_name}.srt",
            mime="text/plain",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            label="⬇ Download .vtt",
            data=vtt_str.encode("utf-8"),
            file_name=f"{base_name}.vtt",
            mime="text/vtt",
            use_container_width=True,
        )

    with col4:
        st.download_button(
            label="⬇ Download .json (with QC metadata)",
            data=json_str.encode("utf-8"),
            file_name=f"{base_name}_qc.json",
            mime="application/json",
            use_container_width=True,
        )
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add subgen_ai/app.py
git commit -m "feat: add 4-column export tab with burn-in .mp4 download button"
```

---

## Chunk 6: Final verification + PR

### Task 12: Smoke-test the full app

- [ ] **Step 1: Run the full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests PASS, zero failures.

- [ ] **Step 2: Start the app and verify manually**

```bash
streamlit run subgen_ai/app.py
```

Open http://localhost:8501 and verify:
- [ ] Upload a short video (< 80 MB `.mp4`)
- [ ] Click "▶ Generate Subtitles" — transcription runs
- [ ] Review tab loads with video player at the top showing subtitle overlay
- [ ] GREEN segments show an editable text area and "💾 Validate & Save Correction" button
- [ ] Export tab shows 4 columns: 🔥 Burn-in | ⬇ SRT | ⬇ VTT | ⬇ JSON
- [ ] Click "🔥 Generate Burn-in .mp4" — spinner appears, download button appears
- [ ] Upload an audio-only `.wav` — Export tab shows the info message instead of the burn-in button

- [ ] **Step 3: Push and open PR**

```bash
git push origin claude/eloquent-benz
gh pr create \
  --title "feat: video player, universal segment editing, and burn-in export" \
  --body "Adds HTML5 video player with WebVTT subtitle overlay to the Review tab, makes all segments (GREEN and RED) editable with the same DB save flow, and adds a burn-in .mp4 export alongside the existing SRT/VTT/JSON downloads."
```
