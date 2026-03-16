"""
SubGEN AI — Export Formatters.

Converts a list of SubtitleSegment into:
  - SRT (SubRip Text)
  - WebVTT
  - JSON (with QC metadata)
"""
import json
import os
import tempfile
from typing import List

from subgen_ai.core.models import SubtitleSegment


def _format_ts_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp: HH:MM:SS.mmm"""
    return _format_ts_srt(seconds).replace(",", ".")


def to_srt(segments: List[SubtitleSegment]) -> str:
    """
    Convert segments to SRT format.

    Each block:
        <index>
        HH:MM:SS,mmm --> HH:MM:SS,mmm
        <text>
        <blank line>
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_ts_srt(seg.start)} --> {_format_ts_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: List[SubtitleSegment]) -> str:
    """
    Convert segments to WebVTT format.

    Starts with 'WEBVTT' header, then timestamp blocks with dot separator.
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_ts_vtt(seg.start)} --> {_format_ts_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def to_json(segments: List[SubtitleSegment]) -> str:
    """
    Convert segments to JSON with full QC metadata.

    Output includes per-segment quality signals (label, fused_conf, asr_conf,
    snr_db, hw_fingerprint) and correction status.
    Indic/Unicode text is preserved unescaped (ensure_ascii=False).
    """
    data = []
    for seg in segments:
        data.append({
            "index":    seg.index,
            "start":    round(seg.start, 3),
            "end":      round(seg.end, 3),
            "text":     seg.text,
            "language": seg.language,
            "quality": {
                "label":         seg.label,
                "fused_conf":    round(seg.fused_conf, 4),
                "asr_conf":      round(seg.asr_conf, 4),
                "snr_db":        round(seg.snr_db, 2),
                "hw_fingerprint": seg.hw_fingerprint,
            },
            "corrected": seg.corrected,
        })
    return json.dumps({"segments": data}, ensure_ascii=False, indent=2)


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
