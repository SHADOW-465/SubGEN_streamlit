"""
SubGEN AI — Export Formatters.

Converts a list of SubtitleSegment into:
  - SRT (SubRip Text)
  - WebVTT
  - JSON (with QC metadata)
"""
import json
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
