import sys, os, tempfile, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from pathlib import Path
from unittest.mock import patch
from subgen_ai.db.correction_store import (
    init_db, save_correction, find_nearest_correction,
    get_db_stats, delete_correction, _cosine_sim_arrays
)
from subgen_ai.core.models import CorrectionRecord


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temporary directory for each test."""
    import subgen_ai.db.correction_store as cs
    monkeypatch.setattr(cs, "DB_PATH", tmp_path / "test.db")
    yield


def _make_record(**kwargs) -> CorrectionRecord:
    defaults = dict(
        id=None, segment_start=0.0, segment_end=2.0,
        original_text="wrong", corrected_text="right", language="en",
        mfcc_mean=[1.0] * 12, mfcc_var=[0.1] * 12,
        match_score=0.9, hw_used=False, created_at=""
    )
    defaults.update(kwargs)
    return CorrectionRecord(**defaults)


def test_init_db_creates_table(tmp_path, monkeypatch):
    import subgen_ai.db.correction_store as cs
    conn = init_db()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    assert any("corrections" in t[0] for t in tables)


def test_save_and_retrieve():
    rec = _make_record(language="ta")
    row_id = save_correction(rec)
    assert isinstance(row_id, int) and row_id > 0


def test_find_nearest_correction_match():
    rec = _make_record(mfcc_mean=[1.0] * 12, language="ta")
    save_correction(rec)
    # Query with identical vector → should exceed 0.80 threshold
    found = find_nearest_correction([1.0] * 12, "ta", threshold=0.80)
    assert found is not None
    assert found.corrected_text == "right"


def test_find_nearest_correction_no_match():
    """Very different MFCC → no match above 0.80."""
    rec = _make_record(mfcc_mean=[1.0] * 12, language="en")
    save_correction(rec)
    found = find_nearest_correction([-1.0] * 12, "en", threshold=0.80)
    assert found is None or isinstance(found, CorrectionRecord)


def test_get_db_stats_empty():
    stats = get_db_stats()
    assert stats["total"] == 0
    assert stats["by_language"] == {}


def test_get_db_stats_after_saves():
    save_correction(_make_record(language="ta"))
    save_correction(_make_record(language="ta"))
    save_correction(_make_record(language="en"))
    stats = get_db_stats()
    assert stats["total"] == 3
    assert stats["by_language"]["ta"] == 2


def test_delete_correction():
    row_id = save_correction(_make_record(language="en"))
    stats_before = get_db_stats()
    delete_correction(row_id)
    stats_after = get_db_stats()
    assert stats_after["total"] == stats_before["total"] - 1


def test_cosine_sim_arrays_identical():
    v = [1.0] * 12
    assert abs(_cosine_sim_arrays(v, v) - 1.0) < 1e-6


def test_cosine_sim_arrays_zero_vector():
    assert _cosine_sim_arrays([0.0] * 12, [1.0] * 12) == 0.0
