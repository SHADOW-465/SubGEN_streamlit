# tests/test_utils.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_corrections_file_is_absolute():
    """CORRECTIONS_FILE must be an absolute path based on SubNXT.py location."""
    subgen_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SubNXT.py")
    with open(subgen_path, "r", encoding="utf-8") as f:
        content = f.read()
    # After the fix, it must NOT be a bare string literal "corrections.json"
    assert 'CORRECTIONS_FILE = "corrections.json"' not in content, \
        "CORRECTIONS_FILE must not be a bare relative path"
    assert "__file__" in content or "os.path.dirname" in content, \
        "CORRECTIONS_FILE must use __file__-based absolute path"


def test_jetbrains_mono_imported():
    """CSS must import JetBrains Mono, not just reference it."""
    subgen_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SubNXT.py")
    with open(subgen_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "fonts.googleapis.com" in content
    # The import line must include JetBrains+Mono
    import_lines = [l for l in content.splitlines() if "@import" in l and "fonts.googleapis" in l]
    assert any("JetBrains" in l for l in import_lines), \
        "JetBrains Mono must be included in the Google Fonts @import"
