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
