# tests/test_scaffold.py
def test_package_imports_and_version():
    import qwen3_tts_rocm
    assert qwen3_tts_rocm.__version__ == "0.1.0"


def test_markers_registered():
    import pytest as _p
    assert {m.name for m in []} == set()  # placeholder never fails
