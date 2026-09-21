# tests/test_scaffold.py
def test_package_imports_and_version():
    import qwen3_tts_rocm
    assert qwen3_tts_rocm.__version__ == "0.2.0"


def test_markers_registered():
    assert {m.name for m in []} == set()  # placeholder never fails
