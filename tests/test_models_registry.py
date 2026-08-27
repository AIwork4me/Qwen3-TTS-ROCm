"""Task 4: model registry & path resolution tests (TDD step 1)."""


import pytest

from qwen3_tts_rocm import models

EXPECTED = {
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom-voice-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

def test_registry_complete():
    assert tuple(models.REPOS.items()) == tuple(EXPECTED.items())
    assert set(models.ALIASES) == set(EXPECTED)

def test_local_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    d = models.local_dir("base")
    assert d == tmp_path / "Qwen3-TTS-12Hz-1.7B-Base"

def test_resolve_path_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    made = tmp_path / "MyLocalCopy"; made.mkdir()
    assert models.resolve_path(made) == made                      # existing dir wins
    assert models.resolve_path("base").name == "Qwen3-TTS-12Hz-1.7B-Base"
    assert models.resolve_path("Qwen3-TTS-12Hz-1.7B-Base").name == "Qwen3-TTS-12Hz-1.7B-Base"
    with pytest.raises(KeyError, match="unknown"):
        models.resolve_path("nope-nope")


# ---------------------------------------------------------------------------
# Supplementary coverage for the remaining public names (still hermetic:
# every test pins QWEN3_TTS_ROCM_MODELS_DIR to tmp_path, no network access).
# ---------------------------------------------------------------------------

def test_aliases_tuple_content():
    assert models.ALIASES == (
        "tokenizer",
        "voice-design",
        "custom-voice",
        "base",
        "custom-voice-0.6b",
        "base-0.6b",
    )

def test_local_dir_none_returns_root(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    assert models.local_dir() == tmp_path

def test_flatten_strips_single_namespace():
    assert models.flatten("Qwen/Foo") == "Foo"
    assert models.flatten("Foo") == "Foo"
    assert models.flatten("Org/Sub/Foo") == "Sub/Foo"

def test_alias_of_repo_roundtrip():
    for alias, repo in models.REPOS.items():
        assert models.alias_of_repo(repo) == alias              # full "Qwen/..." id
        assert models.alias_of_repo(models.flatten(repo)) == alias  # flattened name

def test_alias_of_repo_unknown_raises():
    with pytest.raises(KeyError, match="unknown"):
        models.alias_of_repo("SomeOther/Nobody-Knows-Me")

def test_resolve_full_repo_id(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    got = models.resolve_path("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    assert got == tmp_path / "Qwen3-TTS-12Hz-0.6B-CustomVoice"

def test_resolve_existing_dir_string_form(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    made = tmp_path / "StrCopy"; made.mkdir()
    assert models.resolve_path(str(made)) == made                # str passthrough too

def test_resolve_error_message_bilingual(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    with pytest.raises(KeyError, match="未知的模型引用"):
        models.resolve_path("nope-nope")

def test_is_downloaded_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    assert models.is_downloaded("base") is False                 # missing dir
    d = models.local_dir("base")
    d.mkdir(parents=True)
    (d / "README.md").touch()
    assert models.is_downloaded("base") is False                 # unrelated file is not enough
    (d / "config.json").touch()
    assert models.is_downloaded("base") is True                  # config.json present
    (d / "config.json").unlink()
    (d / ".ok").touch()
    assert models.is_downloaded("base") is True                  # .ok marker suffices
    assert models.is_downloaded("definitely-not-an-alias") is False

def test_mark_ok_creates_marker_and_enables_download_check(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    models.mark_ok("voice-design")
    assert (tmp_path / "Qwen3-TTS-12Hz-1.7B-VoiceDesign" / ".ok").is_file()
    assert models.is_downloaded("voice-design") is True
    made = tmp_path / "LocalMade"; made.mkdir()
    models.mark_ok(made)                                          # existing-dir refs allowed
    assert (made / ".ok").is_file()

def test_mark_ok_rejects_unknown_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    with pytest.raises(KeyError, match="unknown"):
        models.mark_ok(tmp_path / "Does-Not-Exist")
