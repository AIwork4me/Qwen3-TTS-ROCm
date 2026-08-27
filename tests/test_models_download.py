"""Task 5: dual-source downloader tests (TDD step 1).

All tests are hermetic: the network layer is faked via the ``_ms_snapshot`` /
``_hf_snapshot`` indirection points, and every test pins the models root to
``tmp_path`` through the ``models_dir`` override (never real network, never
touching the repo's own ``models/`` checkout).
"""

import os
from typing import ClassVar

import pytest

from qwen3_tts_rocm import models

TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


class FakeMS:
    calls: ClassVar[list] = []

    @classmethod
    def snapshot_download(cls, model_id, local_dir=None, **kw):
        cls.calls.append(("ms", model_id))
        import pathlib

        pathlib.Path(local_dir, ".ok").write_text("ok")
        return local_dir


class FakeMS_Fail(FakeMS):
    @classmethod
    def snapshot_download(cls, model_id, **kw):
        cls.calls.append(("ms-fail", model_id))
        raise OSError("network down")


def test_download_single(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    monkeypatch.setattr(M, "_ms_snapshot", FakeMS.snapshot_download)
    paths = M.download("tokenizer", source="modelscope", models_dir=tmp_path)
    assert paths[0].exists() and (paths[0] / ".ok").exists()
    assert FakeMS.calls[-1][1] == TOKENIZER_REPO
    assert paths == [tmp_path / "Qwen3-TTS-Tokenizer-12Hz"]


def test_download_auto_falls_back_to_hfmirror(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    calls = []
    monkeypatch.setattr(
        M, "_ms_snapshot", lambda mid, **kw: (_ for _ in ()).throw(OSError("down"))
    )
    monkeypatch.setattr(
        M,
        "_hf_snapshot",
        lambda rid, d, **kw: (calls.append(rid), M.mark_ok(d))[1],
    )
    out = M.download(["tokenizer"], source="auto", models_dir=tmp_path)
    assert calls == [TOKENIZER_REPO] and (out[0] / ".ok").exists()


def test_skip_already_downloaded(monkeypatch, tmp_path):
    """models_dir override must win over env/cwd roots; a marked target is skipped."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / ".ok").write_text("")

    def boom(*a, **k):
        raise AssertionError("should not be called")

    monkeypatch.setattr(M, "_ms_snapshot", boom)
    monkeypatch.setattr(M, "_hf_snapshot", boom)
    # No QWEN3_TTS_ROCM_MODELS_DIR pin here on purpose: with cwd inside the
    # project checkout, local_dir() resolves to <cwd>/models (where a real,
    # already-downloaded tokenizer lives).  The explicit models_dir must take
    # precedence so we skip *this* directory, not the ambient one.
    assert M.download("tokenizer", models_dir=tmp_path) == [d]


# ---------------------------------------------------------------------------
# Supporting coverage for the remaining download() contract (still hermetic).
# ---------------------------------------------------------------------------

def test_supported_sources_constant():
    import qwen3_tts_rocm.models as M

    assert M.SUPPORTED_SOURCES == ("modelscope", "hfmirror")


def test_download_all_returns_every_alias_in_order(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    monkeypatch.setattr(M, "_ms_snapshot", FakeMS.snapshot_download)
    out = M.download("all", source="modelscope", models_dir=tmp_path)
    expected_dirs = [
        tmp_path / models.flatten(repo) for repo in models.REPOS.values()
    ]
    assert out == expected_dirs
    assert all((p / ".ok").is_file() for p in out)


def test_explicit_modelscope_failure_does_not_fallback(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    def hf_boom(*a, **k):
        raise AssertionError("hf fallback must not run for an explicit source")

    monkeypatch.setattr(M, "_ms_snapshot", FakeMS_Fail.snapshot_download)
    monkeypatch.setattr(M, "_hf_snapshot", hf_boom)
    FakeMS_Fail.calls.clear()
    with pytest.raises(RuntimeError) as excinfo:
        M.download("tokenizer", source="modelscope", models_dir=tmp_path)
    assert len(FakeMS_Fail.calls) == 2  # initial attempt + one retry
    assert "https://modelscope.cn/models/" + TOKENIZER_REPO in str(excinfo.value)
    assert "https://hf-mirror.com/" + TOKENIZER_REPO in str(excinfo.value)


def test_auto_exhaustion_raises_runtime_with_both_urls(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    monkeypatch.setattr(
        M, "_ms_snapshot", lambda mid, **kw: (_ for _ in ()).throw(OSError("ms down"))
    )
    monkeypatch.setattr(
        M, "_hf_snapshot", lambda rid, d, **kw: (_ for _ in ()).throw(OSError("hf down"))
    )
    with pytest.raises(RuntimeError) as excinfo:
        M.download("tokenizer", source="auto", models_dir=tmp_path)
    msg = str(excinfo.value)
    assert "https://modelscope.cn/models/" + TOKENIZER_REPO in msg
    assert "https://hf-mirror.com/" + TOKENIZER_REPO in msg
    assert "modelscope" in msg and "hfmirror" in msg  # both attempts reported


def test_invalid_source_rejected(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    with pytest.raises(ValueError, match="source"):
        M.download("tokenizer", source="magic-ftp", models_dir=tmp_path)


def test_unknown_alias_rejected(tmp_path):
    import qwen3_tts_rocm.models as M

    with pytest.raises(KeyError, match="unknown"):
        M.download("not-an-alias", models_dir=tmp_path)


def test_hf_snapshot_sets_hf_mirror_endpoint_and_restores_env(monkeypatch, tmp_path):
    import huggingface_hub

    import qwen3_tts_rocm.models as M

    seen = {}

    def fake_snapshot(repo_id, local_dir=None, **kw):
        seen["endpoint"] = os.environ.get("HF_ENDPOINT")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    monkeypatch.setenv("HF_ENDPOINT", "https://original.example")
    M._hf_snapshot("Qwen/Foo", tmp_path)
    assert seen["endpoint"] == "https://hf-mirror.com"
    assert os.environ.get("HF_ENDPOINT") == "https://original.example"

    monkeypatch.delenv("HF_ENDPOINT")
    seen.clear()
    M._hf_snapshot("Qwen/Foo", tmp_path)
    assert seen["endpoint"] == "https://hf-mirror.com"
    assert "HF_ENDPOINT" not in os.environ


def test_resume_flag_accepted(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    monkeypatch.setattr(M, "_ms_snapshot", FakeMS.snapshot_download)
    out = M.download("tokenizer", source="modelscope", models_dir=tmp_path, resume=True)
    assert len(out) == 1 and (out[0] / ".ok").exists()
