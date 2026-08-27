"""Task 5: dual-source downloader tests (TDD step 1).

All tests are hermetic: the network layer is faked via the ``_ms_snapshot`` /
``_hf_snapshot`` indirection points, and every test pins the models root to
``tmp_path`` through the ``models_dir`` override (never real network, never
touching the repo's own ``models/`` checkout).
"""

import os
from pathlib import Path
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
    """models_dir override must win over env/cwd roots; a complete (marked +
    weights) target is skipped."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / ".ok").write_text("")
    (d / "model.safetensors").write_bytes(b"weights")  # S3: skip is weight-aware

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
    assert "docs/troubleshooting.md" in msg  # UX-fix U2: docs pointer


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


def test_resume_false_reinvokes_transport_for_downloaded_alias(monkeypatch, tmp_path):
    """resume=False bypasses the is_downloaded skip: the transport runs again
    for the alias (target dir resolved the same way, .ok rewritten)."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / ".ok").write_text("stale")

    monkeypatch.setattr(M, "_ms_snapshot", FakeMS.snapshot_download)
    FakeMS.calls.clear()
    out = M.download("tokenizer", source="modelscope", models_dir=tmp_path,
                     resume=False)
    assert FakeMS.calls == [("ms", TOKENIZER_REPO)]  # exactly one invocation
    assert out == [d]
    # FakeMS writes "ok", then mark_ok() rewrites .ok with its own marker, so
    # the pre-existing content is guaranteed to be gone after the re-download.
    assert (d / ".ok").read_text() != "stale"


def test_resume_true_keeps_downloaded_skip(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / ".ok").write_text("")
    (d / "model.safetensors").write_bytes(b"weights")  # S3: skip is weight-aware

    def boom(*a, **k):
        raise AssertionError("resume=True must skip an already-downloaded target")

    monkeypatch.setattr(M, "_ms_snapshot", boom)
    monkeypatch.setattr(M, "_hf_snapshot", boom)
    assert M.download("tokenizer", source="modelscope", models_dir=tmp_path,
                      resume=True) == [d]


# ---------------------------------------------------------------------------
# P0-S3 (hardening): weight-aware completeness + pinned auto-fallback-into-
# partial-dir semantics.  Written test-first (see
# .superpowers/sdd/v0.1.0-hardening/s3-report.md).
# ---------------------------------------------------------------------------

def test_is_downloaded_require_weights_rejects_config_only_dir(tmp_path):
    """A config.json-only partial repo still counts as downloaded for the legacy
    default call, but NOT once require_weights=True asks for real weight files."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert M.is_downloaded(d) is True  # default behaviour unchanged
    assert M.is_downloaded(d, require_weights=True) is False


def test_is_downloaded_require_weights_accepts_any_safetensors(tmp_path):
    """require_weights=True wants at least one *.safetensors: model.safetensors
    and sharded names both count; config.json/.ok alone stay required too."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / ".ok").write_text("", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"weights")
    assert M.is_downloaded(d, require_weights=True) is True

    sharded = tmp_path / models.flatten(models.REPOS["base"])
    sharded.mkdir()
    (sharded / "config.json").write_text("{}", encoding="utf-8")
    (sharded / "model-00001-of-00002.safetensors").write_bytes(b"w")
    assert M.is_downloaded(sharded, require_weights=True) is True

    (sharded / "model-00001-of-00002.safetensors").unlink()
    assert M.is_downloaded(sharded, require_weights=True) is False  # weights gone


def test_download_refetches_config_only_partial_repo(monkeypatch, tmp_path):
    """download()'s resume skip is weight-aware: a config.json-only partial repo
    is re-fetched instead of being skipped as already-downloaded."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")

    calls: list = []

    def fake_ms(model_id, local_dir=None, **kw):
        calls.append(("ms", model_id))
        M.mark_ok(local_dir)

    def hf_boom(*a, **k):
        raise AssertionError("hf fallback must not run for an explicit source")

    monkeypatch.setattr(M, "_ms_snapshot", fake_ms)
    monkeypatch.setattr(M, "_hf_snapshot", hf_boom)
    out = M.download("tokenizer", source="modelscope",
                     models_dir=tmp_path)  # resume=True is the default
    assert calls == [("ms", TOKENIZER_REPO)]  # transport ran (was skipped before)
    assert out == [d]
    assert (d / ".ok").exists()


def test_download_refetches_ok_dir_without_weights(monkeypatch, tmp_path):
    """Deliberate S3 tightening: an .ok-only dir whose weights are missing (or
    were deleted after marking) is refetched under resume=True, so a resume
    repairs vanished weights instead of trusting the marker alone."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / ".ok").write_text("")

    monkeypatch.setattr(M, "_ms_snapshot", FakeMS.snapshot_download)
    FakeMS.calls.clear()
    out = M.download("tokenizer", source="modelscope",
                     models_dir=tmp_path)  # resume=True is the default
    assert FakeMS.calls == [("ms", TOKENIZER_REPO)]
    assert out == [d] and (d / ".ok").exists()


def test_download_resume_skips_fully_populated_dir(monkeypatch, tmp_path):
    """config.json + model.safetensors is complete: the resume skip still holds."""
    import qwen3_tts_rocm.models as M

    d = tmp_path / models.flatten(models.REPOS["tokenizer"])
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"weights")

    def boom(*a, **k):
        raise AssertionError("complete target must be skipped under resume=True")

    monkeypatch.setattr(M, "_ms_snapshot", boom)
    monkeypatch.setattr(M, "_hf_snapshot", boom)
    assert M.download("tokenizer", source="modelscope", models_dir=tmp_path,
                      resume=True) == [d]


def test_auto_fallback_into_partial_dir_recovers_and_marks_ok(monkeypatch, tmp_path):
    """Pins the auto fallback into a partially-written directory (Task-5 review
    watch-item): a ModelScope attempt that dies mid-transfer leaves partial files
    behind, the hf-mirror fallback then succeeds into the SAME directory, the
    mixed dir is accepted and marked, and a later resume=True download skips it."""
    import qwen3_tts_rocm.models as M

    def failing_ms(model_id, local_dir=None, **kw):
        # dies AFTER writing a partial file: config.json present, no .ok yet
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        (Path(local_dir) / "model.safetensors.part").write_bytes(b"partial")
        raise OSError("ms died mid-transfer")

    hf_calls: list = []

    def hf_ok(repo_id, local_dir, **kw):
        hf_calls.append(repo_id)
        # a successful hf-mirror snapshot of these repos always carries
        # model.safetensors (Task-8: present in all six); it doubles as the
        # marker proving hf ran and satisfies the weight-aware resume skip below
        (Path(local_dir) / "model.safetensors").write_bytes(b"weights")
        M.mark_ok(local_dir)

    monkeypatch.setattr(M, "_ms_snapshot", failing_ms)
    monkeypatch.setattr(M, "_hf_snapshot", hf_ok)
    out = M.download("tokenizer", source="auto", models_dir=tmp_path)
    d = out[0]
    assert hf_calls == [TOKENIZER_REPO]  # recovered via hfmirror
    assert (d / ".ok").exists()  # dir left marked, no exception escaped
    # BOTH sources' artifacts coexist in the accepted mixed dir:
    assert (d / "config.json").is_file()  # from the failed ms attempt
    assert (d / "model.safetensors.part").is_file()  # ms partial leftover
    assert (d / "model.safetensors").is_file()  # from the hf fallback
    # subsequent resume download skips the mixed (now marked) dir:
    def boom(*a, **k):
        raise AssertionError("marked mixed dir must be skipped under resume=True")

    monkeypatch.setattr(M, "_ms_snapshot", boom)
    monkeypatch.setattr(M, "_hf_snapshot", boom)
    assert M.download("tokenizer", source="auto", models_dir=tmp_path) == [d]
