# tests/test_env.py
import sys
import types

import pytest

from qwen3_tts_rocm import env


def make_fake_cuda(unavailable=False):
    fake = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: not unavailable,
        get_device_count=lambda: 0 if unavailable else 1,
        get_device_properties=lambda i: types.SimpleNamespace(
            name="AMD Radeon Graphics", gcnArchName="gfx1151", multi_processor_count=40),
        get_device_name=lambda i: "AMD Radeon Graphics",
    )
    fake.cuda = cuda; fake.version = types.SimpleNamespace(hip="7.14.0-something")
    return fake

def test_collect_reports_hip_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda())
    r = env.collect()
    assert r.hip_available is True and r.rocm_version.startswith("7.14")
    assert r.gpus[0].arch == "gfx1151" and r.errors == []

def test_collect_cpu_fallback_no_errors(monkeypatch):
    m = make_fake_cuda(unavailable=True)
    del m.version.hip  # simulate non-HIP torch
    monkeypatch.setitem(sys.modules, "torch", m)
    r = env.collect()
    assert r.hip_available is False
    assert any("AMD index" in e or "AMD" in e for e in r.errors)

def test_pick_device_auto(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda())
    assert env.pick_device("auto") == "cuda:0"
    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda(unavailable=True))
    assert env.pick_device("auto") == "cpu"
    assert env.pick_device("cuda:1") == "cuda:1"   # explicit passthrough

def test_require_rocm_torch_raises_with_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda(unavailable=True))
    with pytest.raises(RuntimeError, match="repo.amd.com"):
        env.require_rocm_torch()

def test_cli_check_exits_zero(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda())
    assert env.rocm_check(verbose=False).hip_available is True

# --- fix round 1 ---

def test_collect_never_raises_when_torch_version_access_raises(monkeypatch):
    class HostileVersion:
        def __getattr__(self, name):
            raise RuntimeError("simulated shim failure")  # non-AttributeError escape

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_count=lambda: 0)
    fake.version = HostileVersion()
    monkeypatch.setitem(sys.modules, "torch", fake)
    r = env.collect()  # must NOT raise
    assert r.hip_available is False

def test_no_apu_hint_when_cuda_ok_but_no_gpus(monkeypatch):
    m = make_fake_cuda(unavailable=True)
    m.cuda.is_available = lambda: True  # force cuda_ok=True while get_device_count()==0
    monkeypatch.setitem(sys.modules, "torch", m)
    r = env.collect()
    assert r.gpus == []
    assert not any("unified-memory APU" in w for w in r.warnings)


# --- UX-fix U2: startup honesty about models + docs pointers ---

def test_collect_counts_downloaded_models_into_info_line(monkeypatch):
    """3 of 6 registry aliases downloaded -> one INFO line in warnings; errors empty."""
    from qwen3_tts_rocm import models

    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda())
    downloaded = {"tokenizer", "custom-voice", "base"}
    monkeypatch.setattr(models, "is_downloaded", lambda ref, **kw: str(ref) in downloaded)
    r = env.collect()
    assert r.errors == []
    assert any(
        w.startswith("INFO:") and "models: 3/6 downloaded" in w for w in r.warnings
    )


def test_rocm_check_prints_troubleshooting_tail_once(monkeypatch, capsys):
    """Verbose self-check ends with ONE generic pointer to docs/troubleshooting.md."""
    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda())
    env.rocm_check(verbose=True)
    out = capsys.readouterr().out
    assert out.count("docs/troubleshooting.md") == 1
    assert "更多排障步骤" in out


# --- regression (2026-08 first-user journey, F2): summary counts -------------

def test_summary_counts_only_warn_lines_as_warnings(monkeypatch, capsys):
    """INFO advisories (unified-memory note, models line) ride in the same
    list as real WARN: lines; the summary used to count them all as
    "warnings", so a healthy host printed "warnings: 2" with no WARN line in
    sight.  WARN and INFO must now be reported separately."""
    from qwen3_tts_rocm import models

    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda())
    monkeypatch.setattr(models, "is_downloaded", lambda ref, **kw: False)
    r = env.collect()
    # sanity: the healthy-host shape — advisories only, zero real warnings
    assert any(w.startswith("INFO:") for w in r.warnings)
    assert not any(w.startswith("WARN:") for w in r.warnings)
    env.rocm_check(verbose=True)
    out = capsys.readouterr().out
    assert "warnings: 0" in out
    assert f"notes: {sum(1 for w in r.warnings if w.startswith('INFO:'))}" in out
    assert "警告 0 条" in out and "提示" in out


def test_summary_still_counts_real_warn_lines(monkeypatch, capsys):
    """A genuine WARN: advisory must keep incrementing the warnings counter."""
    monkeypatch.setitem(sys.modules, "torch", make_fake_cuda(unavailable=True))
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")  # emits a WARN: line
    env.rocm_check(verbose=True)
    out = capsys.readouterr().out
    assert "warnings: 1" in out
