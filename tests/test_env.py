# tests/test_env.py
import sys, types, pytest
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
