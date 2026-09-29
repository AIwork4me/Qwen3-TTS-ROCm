"""Architecture-contract regression tests (v0.3 Task 4).

CPU-runnable coverage for the multi-architecture contract implemented in
``scripts/install.sh`` (bash twin table), ``scripts/verify_gpu.sh`` (exact
arch matching) and ``src/qwen3_tts_rocm/gfx.py`` (canonical table):

- gfx1100 / gfx1151 mapping and install-command generation;
- invalid targets and family-pattern shortcuts rejected;
- fail-closed auto detection (no tools / no agent / ambiguity);
- no silent fallback (no CUDA/PyPI index anywhere);
- shell syntax of both scripts;
- pinned versions everywhere the wheel command is constructed;
- verify_gpu.sh arch logic (simulated; CODE LOGIC ONLY — the gfx1151
  simulations are NOT hardware evidence and must never be presented as
  such);
- validation-state classification wording (validated / historically
  validated / not validated);
- the gfx1100 dGPU is not misadvised as a unified-memory APU.
"""

import os
import pathlib
import re
import subprocess
import sys
import types

import pytest

from qwen3_tts_rocm import gfx

REPO = pathlib.Path(__file__).resolve().parents[1]
INSTALL_SH = REPO / "scripts" / "install.sh"
VERIFY_SH = REPO / "scripts" / "verify_gpu.sh"


def _install_text() -> str:
    return INSTALL_SH.read_text(encoding="utf-8")


def _verify_text() -> str:
    return VERIFY_SH.read_text(encoding="utf-8")


# --- canonical mapping -------------------------------------------------------


def test_gfx1100_mapping():
    t = gfx.TARGETS["gfx1100"]
    assert t.extras == "device-gfx1100"
    assert t.validation_state == gfx.VALIDATION_CURRENT
    assert gfx.KNOWN_TARGETS == {"gfx1100", "gfx1151"}


def test_gfx1151_mapping():
    t = gfx.TARGETS["gfx1151"]
    assert t.extras == "device-gfx1151"
    assert t.validation_state == gfx.VALIDATION_HISTORICAL


def test_bash_table_in_install_sh_matches_python():
    """The literal bash case table must agree with gfx.TARGETS (no drift)."""
    text = _install_text()
    block = re.search(
        r">>> gfx-target-table(.*?)# <<< gfx-target-table", text, re.DOTALL
    )
    assert block, "gfx-target-table markers missing from install.sh"
    for name, spec in gfx.TARGETS.items():
        assert re.search(
            rf"{name}\) echo \"{spec.extras}\"", block.group(1)
        ), f"install.sh case table lacks {name} -> {spec.extras}"


# --- install command generation ---------------------------------------------


@pytest.mark.parametrize("name", ["gfx1100", "gfx1151"])
def test_install_command_generation(name):
    argv = gfx.pip_install_argv(name)
    joined = " ".join(argv)
    extras = gfx.TARGETS[name].extras
    assert f"torch[{extras}]=={gfx.TORCH_PIN}" in joined
    assert f"torchvision[{extras}]=={gfx.TORCHVISION_PIN}" in joined
    assert f"torchaudio=={gfx.TORCHAUDIO_PIN}" in joined
    assert "device-gfx" in joined
    # torchaudio ships NO device extra on the amd index
    assert f"torchaudio[{extras}]" not in joined


def test_install_command_generation_rejects_unknown():
    with pytest.raises(KeyError):
        gfx.pip_install_argv("gfx1103")


def test_pinned_versions_and_index_in_install_sh():
    line = next(
        l for l in _install_text().splitlines() if "whl-multi-arch" in l and "pip install" in l
    )
    assert gfx.AMD_ROCM_INDEX in line
    for pin in (gfx.TORCH_PIN, gfx.TORCHVISION_PIN, gfx.TORCHAUDIO_PIN):
        assert pin in line
    # the extras must come from the resolved variable, never a hard-coded arch
    assert "${GFX_EXTRAS}" in line
    assert "device-gfx1151]==" not in line and "device-gfx1100]==" not in line


def test_no_silent_fallback_indexes():
    for text in (_install_text(), _verify_text()):
        low = text.lower()
        assert "pypi.org" not in low, "PyPI fallback would break the pinned contract"
        assert "download.pytorch.org" not in low
        assert "extra-index" not in low


# --- shell syntax ------------------------------------------------------------


def test_install_sh_shell_syntax():
    subprocess.run(["bash", "-n", str(INSTALL_SH)], check=True)


def test_verify_gpu_sh_shell_syntax():
    subprocess.run(["bash", "-n", str(VERIFY_SH)], check=True)


# --- invalid / family-pattern targets (fast-fail, no side effects) ----------


@pytest.mark.parametrize(
    "bad", ["gfx1200", "gfx11", "gfx1103", "gfx1100x", "GFX1100", "", "auto=", "cuda"]
)
def test_invalid_targets_exit_2_before_any_work(bad):
    proc = subprocess.run(
        ["bash", str(INSTALL_SH), "--gfx-target", bad],
        capture_output=True, text=True, cwd="/tmp", timeout=60, check=False,
    )
    assert proc.returncode == 2, (bad, proc.stdout, proc.stderr)
    assert "supported" in proc.stderr or "gfx1100 | gfx1151 | auto" in proc.stderr
    # fast-fail proof: no venv/pip stage may run
    assert "PYSTUB" not in proc.stdout and "pip" not in proc.stdout


def test_no_family_pattern_shortcut_in_scripts():
    # comments may DISCUSS the forbidden pattern; executable lines may not use it
    for path in (INSTALL_SH, VERIFY_SH):
        code_lines = [
            l for l in path.read_text(encoding="utf-8").splitlines()
            if not l.lstrip().startswith("#")
        ]
        joined = "\n".join(code_lines)
        assert "gfx11*" not in joined, f"family-pattern shortcut in {path.name} code"


# --- fail-closed auto detection ---------------------------------------------

_STUBS = pathlib.Path("/tmp/opencode/test-stubs")


def _stub_dir(name: str, enumerator_output: str) -> str:
    d = _STUBS / name
    d.mkdir(parents=True, exist_ok=True)
    stub = d / "rocm_agent_enumerator"
    stub.write_text("#!/bin/sh\nprintf '%s'\n", encoding="utf-8")
    stub.write_text(f"#!/bin/sh\nprintf '{enumerator_output}'\n", encoding="utf-8")
    stub.chmod(0o755)
    return str(d)


def _run_install_auto(stub_path: str):
    env = {
        "PATH": f"{stub_path}:/usr/bin:/bin",
        "HOME": "/tmp",
    }
    return subprocess.run(
        ["bash", str(INSTALL_SH), "--gfx-target", "auto"],
        capture_output=True, text=True, env=env, cwd="/tmp",
        timeout=120, check=False,
    )


def test_auto_ambiguous_architectures_fail_closed():
    stub = _stub_dir("ambiguous", "gfx1100 gfx1151")
    proc = _run_install_auto(stub)
    assert proc.returncode == 1
    assert "ambiguous" in proc.stderr or "multiple distinct" in proc.stderr
    assert "--gfx-target" in proc.stderr  # remediation hint


def test_auto_no_visible_agent_fails_closed():
    stub = _stub_dir("noagent", "")
    proc = _run_install_auto(stub)
    assert proc.returncode == 1
    assert "no visible compute agent" in proc.stderr


def test_auto_unknown_single_arch_fails_closed():
    stub = _stub_dir("unknown", "gfx1103")
    proc = _run_install_auto(stub)
    assert proc.returncode == 1
    assert "not a validated target" in proc.stderr
    # explicitly not an unsupported-hardware claim
    assert "NOT a claim that your GPU is unsupported" in proc.stderr


# --- validation-state classification ----------------------------------------


def test_classification_validated_current():
    line = gfx.classification_line("gfx1100")
    assert line.startswith("validated architecture: gfx1100")


def test_classification_historical_gfx1151_scope_truthful():
    line = gfx.classification_line("gfx1151")
    assert line.startswith("historically validated architecture: gfx1151")
    assert "not rerun on gfx1151 hardware during v0.3" in line


def test_classification_unknown_never_validated():
    line = gfx.classification_line("gfx1103")
    assert line == "ROCm-visible but not validated architecture: gfx1103"
    assert not line.startswith("validated architecture")
    assert not line.startswith("historically validated architecture")


def test_apu_classification_keeps_gfx1100_discrete():
    assert gfx.is_apu_arch("gfx1151") and gfx.is_apu_arch("gfx1150")
    assert not gfx.is_apu_arch("gfx1100")
    assert not gfx.is_apu_arch(None)


def test_env_does_not_misadvise_gfx1100_dgpu_as_apu(monkeypatch):
    from qwen3_tts_rocm import env

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_count=lambda: 1,
        get_device_properties=lambda i: types.SimpleNamespace(
            name="AMD Radeon Pro W7900D", gcnArchName="gfx1100",
            multi_processor_count=96,
        ),
        get_device_name=lambda i: "AMD Radeon Pro W7900D",
    )
    fake.version = types.SimpleNamespace(hip="7.14.60850")
    monkeypatch.setitem(sys.modules, "torch", fake)
    report = env.collect()
    assert report.errors == []
    assert not any("unified-memory APU" in w for w in report.warnings)


def test_env_keeps_apu_advisory_for_gfx1151(monkeypatch):
    from qwen3_tts_rocm import env

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_count=lambda: 1,
        get_device_properties=lambda i: types.SimpleNamespace(
            name="AMD Radeon 8060S Graphics", gcnArchName="gfx1151",
            multi_processor_count=40,
        ),
        get_device_name=lambda i: "AMD Radeon 8060S Graphics",
    )
    fake.version = types.SimpleNamespace(hip="7.14.60850")
    monkeypatch.setitem(sys.modules, "torch", fake)
    report = env.collect()
    assert any("unified-memory APU" in w for w in report.warnings)


# --- verify_gpu.sh arch logic (SIMULATED — CODE LOGIC ONLY) ------------------
# The fake torch below exercises the exact heredoc body extracted from
# scripts/verify_gpu.sh. gfx1151 cases prove the CODE LOGIC accepts the
# historically-validated target; they are NOT gfx1151 hardware evidence.




def _verify_body() -> str:
    text = _verify_text()
    match = re.search(r"<<'EOF'\n(.*?)\nEOF\n", text, re.DOTALL)
    assert match, "verify_gpu.sh heredoc not found"
    return match.group(1)


def _run_verify_simulated(arch: str, target_env: str, tmp_path):
    """Run the extracted verify_gpu.sh heredoc body in a subprocess whose
    ``torch``/``torchaudio`` are fakes built in-process (arch-controlled).

    The child imports the REAL ``qwen3_tts_rocm.gfx`` contract and exercises
    the exact verify logic against the fake stack — CODE LOGIC ONLY."""

    preamble = f'''
import sys, types
sys.path.insert(0, {str(REPO / "src")!r})
arch = {arch!r}
class _Finite:
    def __matmul__(self, other):
        return _FiniteResult()
class _FiniteResult(_Finite):
    pass
fake = types.ModuleType("torch")
fake.__version__ = "2.12.0+rocm7.14.0-fake"
fake.version = types.SimpleNamespace(hip="7.14.60850-fake")
fake.bfloat16 = "bf16-sentinel"
fake.cuda = types.SimpleNamespace(
    is_available=lambda: True,
    get_device_properties=lambda i: types.SimpleNamespace(
        name="Fake GPU", gcnArchName=arch),
    get_device_name=lambda i: "Fake GPU",
)
fake.randn = lambda *a, **k: _Finite()
_F = types.ModuleType("torch.nn.functional")
_F.scaled_dot_product_attention = staticmethod(lambda q, k, v: _FiniteResult())
fake.nn = types.SimpleNamespace(functional=_F)
fake.isfinite = lambda x: types.SimpleNamespace(
    all=lambda: types.SimpleNamespace(item=lambda: True))
sys.modules["torch"] = fake
sys.modules["torch.nn.functional"] = _F
audio = types.ModuleType("torchaudio")
audio.__version__ = "2.11.0+rocm7.14.0-fake"
sys.modules["torchaudio"] = audio
'''
    code = preamble + _verify_body()
    env = dict(os.environ)
    env["QWEN3_TTS_ROCM_GFX_TARGET"] = target_env
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, check=False,
    )


@pytest.mark.parametrize("target,arch,expect_ok", [
    ("auto", "gfx1100", True),            # real class: validated
    ("auto", "gfx1151", True),            # CODE LOGIC: historical acceptance
    ("gfx1100", "gfx1100", True),
    ("gfx1151", "gfx1151", True),         # CODE LOGIC ONLY (no hardware claim)
    ("gfx1151", "gfx1100", False),        # mismatch must fail
    ("gfx1100", "gfx1151", False),        # mismatch must fail
    ("auto", "gfx1103", False),           # unknown never validated
    ("gfx1200", "gfx1100", False),        # invalid target value
])
def test_verify_gpu_arch_logic_simulated(target, arch, expect_ok, tmp_path):
    proc = _run_verify_simulated(arch, target, tmp_path)
    assert (proc.returncode == 0) == expect_ok, (proc.stdout, proc.stderr)
    if expect_ok:
        assert "SPIKE-GPU-OK" in proc.stdout
    if arch == "gfx1151" and expect_ok:
        assert "historically validated architecture: gfx1151" in proc.stdout
    if arch == "gfx1100" and expect_ok:
        assert "validated architecture: gfx1100" in proc.stdout
    if arch == "gfx1103":
        assert "not validated" in proc.stderr or "not validated" in proc.stdout


def test_verify_gpu_mismatch_message_is_configuration_error(tmp_path):
    proc = _run_verify_simulated("gfx1151", "gfx1100", tmp_path)
    assert proc.returncode == 1
    assert "mismatch" in proc.stderr
    assert "not a validation" in proc.stderr


# --- Docker build-time architecture contract (Task 5) ------------------------


def test_dockerfile_build_arg_contract():
    text = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG QWEN3_TTS_ROCM_GFX_TARGET" in text
    # forwarded verbatim to install.sh (no duplicated pip lines in the image)
    assert '--gfx-target "${QWEN3_TTS_ROCM_GFX_TARGET}"' in text
    assert "whl-multi-arch" not in text  # wheel logic stays in install.sh
    assert "device-gfx11" not in text    # no hard-coded extras in the image


def test_dockerfile_oci_metadata_is_multiarch():
    text = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    desc = re.search(r'org\.opencontainers\.image\.description="([^"]+)"', text)
    assert desc, "OCI description label missing"
    assert "gfx1100" in desc.group(1) and "gfx1151" in desc.group(1)
    # gfx1151-ONLY wording (the pre-v0.3 label) must not be the whole story
    assert "gfx1100" in desc.group(1)


def test_docker_readme_documents_both_targets():
    text = (REPO / "docker" / "README.md").read_text(encoding="utf-8")
    assert "--build-arg QWEN3_TTS_ROCM_GFX_TARGET=gfx1100" in text
    assert "--build-arg QWEN3_TTS_ROCM_GFX_TARGET=gfx1151" in text
    assert "fail-closed" in text or "fails the build" in text


# --- gfx1151 branch regression (audit companion, Task 7) ---------------------


def test_gfx1151_branch_produces_expected_wheel_configuration():
    """CPU proof that the preserved gfx1151 branch still generates the
    historically-validated install configuration (CODE LOGIC ONLY)."""
    argv = gfx.pip_install_argv("gfx1151")
    joined = " ".join(argv)
    assert "device-gfx1151" in joined
    assert gfx.AMD_ROCM_INDEX in joined
    assert gfx.TORCH_PIN == "2.12.0+rocm7.14.0"
    assert gfx.TORCHVISION_PIN == "0.27.0+rocm7.14.0"
    assert gfx.TORCHAUDIO_PIN == "2.11.0+rocm7.14.0"
