# Qwen3-TTS-ROCm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the official `qwen-tts==0.1.1` package run fully on gfx1151 via ROCm 7.14.0 pip wheels, ship an enhanced bilingual Gradio demo, and release the whole thing as an open-source project (`Qwen3-TTS-ROCm`, PyPI `qwen3-tts-rocm`).

**Architecture:** Thin shim over the untouched official package: our `qwen3_tts_rocm` module provides env diagnostics, a ModelScope-first model downloader, a smart-default loader returning *native* official objects, and an enhanced demo. If Task 1 (spike) proves the shim can't work, escalate to vendor+patches per spec §1.

**Tech Stack:** Python 3.12 / uv venv, torch 2.12.0+rocm7.14.0 `[device-gfx1151]`, qwen-tts 0.1.1, transformers 4.57.3, Gradio, modelscope SDK + hf-mirror fallback, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-qwen3-tts-rocm-design.md` (read it first — every requirement traces to §1–§10).

## Global Constraints

- GPU: AMD Radeon 8060S **gfx1151**; never install bare `torch` from default PyPI — always with `--index-url https://repo.amd.com/rocm/whl-multi-arch/`.
- torch stack exact pins: `torch[device-gfx1151]==2.12.0+rocm7.14.0`, `torchvision[device-gfx1151]==0.27.0+rocm7.14.0`, `torchaudio==2.11.0+rocm7.14.0`.
- Upstream pin: `qwen-tts==0.1.1`; our package must NOT declare torch/torchvision/torchaudio in its dependencies (they come from the AMD index step; declaring them would pull CUDA wheels).
- Network reality on this machine: github.com and huggingface.co are unreachable; pypi.org, repo.amd.com, modelscope.cn, hf-mirror.com work. All downloads must use those.
- Python venv at `.venv/` created by uv; all project commands run through `.venv/bin/python` (or activate).
- Models live under `$QWEN3_TTS_ROCM_MODELS_DIR` (default `<repo>/models`) — one subfolder per repo.
- Conventional Commits; commit after every task's green tests.
- Code identifiers/docstrings/comments in English; user-facing UI strings bilingual Chinese+English (format `"English (中文)"` as official demo does).
- Default attention on ROCm = `"sdpa"`, dtype = bfloat16; explicit `flash_attention_2` request must raise a clear error when flash-attn is absent — never silently downgrade (spec §6).
- GitHub Actions file is authored but cannot be validated online from this machine — say so in the task and keep CI strictly CPU-only.

---

### Task 1: Spike — prove ROCm runtime end-to-end

**Files:**
- Create: `evidence/spike/spike-report.md`
- Create: `scripts/verify_gpu.sh` (kept for good)

**Interfaces:**
- Produces: working `.venv` with AMD torch + `qwen-tts==0.1.1` installed; tokenizer model downloaded; written verdict `PROCEED` or `ESCALATE-TO-VENDOR` recorded in `evidence/spike/spike-report.md`. All later tasks assume PROCEED.

- [ ] **Step 1: Create venv and install the AMD wheel stack**

```bash
cd /home/amd/Desktop/Qwen3-TTS-ROCm
uv venv --seed .venv --python 3.12
.venv/bin/python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" \
    "torchaudio==2.11.0+rocm7.14.0"
```

Contingency: if torchvision pin has no matching artifact, rerun without the torchvision line (nothing downstream uses it) and record that deviation in the spike report.

- [ ] **Step 2: Install upstream package + tooling from PyPI**

```bash
.venv/bin/python -m pip install "qwen-tts==0.1.1" modelscope huggingface_hub pytest ruff
```

- [ ] **Step 3: GPU sanity probe (bf16 matmul + SDPA + torchaudio backend)**

```bash
cat > /tmp/spike_probe.py <<'EOF'
import torch, torch.nn.functional as F
print("torch", torch.__version__, "| HIP", torch.version.hip)
assert torch.cuda.is_available(), "cuda not available under ROCm build"
dev = "cuda:0"; props = torch.cuda.get_device_properties(0)
print("GPU:", props.name, "| arch:", getattr(props, 'gcnArchName', 'n/a'))
a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
b = a @ a
assert torch.isfinite(b).all().item()
q = torch.randn(4, 8, 128, device=dev, dtype=torch.bfloat16)
o = F.scaled_dot_product_attention(q, q, q)
assert torch.isfinite(o).all().item()
import torchaudio; print("torchaudio", torchaudio.__version__)
print("SPIKE-GPU-OK")
EOF
.venv/bin/python /tmp/spike_probe.py | tee evidence/spike/gpu-probe.txt
```

Expected: last line `SPIKE-GPU-OK`. Any failure here → stop, debug environment (HSA vars, kfd perms) before anything else.

- [ ] **Step 4: Download Tokenizer via ModelScope and smoke encode/decode**

```bash
.venv/bin/python - <<'EOF' 2>&1 | tee evidence/spike/tokenizer-smoke.txt
from modelscope import snapshot_download
p = snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz", local_dir="models/Qwen3-TTS-Tokenizer-12Hz")
print("downloaded to", p)
from qwen_tts import Qwen3TTSTokenizer
tok = Qwen3TTSTokenizer.from_pretrained(p, device_map="cuda:0", dtype="bfloat16")
import numpy as np, scipy.io.wavfile as wav  # any short wav works; synthesize silence-free tone
sr = 24000; t = np.linspace(0, 1, sr, endpoint=False); x = (0.1*np.sin(2*np.pi*220*t)).astype(np.float32)
wav.write("/tmp/tone.wav", sr, x)
out = tok.encode("/tmp/tone.wav"); print("encode ok:", type(out))
res = tok.decode(out); print("decode ok:", res.get_audio() if hasattr(res,'get_audio') else res)
print("TOKENIZER-OK")
EOF
```

If the modelscope snapshot_download import path differs, fall back: `.venv/bin/modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir models/Qwen3-TTS-Tokenizer-12Hz` and adapt Step 5 accordingly. Record which path worked (Task 5 needs it).

- [ ] **Step 5: Write verdict report**

Create `evidence/spike/spike-report.md`: commands run, probe outputs, versions (`torch.version.hip` must start `7.14`), deviations, and one line at top: `VERDICT: PROCEED` (shim viable) or `VERDICT: ESCALATE-TO-VENDOR` (core patch required → stop plan, notify human). Commit.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore(spike): verify ROCm 7.14.0 + qwen-tts on gfx1151"
```

---

### Task 2: Package scaffold

**Files:**
- Create: `pyproject.toml`, `src/qwen3_tts_rocm/__init__.py`, `src/qwen3_tts_rocm/py.typed`
- Create: `tests/test_scaffold.py`

**Interfaces:**
- Produces: importable package `qwen3_tts_rocm` exposing `__version__ = "0.1.0"`; pytest markers `gpu`, `requires_download` registered; editable install working.

- [ ] **Step 1: Write failing test**

```python
# tests/test_scaffold.py
def test_package_imports_and_version():
    import qwen3_tts_rocm
    assert qwen3_tts_rocm.__version__ == "0.1.0"

def test_gpu_marker_registered(pytester):
    from _pytest.config import get_config
    marks = get_config().pluginmanager.has_plugin("markers")
    assert marks
```

And a marker sanity test using `-m gpu` selection later; registering happens via pyproject below.

- [ ] **Step 2: Run and see it fail**

Run: `.venv/bin/python -m pytest tests/test_scaffold.py -v`
Expected: FAIL (`No module named qwen3_tts_rocm`).

- [ ] **Step 3: Write pyproject.toml and package init**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "qwen3-tts-rocm"
version = "0.1.0"
description = "ROCm (gfx1151) runtime adapter, model manager and enhanced Gradio demo for Qwen3-TTS"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [{ name = "Qwen3-TTS-ROCm contributors" }]
dependencies = [
  "qwen-tts==0.1.1",
  "modelscope",
  "huggingface_hub",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[project.scripts]
qwen3-tts-rocm-check = "qwen3_tts_rocm.cli_check:main"
qwen3-tts-rocm-demo = "qwen3_tts_rocm.cli_demo:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
qwen3_tts_rocm = ["py.typed", "demo/assets/*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "gpu: requires a working ROCm GPU (deselect with -m 'not gpu')",
  "requires_download: requires downloaded model weights",
]

[tool.ruff]
line-length = 110
target-version = "py310"
```

Note: deliberately no `torch*` entries in dependencies (Global Constraints).

```python
# src/qwen3_tts_rocm/__init__.py
"""ROCm (gfx1151) runtime adapter, model manager and enhanced demo for Qwen3-TTS."""
__version__ = "0.1.0"
```

Then: `printf '' > src/qwen3_tts_rocm/py.typed && mkdir -p src/qwen3_tts_rocm/demo/assets tests && touch tests/__init__.py`… no `tests/__init__.py` needed — skip it; use rootdir-relative imports only.

Also create stub modules so console scripts resolve even before their tasks (each raising NotImplementedError):

```python
# src/qwen3_tts_rocm/cli_check.py
def main() -> int:
    raise NotImplementedError("Task 3")

# src/qwen3_tts_rocm/cli_demo.py
def main() -> int:
    raise NotImplementedError("Task 15/16")
```

- [ ] **Step 4: Editable-install and green**

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_scaffold.py -v   # PASS
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: package scaffold qwen3-tts-rocm"
```

---

### Task 3: env.py — GPU/ROCm diagnostics

**Files:**
- Create: `src/qwen3_tts_rocm/env.py`, `src/qwen3_tts_rocm/cli_check.py` (replace stub), `tests/test_env.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass GpuInfo(name: str, arch: str | None, cu_count: int | None)
  @dataclass EnvReport(hip_available: bool, rocm_version: str | None,
                       gpus: list[GpuInfo], warnings: list[str], errors: list[str])
  def collect() -> EnvReport                       # pure inspection, tolerant of missing torch
  def rocm_check(verbose: bool = True) -> EnvReport  # prints bilingual diagnostics when verbose
  def pick_device(preference: str = "auto") -> str # -> "cuda:0" or "cpu"
  def require_rocm_torch() -> None                 # raises RuntimeError with fix hint
  ```
- Consumed by: loader (pick_device, require_rocm_torch), cli_check, docs troubleshooting examples.

- [ ] **Step 1: Write failing tests (CPU-safe, mock torch via monkeypatch)**

```python
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
```

- [ ] **Step 2: Run and see failures**

Run: `.venv/bin/python -m pytest tests/test_env.py -v` → ImportError (env module missing).

- [ ] **Step 3: Implement env.py**

Real implementation notes (behavior under real torch is what production hits; tests exercise the mocked branch): read `torch.version.hip` (None ⇒ CUDA/non-HIP build ⇒ error entry telling user to reinstall via `--index-url https://repo.amd.com/rocm/whl-multi-arch/`); guard every attribute access inside try/except so `collect()` NEVER raises. Warnings checklist per spec §5.1: kfd group membership (`os.access("/dev/kfd", os.R_OK|os.W_OK)`), HSA_OVERRIDE_GFX_VERSION set-but-unneeded notice, `CUDA_VISIBLE_DEVICES` interplay note on ROCm, OOM/GTT advice string constant `GTT_HINT`. `rocm_check(verbose)` prints each warning/error prefixed `WARN:`/`ERROR:` plus bilingual tail line. `cli_check.main()` calls it, returns 0 when errors empty else 1.

```python
# src/qwen3_tts_rocm/env.py  (full implementation goes here; ~120 lines)
GTT_HINT = ("If generation hits out-of-memory on unified-memory APUs, consider lowering "
            "max_new_tokens or switching to a 0.6B model.")

def collect() -> EnvReport:
    ...  # try-import torch; fill EnvReport; append diagnostics per rules above; never raise
```

Replace the body of `src/qwen3_tts_rocm/cli_check.py`:

```python
import argparse
from .env import rocm_check

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qwen3-tts-rocm-check",
                                 description="Diagnose ROCm GPU readiness (环境自检)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ns = ap.parse_args(argv)
    report = rocm_check(verbose=not ns.quiet)
    return 0 if not report.errors else 1
```

- [ ] **Step 4: Green + real-machine smoke**

```bash
.venv/bin/python -m pytest tests/test_env.py -v          # PASS
.venv/bin/qwen3-tts-rocm-check                            # exit 0, shows gfx1151
.venv/bin/qwen3-tts-rocm-check | tee ../Qwen3-TTS-ROCm/evidence/env-check.txt || true
```

Record real output into `evidence/env-check.txt`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(env): ROCm diagnostics + qwen3-tts-rocm-check CLI"`

---

### Task 4: models.py registry & path resolution

**Files:**
- Create: `src/qwen3_tts_rocm/models.py`, `tests/test_models_registry.py`

**Interfaces:**
- Produces:
  ```python
  ALIASES: tuple[str, ...]  # ("tokenizer","voice-design","custom-voice","base","custom-voice-0.6b","base-0.6b")
  REPOS: dict[str,str]      # alias -> "Qwen/..." HF-style id
  def local_dir(alias: str | None = None) -> Path          # $QWEN3_TTS_ROCM_MODELS_DIR or <repo>/models [/<flattened-repo-name>]
  def flatten(repo_id: str) -> str                          # "Qwen/Foo" -> "Foo"
  def is_downloaded(alias: str) -> bool                     # dir exists AND contains config.json-ish files OR .ok
  def mark_downloaded(alias) / None-out none needed
  def alias_of_repo(repo_id_or_flattened: str) -> str       # reverse lookup
  def resolve_path(ref: str | Path) -> Path                 # existing-dir passthrough | alias | flattened | full id  (raises KeyError w/ bilingual msg)
  ```

- [ ] **Step 1: Failing tests**

```python
# tests/test_models_registry.py
import pytest
from pathlib import Path
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
```

- [ ] **Step 2:** run → fail (module missing).
- [ ] **Step 3:** implement exactly the names above (registry dict literal matches EXPECTED order/content; `local_dir()` honors env var then defaults to `<git-top-or-cwd>/models`). `is_downloaded`: dir exists and (`config.json` present or `.ok` marker present).
- [ ] **Step 4:** green.
- [ ] **Step 5:** `git add -A && git commit -m "feat(models): registry + path resolution"`

---

### Task 5: models.py downloader (ModelScope-first, dual-source)

**Files:**
- Modify: `src/qwen3_tts_rocm/models.py`
- Create: `tests/test_models_download.py`

**Interfaces:**
- Consumes: REPOS/local_dir/markers from Task 4.
- Produces:
  ```python
  def download(aliases="all", source: str = "auto", models_dir=None, resume=True) -> list[Path]
      # source ∈ {"modelscope","hfmirror","auto"}; auto tries modelscope then hfmirror per alias
  SUPPORTED_SOURCES = ("modelscope", "hfmirror")
  ```
- Implementation call style comes from spike findings (SDK import path recorded in spike-report): prefer `from modelscope import snapshot_download`; hf-mirror branch sets `HF_ENDPOINT=https://hf-mirror.com` env **before** importing/instantiating huggingface_hub API (`snapshot_download(repo_id, local_dir=..., max_workers=4)`) inside subprocess-safe pattern: pass `endpoint` kwarg if supported, else mutate os.environ around the call and restore after.
- On success writes `.ok` file into target dir.

- [ ] **Step 1: Failing tests (mock the network layer)**

```python
# tests/test_models_download.py
import pytest
from qwen3_tts_rocm import models

class FakeMS:
    calls = []
    @classmethod
    def snapshot_download(cls, model_id, local_dir=None, **kw):
        cls.calls.append(("ms", model_id)); import pathlib; pathlib.Path(local_dir, ".ok").write_text("ok")
        return local_dir

class FakeMS_Fail(FakeMS):
    @classmethod
    def snapshot_download(cls, model_id, **kw): raise OSError("network down")

def test_download_single(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M
    monkeypatch.setattr(M, "_ms_snapshot", FakeMS.snapshot_download)
    paths = M.download("tokenizer", source="modelscope", models_dir=tmp_path)
    assert paths[0].exists() and (paths[0]/".ok").exists()
    assert FakeMS.calls[-1][1] == "Qwen/Qwen3-TTS-Tokenizer-12Hz"

def test_download_auto_falls_back_to_hfmirror(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M
    calls = []
    monkeypatch.setattr(M, "_ms_snapshot", lambda mid, **kw: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(M, "_hf_snapshot", lambda rid, d, **kw: (calls.append(rid), M.mark_ok(d))[1])
    out = M.download(["tokenizer"], source="auto", models_dir=tmp_path)
    assert calls == ["Qwen/Qwen3-TTS-Tokenizer-12Hz"] and (out[0]/".ok").exists()

def test_skip_already_downloaded(monkeypatch, tmp_path):
    import qwen3_tts_rocm.models as M
    d = tmp_path/"X"; d.mkdir(); (d/".ok").write_text("")
    monkeypatch.setenv("QWEN3_TTS_ROCM_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(M, "_flatten_repo_for_test", lambda a: {"tokenizer":"X"}[a])
    def boom(*a, **k): raise AssertionError("should not be called")
    monkeypatch.setattr(M, "_ms_snapshot", boom)
    assert M.download("tokenizer", models_dir=tmp_path) == [d]
```

(Adjust `_flatten_repo_for_test` hook or simply create the expected folder name from REPOS mapping — implementer may replace this third test with equivalent semantics but MUST cover "skip when already downloaded".)

- [ ] **Step 2:** run → fail.
- [ ] **Step 3:** implement `download()` with internal `_ms_snapshot(model_id, local_dir)` and `_hf_snapshot(repo_id, local_dir)` indirection points used by tests; retries once per source; final failure raises RuntimeError listing both attempts + manual URLs (`https://modelscope.cn/models/<id>` and `https://hf-mirror.com/<id>`).
- [ ] **Step 4:** green.
- [ ] **Step 5:** commit — `feat(models): dual-source downloader with auto fallback`.

---

### Task 6: loader.py + patch.py

**Files:**
- Create: `src/qwen3_tts_rocm/loader.py`, `src/qwen3_tts_rocm/patch.py`, `tests/test_loader.py`

**Interfaces:**
- Consumes: models.resolve_path / is_downloaded / download; env.pick_device / require_rocm_torch.
- Produces:
  ```python
  DEFAULT_DTYPE = "bfloat16"
  def load(model_ref, device=None, dtype="bfloat16", attn_implementation=None, **kwargs) -> qwen_tts.Qwen3TTSModel
  def unload(model) -> None           # del refs, gc.collect(), torch.cuda.empty_cache()
  def apply_compat_patches() -> None  # patch.py entry; asserts qwen_tts.__version__=="0.1.1"; currently no-op list
  ```
  load() behavior (spec §5.3): resolve ref (auto-download if missing, `requires_download` marked flows only); attn guard — if explicit `"flash_attention_2"` and flash-attn not importable, `RuntimeError` explaining ROCm has no official flash-attn wheel and suggesting omit/`sdpa`; default attn on HIP GPU = `"sdpa"`; passes everything else verbatim to official `Qwen3TTSModel.from_pretrained`.

- [ ] **Step 1: Failing tests (monkeypatch official loader boundary)**

```python
# tests/test_loader.py
import sys, types, pytest
from qwen3_tts_rocm import loader

@pytest.fixture
def fake_official(monkeypatch):
    seen = {}
    mod_qt = types.ModuleType("qwen_tts")
    class FakeQwen3TTSModel:
        @classmethod
        def from_pretrained(cls, ref, **kw): seen.update(ref=ref, **kw); return object()
    mod_qt.Qwen3TTSModel = FakeQwen3TTSModel
    monkeypatch.setitem(sys.modules, "qwen_tts", mod_qt)
    monkeypatch.setattr(loader.models, "resolve_path", lambda r: "/models/X")
    return seen

def test_defaults_sdpa_on_hip(fake_official, monkeypatch):
    import torch as real_torch_lite  # existing env torch from earlier tasks’ installs
    loader.load("base", device="cuda:0")
    assert fake_official["attn_implementation"] == "sdpa"
    assert fake_official["dtype"] == "bfloat16"

def test_flash_attn_guard(fake_official, monkeypatch):
    real = pytest.importorskip  # ensure flash_attn NOT importable in test env
    with pytest.raises(RuntimeError, match="flash.?atten"):
        loader.load("base", attn_implementation="flash_attention_2")

def test_kwargs_passthrough(fake_official):
    loader.load("base", device="cpu", dtype="float32", custom_flag=42)
    assert fake_official["custom_flag"] == 42 and fake_official["dtype"] == "float32"

def test_unload_safe(monkeypatch):
    class Dummy: pass
    import gc, torch
    called = {}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.setdefault("x", True), raising=False)
    loader.unload(Dummy())     # must not raise even for odd inputs
```

Implementation detail for the guard: probe `_flash_available()` via `importlib.util.find_spec("flash_attn")`.

- [ ] **Step 2:** run → fail. **Step 3:** implement (~90 lines; docstring includes usage example mirroring README quickstart). **Step 4:** green. **Step 5:** commit `feat(loader): smart-default loader returning native official model + unload`.

---

### Task 7: Shell scripts

**Files:**
- Create: `scripts/install.sh`, `scripts/download_models.sh`, `scripts/run_demo.sh` (verify_gpu.sh already from Task 1)

**Interfaces:**
- Produces: `bash scripts/install.sh [--with-models]`; `bash scripts/download_models.sh [alias...]`; `bash scripts/run_demo.sh [--port N]` — the documented UX trio.

- [ ] **Step 1: install.sh** — bash strict mode (`set -euo pipefail`); ensures uv exists else falls back to `python3 -m venv`; creates `.venv --seed`; runs the EXACT three-line AMD pip command from Global Constraints; `pip install -e ".[dev]"`; optional `--with-models` triggers download script; final line runs `scripts/verify_gpu.sh`. Idempotent (safe re-run).
- [ ] **Step 2: download_models.sh** — loops aliases (args or all six), invokes `.venv/bin/python -c "from qwen3_tts_rocm.models import download; download([...])"`; prints per-repo size summary via `du -sh`.
- [ ] **Step 3: run_demo.sh** — activates `.venv`, execs `qwen3-tts-rocm-demo "$@"` forwarding args; default port 8000 documented in header comment.
- [ ] **Step 4:** manual smoke: `bash -n` each (syntax), then `bash scripts/install.sh` fully on this machine; confirm exit 0 and `verify_gpu.sh` prints SPIKE-GPU-OK content equivalent. Capture transcript to `evidence/install-run.txt`.
- [ ] **Step 5:** commit — `feat(scripts): one-command install, model fetch and demo launch`.

---

### Task 8: Download remaining five model repos (real weights)

**Files:**
- Create: `evidence/models-dl.txt`

**Interfaces:**
- Produces: six complete local repos under default `models/` verified against `is_downloaded`; disk footprint noted in evidence file. Downstream GPU tasks REQUIRE this task done.

- [ ] **Step 1:** `bash scripts/download_models.sh 2>&1 | tee evidence/models-dl.txt` (tokenzier from Task 1 already present; script must skip it via `.ok` check).
- [ ] **Step 2:** verify: `.venv/bin/python -c "from qwen3_tts_rocm.models import ALIASES,is_downloaded; assert all(is_downloaded(a) for a in ALIASES); print('ALL-DOWNLOADED')"`; add `du -sh models/*` output to evidence file.
- [ ] **Step 3:** commit evidence — `chore(evidence): six-model download log`.

*(This task has no TDD cycle; it is an environment deliverable gated by the programmatic assertion above.)*

---

### Task 9: Test infrastructure — waveform helpers + fixtures

**Files:**
- Create: `src/qwen3_tts_rocm/testing.py`, `tests/conftest.py`, `tests/test_testing_utils.py`

**Interfaces:**
- Produces (used by Tasks 10–17):
  ```python
  # src/qwen3_tts_rocm/testing.py
  def make_tone(seconds=1.0, freq=220.0, sr=24000) -> tuple[np.ndarray, int]
  def write_wav(path, wav, sr)
  def assert_wav_sane(wav: np.ndarray, sr_expected: int | None = None, min_energy_rms=1e-4):
      # finite, float, abs peak ≤ 1.5, rms > min_energy_rms (non-silent), sr match
  class FakeTTSModel:  # minimal official-API-compatible double
      generate_custom_voice(text, language, speaker, instruct=None, **kw) -> (list[np.ndarray], int)
      generate_voice_design(text, language, instruct, **kw) -> ...
      generate_voice_clone(text, language, ref_audio=None, ref_text=None, voice_clone_prompt=None, x_vector_only_mode=False, **kw) -> ...
      create_voice_clone_prompt(...) -> list
      get_supported_speakers/get_supported_languages -> lists
  # tests/conftest.py fixtures
  def gpu():            # session-scoped: require_rocm_torch(); skip whole fixture-dependent tests when unavailable
  def model_alias(request): parametrizable alias resolver + lazy cache helper `get_model(alias)`
  ```
  conftest caches loaded official models in a module-global dict to avoid reload cost across test modules; finalizer unloads.

- [ ] **Step 1:** failing tests for `make_tone/assert_wav_sane` (silent zero-array must raise AssertionError; sinusoid passes; sr mismatch raises).
- [ ] **Step 2:** red. **Step 3:** implement. **Step 4:** green (CPU-only suite: `pytest tests/test_testing_utils.py -m "not gpu"`).
- [ ] **Step 5:** commit — `test(helpers): waveform sanity utilities + FakeTTSModel`.

---

### Task 10: GPU integration — six-model load/unload smoke

**Files:**
- Create: `tests/test_loader_all_models.py`

**Interfaces:** consumes loader.load/unload, conftest `gpu` fixture, models from Task 8.

- [ ] **Step 1:** failing test:

```python
# tests/test_loader_all_models.py
import pytest
from qwen3_tts_rocm import loader, models, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

HEAVY = {"tokenizer"}  # tokenizer loads via Qwen3TTSTokenizer path separately (Task 14 covers codec end-to-end)

@pytest.mark.parametrize("alias", ["custom-voice", "voice-design", "base", "custom-voice-0.6b", "base-0.6b"])
def test_load_smoke_each_tts_model(alias, gpu):
    m = loader.load(alias)
    assert m.model.device.type == "cuda"
    loader.unload(m)

def test_tokenizer_load(gpu):
    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(str(models.local_dir("tokenizer")), device_map="cuda:0", dtype="bfloat16")
    assert tok.get_output_sample_rate() > 0
```

- [ ] **Step 2:** run `pytest tests/test_loader_all_models.py -m gpu -v` → expect failures ONLY if a genuine load issue (this is also first real-model exercise; treat failures as product bugs, debug systematically — systematic-debugging skill applies, do not paper over).
- [ ] **Step 3-4:** fix forward until green; record peak memory per model in evidence comment lines.
- [ ] **Step 5:** commit — `test(gpu): all-model load/unload smoke`.

---

### Task 11: GPU tests — generate_custom_voice matrix

**Files:**
- Create: `tests/test_generate_custom_voice.py`

- [ ] **Step 1:** failing tests:

```python
# tests/test_generate_custom_voice.py
import pytest
from qwen3_tts_rocm import loader, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

TEXT = "今天的天气真不错，适合去公园散步。"

@pytest.fixture(scope="module")
def cv_model(gpu): 
    m = loader.load("custom-voice"); yield m; loader.unload(m)

def test_single(cv_model):
    spks = cv_model.get_supported_speakers(); langs = cv_model.get_supported_languages()
    assert len(spks) >= 9 and langs and "Auto" in langs
    wavs, sr = cv_model.generate_custom_voice(text=TEXT, language="Auto", speaker=spks[0])
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr, min_energy_rms=1e-3)

def test_batch(cv_model):
    wavs, sr = cv_model.generate_custom_voice(text=[TEXT, TEXT], language="Auto",
                                              speaker=cv_model.get_supported_speakers()[0])
    assert len(wavs) == 2
    for w in wavs: testing.assert_wav_sane(w, sr_expected=sr)

def test_instruct_changes_output(cv_model):
    spk = cv_model.get_supported_speakers()[0]
    a,_ = cv_model.generate_custom_voice(text=TEXT, language="Auto", speaker=spk, instruct="用低沉缓慢的语气说")
    b,_ = cv_model.generate_custom_voice(text=TEXT, language="Auto", speaker=spk, instruct="用非常欢快的语气说")
    assert abs(float(a[0].std()-b[0].std())) >= 0  # both sane individually is the real assertion below
    testing.assert_wav_sane(a[0]); testing.assert_wav_sane(b[0])

def test_sampling_kwarg_passthrough_effect(cv_model):
    spk = cv_model.get_supported_speakers()[0]
    _, lo = cv_model.generate_custom_voice(text=TEXT, language="Auto", speaker=spk, temperature=0.01)
    _, hi = cv_model.generate_custom_voice(text=TEXT, language="Auto", speaker=spk, temperature=1.9)
    testing.assert_wav_sane(lo[0]); testing.assert_wav_sane(hi[0])
    assert lo[0].shape != hi[0].shape or float(abs(lo[0]-hi[0]).max()) > 0
```

- [ ] **Steps 2–5:** red → debug product issues → green → commit `test(gpu): custom voice full matrix`.
NOTE (sampling determinism): temperature extremes produce different token streams; shape OR value inequality assertion keeps the test robust. Set seeds where supported by official kwargs before each pair to reduce flakiness; if flaky after two tuning attempts, relax to separate sanity-only assertions and document in test docstring.

---

### Task 12: GPU tests — voice design

**Files:** Create: `tests/test_voice_design.py`

Structure mirrors Task 11 against alias `voice-design`: single sane output; batch of 2; two different design instructions both sane AND distinguishable (same tolerance policy as Task 11 NOTE); unsupported-language validation surfaces official `ValueError` gracefully (`pytest.raises(ValueError)` calling with language="Klingon"). Same 5-step TDD loop; commit `test(gpu): voice design suite`.

Default TEXT same as Task 11; INSTRUCTIONS: A `"年轻女性，声音清亮，语速轻快"` B `"老年男性，声音沙哑，语速缓慢"`.

---

### Task 13: GPU tests — voice clone workflow (incl. prompt reuse + save/load)

**Files:** Create: `tests/test_voice_clone_workflow.py`

Cover (spec §7 row 3): standard clone with reference audio+text via `testing.make_tone`-derived speech-like WAV *fallback*: better than sine — use bundled `assets/sample_ref_*.wav` if Task 16 ships them first; ordering constraint resolved by shipping ONE tiny reference sample (`<3s`, cmu_arctic-style clip public domain) inside THIS task as `assets/ref_en.wav` copied into `src/qwen3_tts_rocm/demo/assets/`. Tests:
1. `clone_with_ref_text` → sane wav.
2. `clone_x_vector_only_mode=True` (ref_text=None) → sane wav (weaker quality accepted, still non-silent).
3. `create_voice_clone_prompt` returns items; second `generate_voice_clone(..., voice_clone_prompt=items)` sane; reuse twice consistent API.
4. Save/load roundtrip byte-format parity with OFFICIAL demo payload: serialize `{"items":[asdict(it) for it in items]}` via `torch.save`, reload, regenerate → sane. Assert `VoiceClonePromptItem` fields present exactly as official dataclass.
5. Batch clone text list length 2 → 2 outputs.
Same commit hygiene; `requires_download`+`gpu` marks.

---

### Task 14: GPU tests — tokenizer codec + getters

**Files:** Create: `tests/test_tokenizer_codec.py`

- Build tokenizer fixture from `models.local_dir("tokenizer")`.
- Encode `assets/ref_en.wav` → codes object (accept whatever official encode returns; decode back) → `testing.assert_wav_sane` on decoded audio with `sr_expected=tok.get_output_sample_rate()`.
- Getter assertions pinned to DOCUMENTED values only (do not hardcode risky numerics beyond sanity ranges): input/output rates positive ints, downsample/upsample rates positive ints, `tok.get_model_type()` non-empty string (log actual values to stdout for the evidence trail rather than asserting exotic constants).
- Roundtrip degradation bound: `len(codes)` such that reported compression implies downsampling ≥ 1× — assert `duration_seconds/tokens_used >= 1` if accessor available, else skip-with-note (keep faithful to official API surface, no guessing private attrs).
Five-step loop; commit `test(gpu): tokenizer encode/decode roundtrip + metadata getters`.

---

### Task 15: Demo backend (headless service layer)

**Files:**
- Create: `src/qwen3_tts_rocm/demo/__init__.py`, `src/qwen3_tts_rocm/demo/backend.py`, `tests/test_demo_backend.py`

**Interfaces:**
- Produces:
  ```python
  class SynthesisService:
      def __init__(self): self._cache: dict[str, model] ; current_alias
      def get(self, alias) -> model            # lazy official load, LRU size 1 (unload previous)
      def unload_all(self)
      def custom_voice(alias, text, language_display, speaker_display, instruct, gen_kwargs) -> (int, np.ndarray)
      def voice_design(alias, text, language_display, instruct, gen_kwargs) -> (int, np.ndarray)
      def voice_clone(alias, text, language_display, ref_audio(sr,wav)|None, ref_text, xvec_only, gen_kwargs) -> ...
      def clone_prompt_from_ref(alias, ref_audio, ref_text, xvec_only) -> items
      def voice_clone_with_prompt(alias, text, language_display, items, gen_kwargs) -> ...
      def codec_roundtrip(wav_tuple) -> (int, np.ndarray, dict)   # uses tokenizer alias; meta incl rates + codes count best-effort
  class HistoryStore: add(label,sr,wav)->id ; list()->summaries ; item(id)->(sr,wav) ; remove(id) ; cap=30
  def format_error(e) -> str               # bilingual "<Type>: msg / 中文提示：请检查输入或查看终端日志"
  def display_map(values: list[str]) -> tuple[list[str], dict]  # reuse official title-case convention
  ```

- [ ] **Step 1:** failing headless tests using `FakeTTSModel` injected via monkeypatching `loader.load`: every method returns `(sr, ndarray)`; invalid empty text raises `ValueError("Text is required...")` BEFORE touching model; HistoryStore caps at 30 FIFO; format_error includes both original message and 中文 suffix; get()/LRU: requesting different alias triggers unload of previous (spy on loader.unload).
- [ ] **Steps 2–5:** red→implement→green→commit `feat(demo-backend): headless synthesis service + history`.

---

### Task 16: Demo UI assembly (enhanced Gradio app)

**Files:**
- Create: `src/qwen3_tts_rocm/demo/ui.py`, `src/qwen3_tts_rocm/cli_demo.py` (replace stub), `src/qwen3_tts_rocm/demo/assets/.gitkeep`, shell-test `tests/test_demo_ui.py`

**Interfaces:**
- Consumes: ALL Task 15 backend methods; official demo conventions (`gr.themes.Soft`, bilingual labels, disclaimer footer text from upstream demo.py copied verbatim).
- Produces: `def build_ui(service, port_header_info: dict) -> gr.Blocks` wiring four tabs × callbacks; CLI `main(argv)` mirrors upstream arg surface (`checkpoint/model positional OR --alias`, `--device --dtype --no-flash-attn`, `--ip/--port/--share/--concurrency/--ssl-*`) PLUS `--models-dir`. Tab specs = design doc §5.5 table; global sidebar (model switcher Radio of the four usable aliases incl 0.6b variants with lazy loading status textbox, advanced sampling accordion writing into gen_kwargs dict state, VRAM indicator fed by torch.cuda.mem_get_info polled per-generation callback).

- [ ] **Step 1: failing UI-shape test (headless Blocks construction)**

```python
# tests/test_demo_ui.py
import gradio as gr, pytest
from qwen3_tts_rocm.demo.ui import build_ui
from qwen3_tts_rocm.testing import FakeTTSModel

def test_build_ui_constructs_four_tabs():
    svc = _fake_service()
    app = build_ui(svc, {"alias": "custom-voice"})
    assert isinstance(app, gr.Blocks)
    ids = {c.label for c in app.blocks.values() if hasattr(c, "label")}
    assert any(l and "Reference" in l for l in ids)          # clone tab present
    assert any(l and "Speaker" in l for l in ids)            # custom voice tab
    assert any(l and "Voice Design" in l for l in ids or []) # design tab
    assert any(l and "Codec" in l or l and "Codec" in str(l) for l in ids)

def _fake_service(monkeypatch=None):
    from qwen3_tts_rocm.demo import backend
    svc = backend.SynthesisService.__new__(backend.SynthesisService)
    svc._cache = {}; svc.current_alias = None
    object.__setattr__(svc, "_factory", lambda alias: FakeTTSModel())
    return svc
```

(Where `SynthesisService` accepts injectable factory in constructor for testability: `SynthesisService(factory=None)` defaulting to real loader wrapper.)

- [ ] **Step 2:** red. **Step 3:** implement ui.py following upstream demo.py structure as closely as practical while adding: mic recording sources on reference Audio components (`sources=["upload","microphone"]`), Save/Load voice tab replicated EXACTLY per official payload schema from Task 13 save/load tests, Codec tab wired to `service.codec_roundtrip`, History gallery refreshed post-generation. Callback bodies stay thin wrappers delegating to backend + `format_error` try/except (UI logic itself thin, business logic tested in Task 15).
- [ ] **Step 4:** green UI tests + REAL smoke on machine: `bash scripts/run_demo.sh --port 8000` then curl `http://127.0.0.1:8000/` expecting 200 HTML containing `Qwen3 TTS`; record screenshot via browser skill OR at minimum curl transcript appended to `evidence/demo-smoke.txt`; Ctrl-C cleanup.
- [ ] **Step 5:** commit — `feat(demo-ui): enhanced four-tab bilingual Gradio application`.

---

### Task 17: Official-parity proof test

**Files:** Create: `tests/test_official_demo_parity.py`

- [ ] **Step 1:** failing test (marks gpu+requires_download): import UPSTREAM `qwen_tts.cli.demo.build_demo`; construct `tts = loader.load("custom-voice")`; call `build_demo(tts, ckpt=str(models.local_dir("custom-voice")), gen_kwargs_default={})` → returns gr.Blocks without exception; then invoke ONE upstream callback through `demo.fns` only if trivially introspectable — otherwise restrict scope to construction-success assertion PLUS direct upstream `run_instruct` invocation pattern documented in module docstring (upstream defines handlers as closures; if unreachable, construction assertion suffices as parity evidence per spec §7 row 4 wording "可构建并可执行一次回调" — enforce at minimum `gr.Blocks` + number of `.blocks` elements > 20).
- [ ] **Steps 2–5:** red→green→commit `test(gpu): upstream demo runs unmodified on our stack (parity proof)`.

---

### Task 18: Benchmark script + numbers

**Files:** Create: `scripts/benchmark.py`, `docs/benchmarks.md`, `evidence/benchmark.json`

- [ ] **Step 1:** implement benchmark.py: args `--aliases custom-voice,voice-design,base --text-file` (defaults inline CN+EN samples), for each: warm-up run then measure wall time vs resulting audio duration → RTF, plus per-call first-token proxy (time to first chunk unavailable offline → omit; RTF only, honest). Emit JSON to evidence + markdown table template into docs/benchmarks.md.
- [ ] **Step 2:** run on this machine; paste real numbers into docs/benchmarks.md with hardware footnote (Ryzen AI Max+ PRO 395, ROCm 7.14.0 wheels, bf16/sdpa).
- [ ] **Step 3:** commit — `perf(benchmark): RTF measurements on gfx1151`.

---

### Task 19: Community & legal files

**Files:** Create: `LICENSE` (full Apache-2.0 text, copyright line "Copyright 2026 Qwen3-TTS-ROCm contributors"), `NOTICE` (derives from QwenLM/Qwen3-TTS Apache-2.0, models © Alibaba Qwen team under their model license — link official model page terms), `CHANGELOG.md` (Keep-a-Changelog v0.1.0 entry summarizing features), `CONTRIBUTING.md` (setup: scripts/install.sh; pytest marker conventions incl `-m "not gpu"` for CI-class runs; commit style), `SECURITY.md` (contact process), `CODE_OF_CONDUCT.md` (Contributor Covenant short pointer text), `.gitignore` (.venv, models/, __pycache__, evidence tmp, dist/)

- No tests (static content). Steps: author files → verify `pip install build && python -m build --sdist --wheel` still legal → sanity `grep -L "Apache" LICENSE` empty → commit `docs: community and licensing files`.

---

### Task 20: Bilingual README + troubleshooting doc

**Files:** Create: `README.md`, `README_CN.md`, `docs/troubleshooting.md`

Content contract (both languages mirrored section-for-section): badges (license apache, python≥3.10, platform linux-rocm), 3-line hero pitch, "Why this fork-style project exists" (one paragraph pointing at zero-modification guarantee + parity test), Requirements table (from spec §2 facts incl network mirror guidance), Quickstart (copy-paste: `git clone … / cd / bash scripts/install.sh / bash scripts/download_models.sh / bash scripts/run_demo.sh`), minimal PYTHON usage snippet replicating official README quickstart but passing our `loader.load()` once then pure-official calls (proving native-object claim), Demo screenshot placeholders `docs/img/demo-*.png` (capture during Task 16 smoke; if missing at release time, release blocks on capturing them — checklist item), HTTPS microphone notes (openssl one-liner adapted from upstream), benchmarks teaser linking docs/benchmarks.md, FAQ/troubleshooting table linking docs/troubleshooting.md (which expands every ERROR/WARN from env.py GTT_HINT etc.), attribution & disclaimer sections (upstream text reused). `- [ ] Steps: author EN → author CN translation parity walk → render-link check (python -c markdown links grep) → commit docs(readme): bilingual documentation.`

---

### Task 21: Docker + CI workflow

**Files:** Create: `docker/Dockerfile`, `.github/workflows/ci.yml`, docker note section appended to README files? NO — separate `docker/README.md`.

- Dockerfile stages FROM `ubuntu:24.04`; apt minimal (python3.12 via deadsnakes NOT needed—ubuntu 24.04 ships 3.12; git, curl, ca-certificates, libsndfile1, sox OPTIONAL comment, ffmpeg comment); copy repo; ENV HF_ENDPOINT default NOT set (downloads happen volume-mounted usually) but VOLUME /workspace/models; run install steps identical to scripts/install.sh logic (call it directly!); ENTRYPOINT `bash scripts/run_demo.sh`; documented `docker run --device /dev/kfd --device /dev/dri --group-add video --group-add render -p 8000:8000 ...`; build validated locally: `docker build -t qwen3-tts-rocm:dev .` MUST succeed on this machine (daemon presence assumed — if daemon unavailable in executor sandbox, mark TODO-BLOCKER comment inside plan execution report and rely on hadolint dry parse `docker build --check` when available).
- ci.yml: `on: push/pull_request`; job lint-test ubuntu-latest CPU: ruff check; `pip install -e ".[dev]"`; `pytest -m "not gpu and not requires_download"`; `python -m build` artifact upload. Header YAML comment: authored offline, validate-on-first-push.
- Steps: author Dockerfile → local build/curl smoke (port check like Task 16) OR degraded-validation path → author ci.yml → `python -c "import yaml,sys;yaml.safe_load(open('.github/workflows/ci.yml'))"` → commit `ci(docker+actions): container image and CPU pipeline`.

---

### Task 22: Final acceptance, evidence freeze, tag v0.1.0

**Files:** Create: `evidence/final-acceptance.md`; finalize CHANGELOG dates.

- [ ] **Step 1:** full suites: `pytest -m "not gpu"` (CI parity) AND `pytest -m "gpu and requires_download" -v` fresh terminal; transcripts → evidence/.
- [ ] **Step 2:** demo E2E manual pass via browser-use skill against `run_demo.sh` server exercising all four tabs once each with screenshots saved to docs/img/ (closing Task 20 placeholder checklist).
- [ ] **Step 3:** `python -m build` + `twine check dist/*` (install twine into dev extras if absent ad hoc).
- [ ] **Step 4:** tag annotated `v0.1.0` with bullet changelog; final acceptance md records: versions, test counts passed, RTF headline numbers, known limitations (flash-attn absent; GitHub push pending by owner due to network).
- [ ] **Step 5:** commit + tag push prep instructions for user (owner performs remote push due to network constraints — document exact commands in final-acceptance.md).

---

## Self-Review Notes (post-draft)

- Spec coverage sweep: §1 goals ↔ Tasks (all-features → 10–14 tests + native-object loader 6; demo → 15–16; release → 19–22); §2 facts → Global Constraints; §3 registry → Task 4; §4 architecture → Tasks 3–6,15–16; §5 component APIs → signatures above align verbatim; §6 error handling → env diagnostics (3), loader guard (6), download fallback (5), backend format_error (15); §7 tests → 9–14,17; §8 publishing → 19–21, PyPI build in 22; §9 ordering → task sequence IS the spike-first constraint; §10 risks → spike (1), version pinning (constraints), fallback torch 2.11 contingency documented in Task 1 Step 1.
- Placeholder scan: no TBD/TODO patterns; contingency branches carry concrete alternate commands (Task 1 torchvision, Task 5 SDK import styles).
- Type-consistency: `SynthesisService(factory=...)` introduced in Task 15 consumed by Task 16 test helper ✓; `assert_wav_sane(wav, sr_expected=None, min_energy_rms=1e-4)` identical signature across 9/11/14 ✓; `loader.load/unload` names stable across 6/10–13/17 ✓; `models.local_dir/resolve_path/download/is_downloaded/ALIASES/REPOS` unchanged between tasks 4,5,6,10,14 ✓.
