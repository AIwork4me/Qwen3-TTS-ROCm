# Troubleshooting — Qwen3-TTS-ROCm

Companion guide to [`README.md`](../README.md) /
[`README_CN.md`](../README_CN.md). It expands **every** `ERROR:` / `WARN:` /
`INFO:` line the project's diagnostics
([`src/qwen3_tts_rocm/env.py`](../src/qwen3_tts_rocm/env.py)) can emit,
plus the downloader failures, the memory playbook and the harmless console
noise. Bilingual (中文) wording appears where the quoted diagnostic itself is
bilingual.

## Read the diagnostics first

```bash
qwen3-tts-rocm-check          # console entry point (installed by scripts/install.sh)
```

* Equivalent API: `qwen3_tts_rocm.env.rocm_check()` /
  `qwen3_tts_rocm.env.collect()`.
* The check is strictly read-only and has a **never-raise guarantee**: even on
  machines without torch it degrades to plain error lines instead of a stack
  trace.
* Line prefixes: `ERROR:` blocks you, `WARN:` matters but continues,
  `INFO:` is an advisory only. A bilingual summary tail reports HIP state,
  ROCm version, GPU count/arch and warning/error counts.

---

## Environment errors (`ERROR:`)

### E1 · `PyTorch is not installed or not importable ...`

Torch is missing from the active interpreter (wrong venv, or the package was
installed without the wheel stack).

Fix — install the AMD ROCm build into **this** environment:

```bash
pip install torch --index-url https://repo.amd.com/rocm/whl-multi-arch/
```

or simply re-run `bash scripts/install.sh`, which pins the full stack
(`torch[device-gfx1151]==2.12.0+rocm7.14.0`, `torchvision`, `torchaudio`)
from that index.

### E2 · `The installed PyTorch is NOT an AMD ROCm/HIP build ...`

A CUDA or CPU-only torch shadows the ROCm one (common after adding packages
that pull `torch` from PyPI). The emitted hint names the exact remedy — run
the verbatim reinstall command:

```bash
pip uninstall torch && pip install torch --index-url https://repo.amd.com/rocm/whl-multi-arch/
```

Verify afterwards: `python -c "import torch; print(torch.__version__, torch.version.hip)"`
should print `2.12.0+rocm7.14.0` and a `7.x`-series HIP value.

### E3 · `AMD ROCm torch found but torch.cuda.is_available() is False`

The right wheel is loaded but no HIP device is visible. As the message says,
check:

1. **Driver load** — `sudo dmesg | grep -i amdgpu`; `rocm-smi` must list the
   iGPU. A kernel without a matching `amdgpu` DRM driver cannot expose the
   device to userspace.
2. **Device permissions** — see W2 below (`/dev/kfd`, groups `render` +
   `video`, relogin).
3. Re-run `bash scripts/verify_gpu.sh` after fixing; success prints
   `SPIKE-GPU-OK`.

If `HSA_OVERRIDE_GFX_VERSION` sneaked into your environment, also see W3 —
it actively hurts on this hardware instead of helping.

---

## Warnings & advisories (`WARN:` / `INFO:`)

### W1 · `GPU enumeration failed: module 'torch.cuda' has no attribute 'get_device_count'`

Some ROCm wheels only ship `torch.cuda.device_count`. The probe already tries
both names; when enumeration still fails the report stays usable (devices may
still be found later at load time). Treat as informational unless loads then
fail too.

### W2 · `/dev/kfd exists but is not read/writable by this process`

and its sharper twin: `No HIP device visible AND /dev/kfd lacks read/write
permission for this user`.

`/dev/kfd` is the ROCm userspace entry point; desktop sessions typically grant
it to the `video` and/or `render` groups. Fix per the emitted text — join both
groups, log out and back in (group membership is resolved at login):

```bash
sudo usermod -aG video,render "$USER"
# relogin, then verify:
id -nG "$USER"                 # must contain video and render
ls -l /dev/kfd /dev/dri        # group column matches what id showed
```

The kfd gate fires twice on purpose with different wordings: once when the HIP
stack came up but the process cannot use the device again later (model init
may fail), and once, as "permissions are the most likely root cause", when no
device was visible at all.

### W3 · `HSA_OVERRIDE_GFX_VERSION="..." is set, but gfx1151 needs NO override on ROCm 7.x`

gfx1151 is natively supported by the ROCm 7.x toolchain that ships in the AMD
wheels. The override forces mis-targeted code objects and breaks more than it
fixes. Remove it:

```bash
unset HSA_OVERRIDE_GFX_VERSION      # and delete any export line in shell profiles / launch scripts
```

### W4 · `CUDA_VISIBLE_DEVICES=... is set — on ROCm it applies to the HIP device order as well`

Not an error: the variable keeps working under ROCm/HIP with CUDA semantics
(commas select/order devices, `-1` hides all GPUs). Nothing to change unless
you expected otherwise.

### W5 / I · `unified-memory APU/iGPU detected — VRAM is shared with system RAM ...`

Contains the standing advisory, verbatim
(`GTT_HINT` in `env.py`):

> If generation hits out-of-memory on unified-memory APUs, consider lowering
> max_new_tokens or switching to a 0.6B model.

There is no dedicated VRAM here — allocations land in the same LPDDR5X pool
as your desktop session (GTT-style accounting), so long generations compete
with everything else in RAM. See the OOM playbook below for the concrete
levers.

---

## Out-of-memory playbook

Symptoms: OOM aborts mid-generation, `MemEfficient attention` style failures,
system-wide freezes under load. Apply the levers in this order (this is the
GTT_HINT above, expanded):

1. **Lower `max_new_tokens`.** Every service here already defaults to the
   512-token guardrail (`demo/backend.py::DEFAULT_GEN_KWARGS`) precisely
   because the official default of 2048 can degenerate into minutes-long
   runs (~23 min measured once on this host) while accumulating KV cache all
   the way. Override consciously via the demo's Advanced accordion or
   `gen_kwargs={"max_new_tokens": N}` — smaller N bounds peak memory.
2. **Switch to a 0.6B model.** Aliases `custom-voice-0.6b` / `base-0.6b`
   roughly halve resident weights versus their 1.7B siblings (measured load
   footprint ≈ 2.3 GiB vs ≈ 4.2 GiB in `evidence/load-smoke.txt`). Switch via
   the sidebar model switcher or:
   ```bash
   bash scripts/run_demo.sh --alias custom-voice-0.6b
   ```
3. **Unload before reloading.** Exactly one model should stay resident;
   `loader.unload(model)` / `SynthesisService.unload_all()` delete weights,
   collect garbage and empty the caching allocator — best-effort, never
   raises. Switching aliases inside the demo does this automatically (LRU of
   size one); manually loading several models in one interpreter does NOT.
4. Close other GPU-pool consumers first (browsers/compositors share the same
   unified memory), then retry.

---

## Download failures

The downloader (`scripts/download_models.sh`,
`qwen3_tts_rocm.models.download`) tries **ModelScope first**, falls back to
**hf-mirror.com**, with one retry per source per repo. When everything fails
you get a `RuntimeError` listing every attempt plus **both manual URLs**, e.g.
for alias `custom-voice`:

```text
Manual fix (手动下载): place the files under <models-root>/Qwen3-TTS-12Hz-1.7B-CustomVoice
  https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Recovery options, cheapest first:

* **Just re-run** `bash scripts/download_models.sh [alias]` — interrupted
  transfers resume natively in both transports and fully-downloaded repos
  (marker plus weights, see below) are skipped entirely.
* **Proxy/network partition:** ModelScope is reachable from ordinary CN
  networks (that is why it is first); if your network can reach
  huggingface.co through `hf-mirror.com` only, keep the default auto mode.
  Force a single channel via the API if you know which works:
  `download("custom-voice", source="modelscope")` (or `"hfmirror"`).
  The mirror endpoint is set around each call
  (`HF_ENDPOINT=https://hf-mirror.com`) and restored afterwards; nothing
  global leaks.
* **Manual placement:** download either URL yourself and unpack so the model
  folder sits at `<models-root>/<flattened-name>` (default models root:
  `<repo>/models`, overridable via `$QWEN3_TTS_ROCM_MODELS_DIR`). The
  downloader then treats the target as done and skips it on every future run
  — provided the folder also holds weights (see the skip rule below); a
  manually placed folder *without* any `*.safetensors` is re-fetched on the
  next run.

### What "already downloaded" means (`is_downloaded` semantics)

There are two distinct completeness levels — don't conflate them:

* **Library predicate** (`models.is_downloaded(ref)`, default): the resolved
  directory exists and holds **either** `config.json` **or** the `.ok`
  marker written after a successful fetch (`models.mark_ok`). This is the
  level `loader.load()` checks to refuse loading a not-yet-downloaded model
  instead of fetching multi-GB weights behind your back.
* **Downloader skip rule** (`download(resume=True)`, the default): the
  target must *additionally* hold at least one `*.safetensors` weight file
  (`is_downloaded(..., require_weights=True)`). A config-only or
  marker-only leftover from an interrupted fetch therefore does **not**
  count as done — the downloader re-fetches it, and both transports resume
  interrupted transfers natively so the fetch continues where it left off.

Consequences worth knowing:

* A partially-fetched snapshot that happens to include `config.json` is no
  longer mistaken for complete by the downloader: the next
  `bash scripts/download_models.sh [alias]` run re-fetches it automatically.
  If a fetch still looks corrupted (e.g. truncated weights), force a clean
  redo by deleting that model folder and re-running the downloader, or call
  `download(..., resume=False)` from Python to bypass the skip entirely and
  re-write every file plus the `.ok` marker.
* The `.ok` marker lives *inside* the model folder; copying folders between
  machines/checkouts preserves completion state automatically.

---

## flash-attn requests on ROCm

Upstream prints this banner at import time (harmless here; auto-suppressed by
the loader's fd-level capture — see "First-run log expectations" below,
re-enable with `QWEN3_TTS_ROCM_VERBOSE_IMPORT=1`):

```text
********
Warning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.
********
```

The validated wheel stack (AMD's `repo.amd.com` ROCm 7.14.0 index) does not
include flash-attn, and FlashAttention is not enabled by this project, so the
manual PyTorch attention path is the supported one. The loader encodes the
policy: on HIP GPUs it defaults to `attn_implementation="sdpa"`; explicitly
requesting `"flash_attention_2"` raises a `RuntimeError` suggesting you omit
the argument or pass `"sdpa"` instead. Experimental AOTriton attention paths
(`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) exist upstream but are left off
by this project.

Related informational warnings during real synthesis (from transformers'
SDPA integration): `Flash/Mem Efficient attention on Current AMD GPU is
still experimental...` — informational, the selected path still runs.

---

## Harmless console noise

| Noise you will see | Meaning / action |
|---|---|
| `/bin/sh: 1: sox: not found` (once, early) | Printed by upstream code probing for the SoX binary at import time. Purely cosmetic: audio paths used here do not need SoX. Auto-suppressed by the loader's fd-level capture (see below); shows again with `QWEN3_TTS_ROCM_VERBOSE_IMPORT=1`. Install `sox` if you want the line gone even then; functionality is identical either way |
| `MIOpen(HIP): Warning [IsEnoughWorkspace] ...` and `a_grid_desc_m_ak_container_...` lines | MIOpen/Composable-kernel diagnostics emitted while kernels compile during warm-up — heavy only on first encounters of a shape and cached across runs afterwards. Ignore them; they are stderr chatter, not errors |
| `Setting pad_token_id to eos_token_id...` | Normal transformers generation-config notice at the start of each generation |

### First-run log expectations (首次运行日志预期)

What a healthy **very first** run looks like, so a scrolling terminal is not
mistaken for a hang:

* **Hundreds of kernel-tuning lines are normal.** On the very first encounter
  with a shape, MIOpen/Composable-kernel can emit **several hundred**
  `MIOpen(HIP): ...` tuning lines; they are one-off and cached across all
  later runs. 首次运行出现数百行内核调优日志属正常，之后缓存复用，不再刷屏。
* **One bilingual expectation line from the loader.** Before the first load
  the loader prints to stderr
  `[qwen3-tts-rocm] 加载 <model> … 首次加载需数十秒，终端将出现大量内核日志（属正常）/ loading; verbose kernel logs are expected on first run` —
  a quiet terminal during the following tens of seconds is expected, not a
  freeze. loader 的这行双语提示即为此预期而设。
* **Upstream import banners are auto-suppressed.** The official package
  prints its SoX "not found" ad and flash-attn banner straight to the file
  descriptors at import time; both `loader.load()` and the demo backend wrap
  that lazy import in an fd-level capture (both fds → `/dev/null`, always
  restored), so a default run shows neither. Escape hatch:
  `QWEN3_TTS_ROCM_VERBOSE_IMPORT=1` keeps the import-time output visible when
  debugging (the capture covers the import only — model loading and
  generation output always show). 上游导入横幅（SoX 广告 / flash-attn）默认已被
  fd 级捕获自动抑制；调试时设 `QWEN3_TTS_ROCM_VERBOSE_IMPORT=1` 可重新看到。
* **Quieting the loader line.** `QWEN3_TTS_ROCM_QUIET=1` silences the
  bilingual expectation announcement entirely. 设 `QWEN3_TTS_ROCM_QUIET=1`
  可关闭 loader 的提示行。

---

## Still stuck?

Re-run `qwen3-tts-rocm-check`, capture full command output including stderr
(set `QWEN3_TTS_ROCM_VERBOSE_IMPORT=1` if the upstream import-time banners
belong in your log), open an issue in this repository's tracker, and quote
the diagnostic block verbatim along with `rocm-smi` output. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the hardware-log etiquette (text
logs only, no binaries or generated audio).
