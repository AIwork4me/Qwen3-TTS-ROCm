# Qwen3-TTS-ROCm

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20ROCm%207.14.0-orange.svg)](https://rocm.docs.amd.com/)
[![Hardware](https://img.shields.io/badge/hardware-gfx1151-red.svg)](https://rocm.docs.amd.com/)

**English** | [简体中文](README_CN.md)

Run the official Qwen3-TTS text-to-speech on the AMD Ryzen AI Max+ PRO 395 /
Radeon 8060S iGPU (`gfx1151`) with **zero patches** to the official stack.
One command installs AMD's pinned ROCm 7.14.0 PyTorch wheels, another
downloads the six official checkpoints, a third launches an enhanced
bilingual five-tab Gradio demo on `http://localhost:8000`. Every synthesis
call stays on unmodified official APIs —
[详见下文 Why / see Why below](#why-this-project-exists).

## Requirements

| Component | Requirement |
|---|---|
| APU / iGPU | AMD Ryzen AI Max+ PRO 395 with Radeon 8060S Graphics — architecture `gfx1151` (Strix Halo class); developed and validated on exactly this device |
| Memory | Unified-memory class hardware: 94 GB LPDDR5X pool shared by CPU and iGPU (~80 GiB visible to torch/HIP) |
| Kernel | Linux with the `amdgpu` DRM driver bound to the iGPU (verify with `rocm-smi`); validated on kernel `6.17.0-1032-oem` |
| ROCm | ROCm 7.14.0-era wheels from AMD's pip index `https://repo.amd.com/rocm/whl-multi-arch/`: `torch[device-gfx1151]==2.12.0+rocm7.14.0`, `torchvision[device-gfx1151]==0.27.0+rocm7.14.0`, `torchaudio==2.11.0+rocm7.14.0` — installed automatically by `scripts/install.sh`, you never type these by hand |
| Python | ≥ 3.10 (validated on 3.12) |
| Disk | ~18 GB free for the six official repositories, which each bundle their own speech-tokenizer copies (weights live under `models/` and are never committed) |
| Network | ModelScope (`modelscope.cn`) reachable — the default channel already works from CN networks; when `huggingface.co` is blocked the fallback transport routes through `hf-mirror.com`, so no VPN is required |

The six official repositories behind the aliases, with their approximate
download sizes (numbers identical to `scripts/download_models.sh --help`):

| 别名 / alias | 仓库 / repo | 大小 / size |
|---|---|---|
| `tokenizer` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 651M |
| `custom-voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 4.3G |
| `voice-design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 4.3G |
| `base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 4.3G |
| `custom-voice-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 2.4G |
| `base-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 2.4G |
| — | **合计 / Total** | **≈ 18 GB** |

## Quickstart

Five commands from zero to a talking browser tab:

```bash
git clone https://github.com/AIwork4me/Qwen3-TTS-ROCm.git
cd Qwen3-TTS-ROCm
bash scripts/install.sh          # venv + pinned AMD ROCm wheels + editable install + GPU gate
bash scripts/download_models.sh  # six official checkpoints, ModelScope-first, hf-mirror fallback
bash scripts/run_demo.sh         # enhanced demo -> http://localhost:8000
```

No need for all six up front: subset downloads are supported, e.g.
`bash scripts/download_models.sh tokenizer custom-voice  # ≈5GB, enough for the Python snippet and preset voices / cloning`.

Notes:

* `scripts/install.sh` is idempotent (safe to re-run) and ends with a GPU
  sanity gate that prints `SPIKE-GPU-OK` on success.
* Model weights download to `<repo>/models/` (overridable via
  `$QWEN3_TTS_ROCM_MODELS_DIR`). Interrupted transfers resume; already-complete
  repos short-circuit so re-runs are cheap.
* `scripts/install.sh --with-models` chains the download step.
* After install, `qwen3-tts-rocm-check` runs the bilingual environment
  self-check at any time (read-only, never raises).

### Minimal Python usage

The smallest program that proves the loader promise — load once through us,
then everything else is pure official API, mirroring the upstream README
quickstart:

```python
from qwen3_tts_rocm import loader

tts = loader.load("custom-voice")            # sdpa/bf16 defaults on gfx1151
wavs, sr = tts.generate_custom_voice(text="你好，ROCm。", language="auto",
                                     speaker=tts.get_supported_speakers()[0])

import soundfile as sf
sf.write("hello-rocm.wav", wavs[0], sr)  # 保存 / save
```

**Expected output / 预期输出：** the first `loader.load` takes tens of seconds.
The terminal prints one loader status line, and — only with
`QWEN3_TTS_ROCM_VERBOSE_IMPORT=1` — the upstream import banners; verbose
MIOpen kernel-tuning lines on the very first run are normal and are cached
afterwards (see [docs/troubleshooting.md](docs/troubleshooting.md)).

* `loader.load("custom-voice")` resolves the registry alias to
  `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` under your models directory and
  applies ROCm smart defaults (`device_map=auto`, `dtype=bfloat16`,
  `attn_implementation=sdpa` on HIP). The returned `tts` is the native
  official object; `generate_custom_voice`,
  `get_supported_speakers()`, … are untouched official methods.
* Language values come back from the official getters in **lowercase**
  (`"auto"`, `"chinese"`, `"english"`, …) — pass them lowercase to the API.
  Display casing ("Chinese") is handled inside the demo UI layer.
* Loading never downloads multi-GB weights behind your back: if the target is
  missing you get a `RuntimeError` that names
  `bash scripts/download_models.sh` as the remedy.

## The five-tab Gradio demo

`bash scripts/run_demo.sh` serves the enhanced bilingual (中文/English) demo
application with **five tabs**, backed by a headless `SynthesisService` with
server-side file validation:

1. **Voice Clone** — reference-audio cloning, incl. a save/load voice
   sub-tab that persists reusable `.pt` prompt files.
2. **Preset Speakers** — pick any bundled speaker plus an optional
   instruction, and synthesize instantly.
3. **Voice Design** — describe the voice in natural language and generate.
4. **Codec** — encode→decode roundtrip visualization through the official
   12Hz speech tokenizer, with rate/steps metadata and a downloadable WAV.
5. **History** — play, download or delete the current session's generated
   clips.

A sidebar hosts the model switcher (exactly **one** TTS model stays resident
at a time — LRU slot of one), a live VRAM/GTT status line and an Advanced
sampling-parameter accordion (empty = defaults).

### Queue serialization note

The default lives in the `qwen3-tts-rocm-demo` entry point itself
(`src/qwen3_tts_rocm/cli_demo.py`, where `--concurrency` defaults to 1 per the
single-GPU queue ruling — upstream's default queue concurrency of 16 makes no
sense here); `scripts/run_demo.sh` merely forwards arguments to it. Generation
requests therefore run strictly one at a time, because the single-GPU
unified-memory device keeps only one model resident, and concurrent synthesis
would serialize into the same compute pool anyway. The same note is rendered
visibly in the demo footer. Raise it only if you know why.

### Microphone capture needs HTTPS (or localhost)

Per the browser security model upstream relies on, microphone input in the
Voice Clone / Codec tabs requires a secure context: either open the page on
the machine itself (`http://localhost:8000`) or serve the demo over TLS.
Adapted from the upstream notes, a self-signed pair in one line:

```bash
openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 \
    -out cert.pem -subj "/CN=localhost"
bash scripts/run_demo.sh --ssl-certfile cert.pem --ssl-keyfile key.pem
```

Accept your browser's one-time warning about the self-signed certificate
and microphone capture works over HTTPS from other LAN devices too.

### Screenshots

Captured from the real gfx1151 deployment during the v0.1.0 acceptance run
(bilingual UI, live synthesis through the Gradio queue):

| | |
|---|---|
| ![Voice Clone tab](docs/img/demo-clone.png) | ![Preset Speakers tab](docs/img/demo-customvoice.png) |
| ![Voice Design tab](docs/img/demo-voicedesign.png) | ![Codec tab](docs/img/demo-codec.png) |
| ![History tab](docs/img/demo-history.png) | *Voice Clone · Preset Speakers · Voice Design · Codec · History* |

`demo-voicedesign.png` and `demo-history.png` show the sidebar model switcher
mid-session: `[voice-design] loaded (已驻留)` with live VRAM/GTT readout, a
finished 5.6 s synthesis, and the newest-first history table with preview,
download and delete.

## Performance preview (gfx1151)

Measured median real-time factor — RTF, wall-clock seconds per second of
generated audio, lower is better — on Ryzen AI Max+ PRO 395, bfloat16/sdpa,
short and medium texts, capped at `max_new_tokens=512`:

| Family | Median RTF range |
|---|---|
| Tuned voices, `custom-voice` | 1.31 – 1.51 |
| Tuned voices, `voice-design` | 1.27 – 1.62 |
| Zero-shot voice clone (`base`) | 1.71 – 1.88 |

In other words a few seconds of wait for a few seconds of speech on an
iGPU — usable interactive sentence-scale demos, comfortable batch workloads.
We quote ranges, not latency promises: numbers drift with clocks, thermals,
memory pressure and background load on shared-pool unified memory, so treat
any single session as indicative (full per-cell tables, methodology, n=2
caveats and reproduce block in [`docs/benchmarks.md`](docs/benchmarks.md)).

> **Degenerate-generation warning:** under the upstream default
> `max_new_tokens=2048`, sampling can rarely fall into a degenerate loop that
> keeps rendering for minutes on the iGPU (one measured runaway ran ~23
> minutes). Every service in this repository therefore defaults to a
> **512-token guardrail** — override it knowingly via the demo's Advanced
> accordion or `gen_kwargs`.

<a id="why-this-project-exists"></a>

## Why this project exists

Qwen3-TTS ships as a CUDA-first stack: upstream expects NVIDIA GPUs and the
flash-attn kernel library, and nothing about `gfx1151`-class integrated
graphics worked out of the box. This repository is deliberately *not* a fork
of that code — it is a thin community shim around the **unmodified official
`qwen-tts` package**: environment diagnostics, a smart-default model loader,
a dual-source (ModelScope / hf-mirror) downloader and an enhanced demo UI,
nothing more. The core promise is stated in the
[zero-modification guarantee](#zero-modification-guarantee) below and is
enforced by a dedicated parity test; if you only read one thing before
trusting this repo, make it that section.

<a id="zero-modification-guarantee"></a>

## Zero-modification guarantee

* [`loader.load()`](src/qwen3_tts_rocm/loader.py) returns **exactly what the
  official `qwen_tts.Qwen3TTSModel.from_pretrained` returns** — the native
  model object, never a wrapper. Smart defaults (`bfloat16` + `sdpa` on HIP
  GPUs) are applied only through official, public keyword arguments.
* Proof lives in the test suite:
  [`tests/test_official_demo_parity.py`](tests/test_official_demo_parity.py)
  builds the stock, unmodified upstream
  `qwen_tts.cli.demo.build_demo()` around a model object loaded by *our*
  loader, then executes the demo's own handler closures through Gradio's
  event registry — including one real GPU synthesis. The recorded transcript
  of the green run is archived in
  [`evidence/official-parity.txt`](evidence/official-parity.txt).
* Project policy (see [`CONTRIBUTING.md`](CONTRIBUTING.md) and
  [`NOTICE`](NOTICE)): no vendored or patched upstream source, ever;
  `pyproject.toml` depends on the published `qwen-tts==0.1.1` artifact as-is.

## FAQ & troubleshooting

Symptom → where the fix is documented. The companion guide expands every
`ERROR:` / `WARN:` / `INFO:` line this project's diagnostics emit.

| Symptom or message | Fix documented in |
|---|---|
| `PyTorch is not installed or not importable` | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `The installed PyTorch is NOT an AMD ROCm/HIP build` | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `torch.cuda.is_available() is False` / `/dev/kfd` permission complaints | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `HSA_OVERRIDE_GFX_VERSION` warning | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Download fails on both Hugging Face and ModelScope | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Out-of-memory or runaway-long generations on unified memory | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `flash_attention_2` requested but flash-attn missing | [docs/troubleshooting.md](docs/troubleshooting.md) |
| SoX banner / MIOpen console chatter | [docs/troubleshooting.md](docs/troubleshooting.md) |

Quick answers to the two most common questions:

* *"Is this a model?"* No — models stay in their official repositories and
  download separately (~18 GB total). This project is glue around them.
* *"Does it need the weights in git?"* Never; see `.gitignore` and
  `$QWEN3_TTS_ROCM_MODELS_DIR`.

## Attribution & disclaimer

* Qwen3-TTS and all model weights are the work of the **Alibaba Qwen team**
  ([upstream repository](https://github.com/QwenLM/Qwen3-TTS), Apache-2.0;
  weights carry Alibaba's own Qwen model license — download terms apply, see
  [`NOTICE`](NOTICE)). Model weights are not redistributed here.
* Qwen3-TTS-ROCm is an **unofficial community adaptation**; it is not
  affiliated with, endorsed by, or produced by Alibaba or AMD.
* Short version of the upstream audio-generation disclaimer (condensed from
  the official demo footer):
  *Generated audio may be inaccurate or inappropriate, does not represent
  anyone's views, and is your responsibility to use lawfully — do not create
  unlawful, harmful, deepfake or infringing content.*

  中文（同义简版，自上游页脚节选）：*音频由 AI 模型自动生成，可能不准确或不当，
  不代表任何一方立场；请依法使用，严禁生成违法、有害、深度伪造或侵权内容。*

## Contributing & contact

* Ground rules, dev setup and PR checklist: [`CONTRIBUTING.md`](CONTRIBUTING.md).
  Headline rule: contributions must preserve the thin-shim guarantee above —
  no patched upstream source, no committed weights.
* Bug reports and feature ideas go to this repository's issue tracker.
* Canonical home of this project:
  `https://github.com/AIwork4me/Qwen3-TTS-ROCm` — maintained by [@AIwork4me](https://github.com/AIwork4me).

## License

Code: Apache-2.0 — see [`LICENSE`](LICENSE). Model weights remain governed
by Alibaba's own model license (provenance and download terms in
[`NOTICE`](NOTICE)).
