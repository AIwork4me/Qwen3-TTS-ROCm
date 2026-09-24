<p align="center">
  <img src="docs/hero.jpg" alt="Qwen3-TTS on AMD ROCm hero" width="100%"/>
</p>

# Qwen3-TTS on AMD ROCm

**Official Qwen3-TTS. AMD Radeon. Zero upstream patches.**

> ### North Star
>
> **Official Qwen3-TTS on AMD Radeon — capability by capability, benchmark by benchmark, with zero upstream patches.**
>
> Every capability claim marked ✅ below was exercised end to end on the
> real Radeon 8060S (`gfx1151`) validation host through unmodified official
> `qwen-tts` APIs and links the verbatim evidence transcript that proves it.
> What is not proven is labelled as such. No percentage scores, no
> generalization beyond the validated configuration.

Run the unmodified official [`qwen-tts`](https://github.com/QwenLM/Qwen3-TTS)
package on AMD Ryzen AI Max+ PRO 395 / Radeon 8060S (`gfx1151`): one command
installs AMD's pinned ROCm 7.14.0 PyTorch wheels, the next downloads the
official checkpoints, the third opens a bilingual six-tab Gradio demo on
`http://localhost:8000`. Every synthesis call stays on official APIs.

**English** | [简体中文](README_CN.md)

[![CI](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/ci.yml)
[![GPU CI](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/gpu-nightly.yml/badge.svg?branch=main)](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/gpu-nightly.yml)
[![Release](https://img.shields.io/github/v/release/AIwork4me/Qwen3-TTS-ROCm)](https://github.com/AIwork4me/Qwen3-TTS-ROCm/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![ROCm](https://img.shields.io/badge/ROCm-7.14.0-orange.svg)](https://rocm.docs.amd.com/)
[![Hardware](https://img.shields.io/badge/hardware-gfx1151%20%7C%20Radeon%208060S-red.svg)](docs/benchmarks.md#platform)

Unofficial community project — not affiliated with or endorsed by Alibaba or
AMD. See [Attribution](#attribution--disclaimer).

## Verified, not promised

| Validation | Result |
|---|---|
| Official model repositories | **6 / 6 load-validated** — 5 TTS checkpoints + tokenizer |
| Automated tests | **353 / 353 on validation host** — 313 CPU + 40 real-GPU (40 since 2026-09-24: +2 reusable-prompt batch tests) |
| Patches to upstream `qwen-tts` | **0** — enforced by a dedicated parity test |
| GPU · ROCm | Radeon 8060S (`gfx1151`) · ROCm 7.14.0 (`torch 2.12.0+rocm7.14.0`) |
| Precision / attention | bfloat16 · PyTorch SDPA — FlashAttention not used in the validated stack |
| Evidence | Verbatim transcripts in [`evidence/`](evidence/README.md) |

### Capability matrix (Radeon 8060S · `gfx1151`)

One row per capability, each green cell linked to the verbatim artifact that
proves it. States: ✅ **Radeon E2E validated** (real synthesis / real run on
the validation host, asserted sane) · 🟡 **partial — load-only** (loads, no
functional validation) · ⬜ **not validated** · 🚫 **not exposed upstream or
intentionally not claimed**.

| Capability | Model | Radeon status | Evidence |
|---|---|---|---|
| CustomVoice generation (preset / custom speakers) | 1.7B | ✅ E2E validated | [`gen-customvoice.txt`](evidence/gen-customvoice.txt) · RTF in [`benchmark.json`](evidence/benchmark.json) |
| CustomVoice generation | 0.6B | ✅ E2E validated | [`gpu-suite-2026-09-20.txt`](evidence/gpu-suite-2026-09-20.txt) · [`benchmark-06b-2026-09-20.json`](evidence/benchmark-06b-2026-09-20.json) |
| VoiceDesign (text-described voice creation) | 1.7B | ✅ E2E validated | [`gen-voicedesign.txt`](evidence/gen-voicedesign.txt) · RTF in [`benchmark.json`](evidence/benchmark.json) |
| Voice Clone — zero-shot cloning from reference audio | 1.7B Base | ✅ E2E validated | [`gen-voiceclone.txt`](evidence/gen-voiceclone.txt) · RTF in [`benchmark.json`](evidence/benchmark.json) |
| Base family (zero-shot cloning + fine-tuning base) | 0.6B | ✅ E2E validated | [`gpu-suite-2026-09-20.txt`](evidence/gpu-suite-2026-09-20.txt) · [`benchmark-06b-2026-09-20.json`](evidence/benchmark-06b-2026-09-20.json) |
| Reusable clone prompt (`create_voice_clone_prompt` → save → load → reuse) | 1.7B & 0.6B Base | ✅ E2E validated | [`gen-voiceclone.txt`](evidence/gen-voiceclone.txt) · [`gpu-suite-2026-09-20.txt`](evidence/gpu-suite-2026-09-20.txt) |
| Design → Clone → Reuse (Voice Studio one-click flow) | VoiceDesign 1.7B + Base | ✅ E2E validated | [`voice-workflow-2026-09-20.txt`](evidence/voice-workflow-2026-09-20.txt) · [`voice-workflow-2026-09-20.json`](evidence/voice-workflow-2026-09-20.json) |
| Official API batch inference (list-of-texts, zero custom batching) | 0.6B & 1.7B CustomVoice · 1.7B VoiceDesign · 0.6B & 1.7B Base via reusable clone prompt | ✅ E2E validated — B ∈ {1,2,4,8} all green, max validated B = 8 for all five families (this host/config only) | [`batch-inference-gfx1151-2026-09-24.txt`](evidence/batch-inference-gfx1151-2026-09-24.txt) · [`batch-inference-gfx1151-2026-09-24.json`](evidence/batch-inference-gfx1151-2026-09-24.json) |
| Long-text synthesis & sustained-session stability | 1.7B CustomVoice | ✅ E2E validated — ladder 8/8 tiers to 892 chars / 58 s audio (natural EOS, no max-length claim); 60-min resident soak 815/815 OK, zero failures; 10/10 load→generate→unload cycles (RSS plateau; no leak-free claim) | [`long-text-gfx1151-2026-09-24.txt`](evidence/long-text-gfx1151-2026-09-24.txt) · [`soak-gfx1151-2026-09-24.txt`](evidence/soak-gfx1151-2026-09-24.txt) |
| 12Hz tokenizer codec (encode → decode roundtrip) | Tokenizer-12Hz | ✅ E2E validated | [`tokenizer-codec.txt`](evidence/tokenizer-codec.txt) |
| Multilingual matrix — all 10 officially supported languages, end to end | 1.7B CustomVoice + VoiceDesign + Base | ✅ E2E validated | [`multilingual-matrix.txt`](evidence/multilingual-matrix.txt) · [`multilingual-matrix.json`](evidence/multilingual-matrix.json) |
| ASR-based content correctness + clone-similarity controls (quality v2) | 1.7B CustomVoice ×10 languages · 1.7B Base clones | ✅ measured — whisper-small CER/WER 0.00–0.05 for 9/10 languages (German 0.55 outlier, ASR-agreement only); clone cosines positives 0.63–0.68 > negatives 0.58–0.60; no thresholds, no composite score, no MOS | [`quality-v2-gfx1151-2026-09-24.txt`](evidence/quality-v2-gfx1151-2026-09-24.txt) · [docs](docs/quality-v2.md) |
| Fine-tuning (official `finetuning/` SFT workflow) | 1.7B Base · 0.6B Base | ✅ scoped — **execution-only smoke** (1.7B: prep → 12 steps → save → reload → sane synthesis, 2026-09-20; 0.6B: same protocol ×2 independent runs, 2026-09-24 — prep → 12 steps → save → fingerprint → reload → sane synthesis). NO convergence / quality / multi-speaker claims. Two disclosed upstream blockers for out-of-the-box ROCm fine-tuning: PR #373 (OPEN; flash-attn hardcode, issue #372) and the 0.6B text-projection omission (sft_12hz.py adds text+codec embeddings without the mandatory `text_projection`; shape error by construction on 0.6B; workaround confined to the gitignored clone, upstream issue drafted) ([docs](docs/finetuning-rocm.md#06b-base-execution-validation-v021-task-6-2026-09-24)) | 1.7B: [`finetune-smoke-2026-09-20.txt`](evidence/finetune-smoke-2026-09-20.txt) · 0.6B: [`finetune-06b-gfx1151-run1-2026-09-24.txt`](evidence/finetune-06b-gfx1151-run1-2026-09-24.txt) · [`run2`](evidence/finetune-06b-gfx1151-run2-2026-09-24.txt) · PR #373 chain: [`upstream-372-root-cause.md`](evidence/upstream-372-root-cause.md) · [`upstream-372-e2e-run1.txt`](evidence/upstream-372-e2e-run1.txt) · [`upstream-372-e2e-run2.txt`](evidence/upstream-372-e2e-run2.txt) |
| Instruction control on CustomVoice | 0.6B | 🚫 not exposed upstream (wrapper silently ignores `instruct`) — pinned by tests | Task 0 audit: [`ground-truth-2026-09-20.md`](evidence/ground-truth-2026-09-20.md) |
| vLLM-Omni serving | — | 🟡 partial — offline proven on the CURRENT stack (2026-09-24: vllm 0.30.0+rocm723 + vllm-omni 0.30.0rc1, upstream end2end.py verbatim): all three 1.7B task families — CustomVoice, VoiceDesign, Base voice-clone — finite non-silent 24 kHz outputs; isolated venv, no serving/streaming/perf claims yet | [`vllm-omni-current-gfx1151-2026-09-24.txt`](evidence/vllm-omni-current-gfx1151-2026-09-24.txt) · [`vllm-omni-current-gfx1151-2026-09-24.json`](evidence/vllm-omni-current-gfx1151-2026-09-24.json) · prior 0.28.0 snapshot: [`vllm-omni-feasibility-2026-09-21.txt`](evidence/vllm-omni-feasibility-2026-09-21.txt) |
| True streaming inference | — | 🚫 not exposed upstream — measured 2026-09-21 on gfx1151: qwen-tts 0.1.1's official Python API delivers audio only at completion (exactly 1 chunk every run; TTFB == total wall) | [`streaming-2026-09-21.txt`](evidence/streaming-2026-09-21.txt) · [roadmap](#true-streaming-inference) |

Conceptual boundary worth stating plainly (the demo and docs honour it):
**CustomVoice is preset-speaker / custom-voice generation** — it does not
clone from reference audio. **Base is the zero-shot voice-cloning and
fine-tuning family.** The per-checkpoint and multilingual tables below are
detail views of the same evidence.

Per-checkpoint validation level (load = loads through `loader.load`; E2E
Generate = real synthesis asserted sane on GPU; Benchmark = archived RTF run):

| Model | Load | E2E Generate | Benchmark |
|---|---|---|---|
| 1.7B CustomVoice | ✅ | ✅ | ✅ |
| 1.7B VoiceDesign | ✅ | ✅ | ✅ |
| 1.7B Base (clone) | ✅ | ✅ | ✅ |
| 0.6B CustomVoice | ✅ | ✅ | ✅ (see `evidence/benchmark-06b-2026-09-20.json`) |
| 0.6B Base (clone) | ✅ | ✅ | ✅ (same) |
| 12Hz Tokenizer | ✅ | codec ✅ | n/a |

Note: 0.6B CustomVoice has no instruction control upstream; the demo and
docs reflect that boundary.

### Multilingual capability matrix

Every officially supported language, exercised end to end on the Radeon GPU
(10 CustomVoice + 10 VoiceDesign generations + 4 representative
cross-lingual clone pairs via the 1.7B Base model; one model resident at a
time). Reproduce with
`.venv/bin/python scripts/validate_languages.py 2>&1 | tee evidence/multilingual-matrix.txt`
(transcript: `evidence/multilingual-matrix.txt`, machine-readable rows:
`evidence/multilingual-matrix.json`).

| Language | CustomVoice | VoiceDesign | Clone (cross-lingual) |
|---|---|---|---|
| Chinese | ✅ | ✅ | ✅ (ref: en, fr) |
| English | ✅ | ✅ | ✅ (ref: zh, ja) |
| Japanese | ✅ | ✅ | — |
| Korean | ✅ | ✅ | — |
| German | ✅ | ✅ | — |
| French | ✅ | ✅ | — |
| Russian | ✅ | ✅ | — |
| Portuguese | ✅ | ✅ | — |
| Spanish | ✅ | ✅ | — |
| Italian | ✅ | ✅ | — |

✅ = end-to-end generation completed on the validated Radeon 8060S
(`gfx1151`) host (waveform sanity: finite,
non-silent, valid sample rate, bounded duration) — see
`evidence/multilingual-matrix.json`. This is NOT a pronunciation-quality
claim. Cross-lingual clone coverage is representative (4 pairs), not
exhaustive.

### Voice Design → reusable voice (Voice Studio)

The official demo never wired its own capabilities end to end: you could
design a voice in one tab and clone from uploaded audio in another, but
turning a *described* voice into a *reusable* one meant a manual
download/re-upload hop. `qwen3_tts_rocm.voice_workflow` closes that gap using
ONLY the three official APIs — `generate_voice_design`,
`create_voice_clone_prompt`, `generate_voice_clone` — on unmodified
`loader.load` objects, with the official model split (design runs on the
VoiceDesign checkpoint; prompt creation and reuse run on Base — the official
wrapper hard-gates each call on `tts_model_type`):

```python
from qwen3_tts_rocm import loader, voice_workflow

vd, bc = loader.load("voice-design"), loader.load("base")
res = voice_workflow.design_voice(            # preview + reusable prompt items
    vd, prompt_model=bc, text="今天的天气真不错，适合去公园散步。",
    language="Auto", description="年轻女性，声音清亮，语速轻快")
voice_workflow.save_voice(res, "voices/bright.pt")   # official payload + meta
loader.unload(vd)                             # Base alone remains resident
res2 = voice_workflow.load_voice("voices/bright.pt")
wav, sr, gen_s = voice_workflow.reuse_voice(  # any new sentence, same voice
    bc, prompt_items=res2.prompt_items, text="晚风轻轻吹过湖面。", language="Auto")
```

The transcript rule is enforced by construction: the `ref_text` inside every
prompt item is exactly the text that generated the reference audio (one
variable feeds both official calls). Saved voices are official-demo files —
the `"items"` key is byte-format-identical to the upstream demo's
`{"items": [asdict(item) ...]}` payload, so the stock demo can load them too;
a `voice_meta` sidecar in the same `.pt` keeps the description/language/
ref_text provenance.

The demo's **⑥ Voice Studio (音色工坊)** tab runs this as a first-class
one-click flow — describe → preview → save → reuse — with no download or
re-upload anywhere (the saved-voice dropdown replaces the file round-trip).
Three-phase latency on the validation host (Radeon 8060S, bf16, every
generation `max_new_tokens=512`, phases recorded separately — never
collapsed): **design ≈ 5.5 s · prompt creation ≈ 0.3 s · reuse ≈ 5.3–6.4 s**
per sentence. Reproduce with
`.venv/bin/python -m pytest tests/test_voice_workflow.py -m gpu -v -s`
(transcript: `evidence/voice-workflow-2026-09-20.txt`, machine-readable
timings: `evidence/voice-workflow-2026-09-20.json`).

The CPU-only CI matrix collects the same 313 CPU tests on Python 3.10 / 3.11 /
3.12, with 1 HIP-gated test skipped because no AMD GPU is present on the
runner — plus, in a plain `.[dev]` environment without the optional
`[quality]` extra, 15 further *visible* skips (12 jiwer + 3 resemblyzer
quality-benchmark tests; fixed to skip instead of error by the 2026-09-21
claims audit after Task 14's unguarded `import jiwer` had turned CI red).
On the validated ROCm host that test also runs, giving 313 CPU +
38 GPU = 351 / 351 at that time (suite since grew to 313 + 40 = 353 on 2026-09-24 with the batch-inference tests). First verified CI run on this suite: all jobs green at
push `8815238` ([run 35526426415](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/runs/35526426415),
`251 passed, 1 skipped, 38 deselected` per Python job — verified 2026-09-21
when the suite stood at 252 CPU tests; the suite has since grown to the
counts above. Transcript: `evidence/ci-2026-09-21-8815238.txt`).

**GPU CI state:** **LIVE** — the self-hosted GPU regression workflow
(`.github/workflows/gpu-nightly.yml` — nightly short suite + manual
full-weekly matrix on the `gfx1151` host) ran its first real green run on
2026-09-23 ([run 35857806038](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/runs/35857806038),
`gpu-short`, 32 GPU test nodes, commit `a3a8a75`) and its first green
`full-weekly` run on 2026-09-24 ([run 35964051504](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/runs/35964051504),
commit `09ca9fd`: all 38 GPU test nodes passed in 314 s, `verify_gpu.sh`
stack sanity green, RTF benchmark replication across the three 1.7B
aliases — 12 cells, median RTF 1.27–1.40 — with the benchmark JSON
persisted as a downloadable run artifact). The nightly window is
02:00 local (+08:00) = 18:00 UTC, and it runs when the validation host is
powered/online at the window — a missed window is not a regression signal
and can be re-dispatched manually per the runbook. Filter-selection
proofs: `evidence/gpu-ci-prep-validation.txt`; first-run transcript:
`evidence/gpu-ci-first-green-2026-09-23.txt`; full-weekly transcript:
`evidence/gpu-ci-full-weekly-first-green-2026-09-24.txt`; runbook:
[`docs/development/gpu-ci-runbook.md`](docs/development/gpu-ci-runbook.md).

The flagship demo tab, captured live on the validation machine:

![Preset Speakers tab synthesizing on Radeon 8060S](docs/img/demo-customvoice.png)

🔊 **Hear it** — [5.5 s sample (Mandarin)](evidence/demo-rest-gen-zh.wav)
generated through the demo's REST API during the validation run recorded in
[`evidence/demo-smoke.txt`](evidence/demo-smoke.txt).

## Quick Start

### 🚀 Try it first (~5 GB download)

You do **not** need all six checkpoints to get started. The `tokenizer` +
`custom-voice` subset (~5 GB) is enough for the Python snippet below and the
demo's **Preset Speakers** and **Codec** tabs:

```bash
git clone https://github.com/AIwork4me/Qwen3-TTS-ROCm.git
cd Qwen3-TTS-ROCm
bash scripts/install.sh                              # venv + pinned AMD ROCm wheels + GPU gate
bash scripts/download_models.sh tokenizer custom-voice   # ~5 GB, ModelScope-first
bash scripts/run_demo.sh                             # -> http://localhost:8000
```

Or skip the UI — the smallest program that proves the loader promise (load
once through us, then it's pure official API, mirroring the upstream
quickstart):

```python
from qwen3_tts_rocm import loader

tts = loader.load("custom-voice")            # sdpa/bf16 defaults on gfx1151
wavs, sr = tts.generate_custom_voice(text="你好，ROCm。", language="auto",
                                     speaker=tts.get_supported_speakers()[0])

import soundfile as sf
sf.write("hello-rocm.wav", wavs[0], sr)  # 保存 / save
```

Save it as `hello.py` and run it **inside the installer's venv**:
`source .venv/bin/activate` once per terminal, then `python hello.py` — or
call `.venv/bin/python hello.py` directly. The system `python` cannot see the
package, and `qwen3-tts-rocm-check` needs the activated venv too.

Lighter still: `bash scripts/download_models.sh tokenizer custom-voice-0.6b`
(~3 GB) and use the `custom-voice-0.6b` alias in the snippet.

**What to expect:** model load time depends strongly on filesystem cache
state — 2.2–4.9 s per in-session load on the validation host, longer when
cold, plus one-time MIOpen kernel tuning on the very first run (cached
afterwards). The terminal prints one loader status line; MIOpen console
chatter on first run is normal ([troubleshooting](docs/troubleshooting.md)).
Weights download to `models/` (override with `$QWEN3_TTS_ROCM_MODELS_DIR`);
interrupted transfers resume, and re-runs short-circuit what's already there.

### Full experience (~18 GB)

```bash
bash scripts/download_models.sh   # all six official repositories
bash scripts/run_demo.sh
```

| Alias | Official repository | Size |
|---|---|---|
| `tokenizer` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 651M |
| `custom-voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 4.3G |
| `voice-design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 4.3G |
| `base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 4.3G |
| `custom-voice-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 2.4G |
| `base-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 2.4G |

The demo starts even with a partial download; tabs whose model family is
missing say so instead of failing mysteriously. `scripts/install.sh
--with-models` chains install + download; `qwen3-tts-rocm-check` runs the
bilingual environment self-check any time (read-only, never raises).

## What You Get

* **`qwen3-tts-rocm-check`** — bilingual ROCm environment self-check with a
  never-raise guarantee.
* **`loader.load(alias)` / `loader.unload()`** — smart defaults
  (`device_map=auto`, `bfloat16`, `sdpa` on HIP) applied through official
  kwargs; returns the **native official model object**, never a wrapper.
  Loading never downloads weights behind your back — a missing model raises
  a `RuntimeError` that names the download command.
* **Six-alias downloader** — ModelScope-first with `hf-mirror.com` fallback
  (the recorded validation host downloaded everything from a CN network
  without a VPN; other networks may vary), resume support, per-alias or bulk.
* **Six-tab bilingual Gradio demo** (中文/English): ① Voice Clone
  (incl. save/load reusable voice prompts) · ② Preset Speakers ·
  ③ Voice Design · ④ Codec roundtrip · ⑤ History · ⑥ Voice Studio
  (音色工坊 — describe → preview → save → reuse, no download/re-upload hop).
  Sidebar model switcher (one model resident at a time), live VRAM/GTT
  readout, advanced sampling accordion.
  <details>
  <summary>Tab screenshots (①–⑤)</summary>

  | | |
  |---|---|
  | ![Voice Clone tab](docs/img/demo-clone.png) | ![Preset Speakers tab](docs/img/demo-customvoice.png) |
  | ![Voice Design tab](docs/img/demo-voicedesign.png) | ![Codec tab](docs/img/demo-codec.png) |
  | ![History tab](docs/img/demo-history.png) | *Voice Clone · Preset Speakers · Voice Design · Codec · History* |

  </details>

  Demo notes: generation runs at queue concurrency 1 — the single-GPU
  unified-memory device keeps one model resident, so concurrent synthesis
  would serialize into the same compute pool anyway. Microphone input in the
  clone/codec tabs needs a secure context: use `http://localhost:8000` on the
  machine itself, or serve TLS (`bash scripts/run_demo.sh --ssl-certfile
  cert.pem --ssl-keyfile key.pem` after a one-line `openssl` self-signed
  pair). More in [`docs/troubleshooting.md`](docs/troubleshooting.md).
* **512-token generation guardrail** — bounded render time; under the
  upstream 2048 default, sampling can rarely degenerate into a minutes-long
  loop (measured: ~23 min once). Override knowingly via the Advanced
  accordion or `gen_kwargs`.
* **Reproducible RTF benchmark** (`scripts/benchmark.py`) with published
  methodology and archived raw output.
* **Docker image** with `/dev/kfd` + `/dev/dri` passthrough
  ([docker/README.md](docker/README.md)).
* **Test suite** — 353/353 on the validated ROCm host (313 CPU + 40
  real-GPU); the CPU-only CI matrix collects the same 313 CPU tests on
  Python 3.10 / 3.11 / 3.12 with 1 HIP-gated skip (and the 15 visible
  `[quality]`-extras skips noted above in a plain `.[dev]` environment;
  last size-verified green CI run: the 267-CPU era, run
  [35545854932](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/runs/35545854932),
  `266 passed, 1 skipped` — the 313-CPU era had been red from Task 14's
  undeclared jiwer import until the claims-audit fix; see
  [`evidence/claims-audit-2026-09-21.md`](evidence/claims-audit-2026-09-21.md)).
  The upstream-parity proof below is one of the 40 GPU tests — it runs on
  the validation host, not in CPU-only CI (a
  prepared-but-blocked self-hosted GPU workflow exists; see the GPU CI state
  note above and [`docs/development/gpu-ci-runbook.md`](docs/development/gpu-ci-runbook.md)).

<a id="why-this-project-exists"></a>

## Why Qwen3-TTS-ROCm?

Upstream Qwen3-TTS primarily documents CUDA / FlashAttention deployment.
This project adds a validated `gfx1151` ROCm deployment path while keeping
the official `qwen-tts` package unmodified — a thin shim of environment
diagnostics, a smart-default loader, a dual-source downloader and an enhanced
demo UI. Not a fork; no vendored or patched upstream source, ever.

| Capability | Upstream Qwen3-TTS | Qwen3-TTS-ROCm |
|---|---|---|
| Official Qwen3-TTS APIs | ✅ | ✅ (unmodified) |
| Official model weights | ✅ | ✅ (not redistributed) |
| gfx1151 ROCm path validated | — | ✅ |
| One-command ROCm wheel install | — | ✅ |
| ROCm environment self-check | — | ✅ |
| ModelScope-first downloader | — | ✅ |
| RTF benchmark evidence on AMD iGPU | — | ✅ |
| Official fine-tuning workflow (SFT) on ROCm | ✅ (CUDA + FlashAttention docs) | ✅ scoped — **execution-only smoke** (prep → 12 steps → checkpoint save → reload → sane synthesis; no quality/convergence claims). ROCm E2E execution proven. Upstream portability fix submitted as Qwen3-TTS PR #373 (OPEN) — validated by a pristine double reproduction of the failure, a minimal-fix controlled isolation, 3× targeted loader validations, two independent E2E runs from the fix branch, preserved default `flash_attention_2` semantics, and an independent chain-verifier PASS. Until it merges, current published `qwen-tts==0.1.1` still requires the documented temporary workaround: upstream's hard-coded `flash_attention_2` → `sdpa`, applied inside a gitignored clone, restored pristine. See [`docs/finetuning-rocm.md`](docs/finetuning-rocm.md) |

<a id="compatibility"></a>

## Compatibility

| GPU / Platform | Arch | ROCm | Status | Evidence |
|---|---|---|---|---|
| Radeon 8060S / Ryzen AI Max+ PRO 395 | `gfx1151` | 7.14.0 | ✅ Verified — the only independently validated configuration | [`evidence/`](evidence/README.md) |
| Other ROCm-capable AMD GPUs | — | — | 🧪 **Not yet validated — community testing wanted** | open an issue with your `qwen3-tts-rocm-check` output |

The loader's HIP defaults are generic, but every number and claim in this
repository traces to the one validated configuration above. Please don't
assume other cards work (or don't) — reports from other ROCm hardware are
very welcome and will be listed here. Note that the bundled
`scripts/install.sh` is the validated `gfx1151` path (pinned
`device-gfx1151` wheels); for other architectures, use an appropriate ROCm
PyTorch stack and report the exact install method in your validation report.

**Tested another AMD GPU? [Submit a hardware validation report](https://github.com/AIwork4me/Qwen3-TTS-ROCm/issues/new?template=hardware-validation.yml)** — measured results only, and the matrix grows.

## Performance

Measured median **RTF** — *wall-clock seconds per second of generated audio;
lower is better* — on Ryzen AI Max+ PRO 395, bfloat16/sdpa, short/medium
texts, capped at `max_new_tokens=512`. RTF 1.3 means roughly 1.3 seconds of
compute for 1 second of audio.

| Workload | Median RTF |
|---|---:|
| Custom Voice (1.7B) | 1.31 – 1.51 |
| Voice Design (1.7B) | 1.27 – 1.62 |
| Voice Clone, zero-shot (`base` 1.7B) | 1.71 – 1.88 (2026-08-27) |

A few seconds of wait for a few seconds of speech on an iGPU — usable
sentence-scale interactive demos, comfortable batch workloads. The `base`
row is the 2026-08-27 session; the 2026-09-21 replication measured
1.27 – 1.40 under non-heat-soaked conditions (the baseline's `base` cells
followed a 1396 s warmup — see the cross-day section of
[`docs/benchmarks.md`](docs/benchmarks.md)). We quote
ranges, not latency promises: numbers drift with clocks, thermals, memory
pressure and background load on shared-pool unified memory. Full per-cell
tables, methodology, n=2 caveats and the reproduce block:
[`docs/benchmarks.md`](docs/benchmarks.md). **Benchmark v2 (2026-09-24)**
adds controlled reproducibility — all five aliases (1.7B + 0.6B per
family), n=5 seeded runs per cell, cold-start/warmup/measured phases
separated, per-cell median/min/max/mean/stdev — in
[`docs/benchmarks-v2.md`](docs/benchmarks-v2.md) (RTF medians 1.04–1.52,
per-cell stdev ≤ 0.11).

<a id="roadmap-not-yet-validated"></a>

## Roadmap (not yet validated)

Nothing in this section is claimed beyond its evidence-linked status lines —
these are the rungs of the ladder, each gated on evidence before any ✅
appears anywhere for it. Upstream features are listed here when they exist
upstream but have **no Radeon evidence yet** (for vLLM-Omni, evidence so
far is offline-scope: the 2026-09-21 feasibility snapshot plus the
2026-09-24 current-stack three-family offline run linked below).

### vLLM-Omni on ROCm

The official stack ships a vLLM-Omni serving path for Qwen3-TTS; it is a
CUDA-oriented deployment. Plan, strictly in order — feasibility first, no
skipping rungs:

1. **Feasibility validation** — determine whether vLLM-Omni builds and
   imports on the pinned ROCm wheel stack at all (it may need kernels or
   wheels this stack does not carry). Output: a go/no-go with evidence.
   **Status 2026-09-21: GO** — checked per upstream's own install docs
   (`vllm==0.28.0+rocm723` wheel + `vllm-omni==0.28.0` +
   `onnxruntime-rocm`, installed in an isolated gitignored venv, validated
   `.venv` untouched). The wheel embeds compiled `gfx1151` code objects and
   the documented smallest offline example
   (`end2end.py --query-type CustomVoice`, byte-unmodified) produced a
   finite, non-silent 6.0 s / 24 kHz WAV on gfx1151 twice — see
   [`vllm-omni-feasibility-2026-09-21.txt`](evidence/vllm-omni-feasibility-2026-09-21.txt)
   / [`.json`](evidence/vllm-omni-feasibility-2026-09-21.json).
   Scope: one example, single configuration, isolated venv — nothing more
   is claimed. **Current-stack re-check 2026-09-24 (v0.2.1 Task 8): all
   three 1.7B task families green offline** on vllm 0.30.0+rocm723 +
   vllm-omni 0.30.0rc1, upstream `end2end.py` verbatim at main
   `7e5897b…` — see
   [`vllm-omni-current-gfx1151-2026-09-24.txt`](evidence/vllm-omni-current-gfx1151-2026-09-24.txt).
   Still offline-scope only; serving/streaming rungs remain unrun.
2. If feasible: **PyTorch / `qwen-tts` ROCm path re-used as the baseline**
   (this repository's proven path) as the reference point for correctness.
3. **vLLM-Omni offline inference** on gfx1151 — single-request correctness
   versus the PyTorch path first; performance later.
4. **Performance characterization** — RTF, load time, memory, under the same
   published-methodology discipline as [`docs/benchmarks.md`](docs/benchmarks.md).
5. **Online serving** — only when upstream supports the required serving
   path on a ROCm-compatible runtime; not attempted before that exists.
6. **Concurrency testing** — the validated device is a single-GPU
   unified-memory iGPU; concurrent-request behaviour must be measured, not
   assumed.
7. **Production guidance** — only after all of the above, and scoped to the
   validated configuration.

### True streaming inference

Upstream documents streaming generation with a "97 ms"-class first-audio
figure. **That number is upstream's, measured on upstream's stack — it is
NOT a Radeon number**, and it must never appear here as one.

**Finding 2026-09-21 (measured, archived):** the installed official
`qwen-tts` 0.1.1 Python API exposes **no incremental-audio path**. All
three official entry points (`generate_custom_voice`,
`generate_voice_design`, `generate_voice_clone`) are plain blocking
functions that return the complete waveform list at call completion —
none is a generator, none documents a chunk callback or streamer
parameter (the model-internal talker `generate` is invoked with a fixed
keyword set, so no streamer-style kwarg is forwarded), and the codec
decode is one full-sequence call. The wrapper's own docstrings say
`non_streaming_mode=False` "only simulates streaming text input … rather
than enabling true streaming input or streaming generation" (machine-
captured signatures and verbatim quotes with file:line references in the
archive). Measured on the validated CustomVoice 1.7B path on gfx1151 with
[`scripts/streaming_probe.py`](scripts/streaming_probe.py) — 2 runs per
scenario, both `non_streaming_mode` values, short (17 chars) and long
(205 chars) texts: **every run delivered exactly one audio chunk, at
return**. Time to first audio therefore equals total wall on every run
(3.4–4.0 s and RTF 1.22–1.25 short; 66–79 s wall for 52–59 s of audio
and RTF 1.28–1.34 long), chunk cadence is undefined (single delivery),
and the simulated real-time consumer logs 0 underruns only because 100%
of the audio already exists when playback starts. Full data:
[`streaming-2026-09-21.txt`](evidence/streaming-2026-09-21.txt) /
[`.json`](evidence/streaming-2026-09-21.json).

Streaming stays 🚫 in the capability matrix until upstream ships a
genuine incremental-delivery API. When that exists, the five measurements
below will be re-taken with the same probe on the validated gfx1151 host:

* **time to first audio** — wall time from request to the first audible
  chunk;
* **chunk cadence** — inter-chunk gap distribution (underrun-safe or not);
* **total RTF** — end-to-end real-time factor for the same text, compared
  against the non-streaming baseline;
* **buffer-underrun behaviour** — does playback ever starve on sustained
  generation;
* **long-text behaviour** — how cadence and memory hold up over long inputs.

Until those measurements exist for a real incremental path and are
archived under [`evidence/`](evidence/README.md), streaming stays 🚫 in
the capability matrix.

## Validation & Reproducibility

<a id="zero-modification-guarantee"></a>

* [`loader.load()`](src/qwen3_tts_rocm/loader.py) returns **exactly what the
  official `qwen_tts.Qwen3TTSModel.from_pretrained` returns** — the native
  model object, never a wrapper; smart defaults go through public official
  kwargs only.
* Proof by test:
  [`tests/test_official_demo_parity.py`](tests/test_official_demo_parity.py)
  builds the stock, unmodified upstream `qwen_tts.cli.demo.build_demo()`
  around a model loaded by *our* loader, then drives the demo's own handler
  closures through Gradio's event registry — including one real GPU
  synthesis. Green transcript:
  [`evidence/official-parity.txt`](evidence/official-parity.txt).
* Policy (see [`CONTRIBUTING.md`](CONTRIBUTING.md), [`NOTICE`](NOTICE)): no
  vendored or patched upstream source; `pyproject.toml` depends on the
  published `qwen-tts==0.1.1` artifact as-is.

Re-run the proof yourself:

```bash
bash scripts/verify_gpu.sh                    # SPIKE-GPU-OK on working ROCm
qwen3-tts-rocm-check                          # environment self-check
python -m pytest -m "not gpu and not requires_download" -q   # 313 CPU tests (1 HIP-gated skip without an AMD GPU; 15 more skip without .[quality])
python -m pytest -m "gpu" -q                  # 38 on-GPU tests (weights required)
.venv/bin/python scripts/benchmark.py         # fresh RTF numbers
```

Every quoted number traces to a verbatim artifact listed in
[`evidence/README.md`](evidence/README.md).

## Verified configuration

Developed and validated on exactly this machine ("tested", not "minimum
required"):

| Fact | Value |
|---|---|
| APU | AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S (`gfx1151`, Strix Halo class) |
| Memory | 94 GB LPDDR5X unified pool, ~80 GiB visible to torch/HIP |
| Kernel | Linux 6.17.0-1032-oem with `amdgpu` DRM driver (check `rocm-smi`) |
| ROCm / torch | 7.14.0-era wheels from `repo.amd.com`: `torch[device-gfx1151]==2.12.0+rocm7.14.0` (+torchvision/torchaudio) — installed automatically by `scripts/install.sh`, never typed by hand |
| Python | 3.12 on the validation host; the CPU CI matrix runs 3.10 / 3.11 / 3.12 |

### Requirements & known constraints

* **Disk** — ~5 GB for the minimal subset, ~18 GB for all six repositories
  (weights live under `models/`, never committed).
* **Network** — ModelScope (`modelscope.cn`) reachable; when
  `huggingface.co` is blocked the fallback routes via `hf-mirror.com`. The
  recorded validation host completed all downloads from a CN network without
  a VPN — other networks may vary.
* **GPU** — validated only on `gfx1151` (see
  [Compatibility](#compatibility)); a working `amdgpu` DRM driver and
  `/dev/kfd` + `/dev/dri` access are required (`render`/`video` groups).
* **Memory** — a loaded 1.7B model (bf16) used ~4.6 GiB of the unified pool
  in our session readouts; minimum total-system-memory requirements for
  smaller machines have **not** been measured. On a shared unified pool,
  close memory-hungry desktop apps for best RTF.

## Docker

Reproducible Docker build with `/dev/kfd` + `/dev/dri` passthrough for ROCm
execution; mount your `models/` directory into the image's declared
volume:

```bash
docker build -f docker/Dockerfile -t qwen3-tts-rocm:dev .
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    -v "$PWD/models:/workspace/models" \
    -p 8000:8000 \
    qwen3-tts-rocm:dev
```

Image validation transcripts: build-only era
[`evidence/docker-build-final.txt`](evidence/docker-build-final.txt); **GPU
runtime E2E (2026-09-24, v0.2.1)** — a fresh `--no-cache` image ran the full
in-container chain on the Radeon 8060S: ROCm torch/HIP 7.14 + gfx1151
diagnostics, finite bf16 matmul + SDPA, torchaudio import, repository-loader
load of the 0.6B CustomVoice checkpoint, a real official-API synthesis
(3.84 s of audio @ 24 kHz, non-silent, WAV written), plus a 4-node GPU
pytest slice — [`evidence/docker-gpu-e2e-gfx1151-2026-09-24.txt`](evidence/docker-gpu-e2e-gfx1151-2026-09-24.txt)
· [JSON](evidence/docker-gpu-e2e-gfx1151-2026-09-24.json) ·
[generated WAV](evidence/docker-gpu-e2e-gen-2026-09-24.wav). Full
guide — group-GID caveats, CPU-only diagnostics, smoke test without a GPU:
[`docker/README.md`](docker/README.md).

## Troubleshooting

Run `qwen3-tts-rocm-check` first (activate the venv in your terminal:
`source .venv/bin/activate`) — then look up your symptom in
[`docs/troubleshooting.md`](docs/troubleshooting.md), which expands every
`ERROR:` / `WARN:` / `INFO:` line the diagnostics emit:

| Symptom | Documented fix |
|---|---|
| `torch.cuda.is_available() is False` / `/dev/kfd` permissions | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `The installed PyTorch is NOT an AMD ROCm/HIP build` | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Download fails on Hugging Face *and* ModelScope | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Out-of-memory or runaway-long generations | [docs/troubleshooting.md](docs/troubleshooting.md) |
| MIOpen / SoX console chatter on first run | [docs/troubleshooting.md](docs/troubleshooting.md) |

Quick answers: **"Is this a model?"** No — models stay in the official
repositories and download separately; this project is glue around them.
**"Are weights committed to git?"** Never; see `.gitignore` and
`$QWEN3_TTS_ROCM_MODELS_DIR`.

## Contributing

Ground rules, dev setup and the PR checklist live in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Headline rule: contributions must
preserve the zero-modification guarantee — no patched upstream source, no
committed weights. Bugs and hardware reports go to the
[issue tracker](https://github.com/AIwork4me/Qwen3-TTS-ROCm/issues).

<a id="attribution--disclaimer"></a>

## Attribution & disclaimer

* Qwen3-TTS and all model weights are the work of the **Alibaba Qwen team**
  ([upstream repository](https://github.com/QwenLM/Qwen3-TTS), Apache-2.0;
  weights carry Alibaba's own Qwen model license — download terms apply, see
  [`NOTICE`](NOTICE)). Model weights are not redistributed here.
* Qwen3-TTS-ROCm is an **unofficial community adaptation**; it is not
  affiliated with, endorsed by, or produced by Alibaba or AMD.
* Condensed from the upstream demo footer: *generated audio may be inaccurate
  or inappropriate, does not represent anyone's views, and it is your
  responsibility to use it lawfully — do not create unlawful, harmful,
  deepfake or infringing content* (中文：*音频由 AI 模型自动生成，可能不准确或不当，
  不代表任何一方立场；请依法使用，严禁生成违法、有害、深度伪造或侵权内容*).

## License

Code: Apache-2.0 — see [`LICENSE`](LICENSE). Model weights remain governed by
Alibaba's own model license (provenance and download terms in
[`NOTICE`](NOTICE)).
