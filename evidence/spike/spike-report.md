# Spike Report: ROCm 7.14.0 + qwen-tts 0.1.1 on gfx1151

VERDICT: PROCEED

Date: 2026-08-27 | Branch: feat/impl-v0.1.0 | Host: Radeon 8060S (gfx1151), 94GB unified RAM, system /opt/rocm 7.2.1 coexisting with self-contained pip wheels.

## Result summary

| Probe | Outcome |
|---|---|
| `SPIKE-GPU-OK` | PASS (see `evidence/spike/gpu-probe.txt`) |
| `TOKENIZER-OK` | PASS (see `evidence/spike/tokenizer-smoke.txt`) |

## Commands run and outputs

### Step 1 — venv + AMD wheel stack (repo.amd.com)

```
uv venv --seed .venv --python 3.12
.venv/bin/python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" \
    "torchaudio==2.11.0+rocm7.14.0"
```

Exit 0 on first attempt. Installed:
`torch-2.12.0+rocm7.14.0`, `torchvision-0.27.0+rocm7.14.0`, `torchaudio-2.11.0+rocm7.14.0`,
`rocm-7.14.0`, `rocm-sdk-core-7.14.0`, `rocm-sdk-libraries-7.14.0`, `rocm-sdk-device-gfx1151-7.14.0`,
`triton-3.7.1+git0263a6a6.rocm7.14.0`, `amd-torch-device-gfx1151-2.12.0+rocm7.14.0`,
`amd-torchvision-device-gfx1151-0.27.0+rocm7.14.0`, plus numpy 2.4.4 etc.

### Step 2 — PyPI packages

```
.venv/bin/python -m pip install "qwen-tts==0.1.1" modelscope huggingface_hub pytest ruff
```

Exit 0. Notable versions: qwen-tts 0.1.1, modelscope 1.39.1 (+modelscope-hub 0.2.0),
huggingface_hub 0.36.2, transformers 4.57.3, tokenizers 0.22.2, pytest 9.1.1, ruff 0.16.4,
librosa 1.0.0, soundfile 0.14.0.

### Step 3 — GPU sanity probe

Command as in plan (`/tmp/spike_probe.py` → tee `evidence/spike/gpu-probe.txt`). Key output:

```
torch 2.12.0+rocm7.14.0 | HIP 7.14.60850
GPU: AMD Radeon 8060S Graphics | arch: gfx1151
torchaudio 2.11.0+rocm7.14.0
SPIKE-GPU-OK
```

- `torch.version.hip` = `7.14.60850` → starts with `7.14` as required.
- bf16 matmul finite; SDPA finite; torchaudio imports cleanly.
- No HSA_* env vars, no kfd permission changes, no override of system /opt/rocm needed. The pip wheels are self-contained and coexist with system ROCm 7.2.1.

### Step 4 — Tokenizer via ModelScope + encode/decode smoke

Command as in plan → tee `evidence/spike/tokenizer-smoke.txt`.

- Download path that worked: **`from modelscope import snapshot_download`** (Python API). No CLI fallback needed; no HF mirror needed. Total download ~651MB (6 files, `model.safetensors` 682MB claimed on wire at ~18.6 MB/s, finished in ~38s) into `models/Qwen3-TTS-Tokenizer-12Hz`.
- `Qwen3TTSTokenizer.from_pretrained(p, device_map="cuda:0", dtype="bfloat16")` accepted verbatim (signature is `(pretrained_model_name_or_path: str, **kwargs)`; str dtype passed through fine).
- `tok.encode("/tmp/tone.wav")` → `qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2.Qwen3TTSTokenizerV2EncoderOutput`.
- `tok.decode(out)` → tuple `(np.ndarray[24960] float32 audio, 24000)`; result has **no** `.get_audio()` method (it is a plain `(audio_array, sample_rate)` tuple) — shim code in later tasks should expect the tuple form.
- Final line: `TOKENIZER-OK`, exit 0.

## Deviations from brief

1. **None functionally required.** All bindings resolved without fallbacks: torchvision pin matched an artifact; modelscope Python import path worked; `dtype="bfloat16"` accepted as-is.
2. `.gitignore` added before commit so the brief's `git add -A` does not commit `.venv/` (~15GB) or `models/` (~651MB artifacts).
3. `scripts/verify_gpu.sh` hardened version of the probe written per task "Files" list (asserts HIP 7.14.x + gfx1151, exits non-zero on failure); verified passing before commit.

## Environment notes for later tasks

- Mem-efficient SDPA on this GPU emits `UserWarning: ... experimental. Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and falls back to another (working) backend by default. Consider setting `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` in later perf work; not required for correctness.
- flash-attn is absent (CPU-only wheel); qwen_tts warns and uses its manual PyTorch attention. Expected on ROCm; harmless here.
- System lacks the SoX binary; a warning is printed at import (`sox: not found`). Everything used in this spike works via soundfile/librosa paths. If a later task needs SoX effects, `apt install sox` then.
- MIOpen first-run kernel tuning floods stderr with solver-desc lines and `IsEnoughWorkspace` warnings during first inference; benign, one-time warmup cost (~2-3 min on first model load/inference).
- ModelScope is the proven artifact channel for weights (github.com/huggingface.co unreachable from this host). Task 5 must use `snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz", local_dir=...)`.
- Decoder output sample count for 1s @24kHz input was 24960 samples — slight length overshoot vs input; do not assume decode preserves exact sample count.

## Verdict rationale

End-to-end foundation proven with zero deviations requiring workarounds: AMD ROCm 7.14.0 pip wheels run bf16 matmul + SDPA on gfx1151, the official `qwen-tts==0.1.1` package loads the 12Hz tokenizer onto `cuda:0` in bfloat16, and encode/decode round-trip succeeds. Shim approach viable; no vendor escalation required.
