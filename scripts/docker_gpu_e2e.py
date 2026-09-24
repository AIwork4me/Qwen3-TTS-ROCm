#!/usr/bin/env python3
"""In-container GPU E2E probe for the qwen3-tts-rocm Docker image (v0.2.1 Task 2).

Runs INSIDE the built image (never on the host — the host has its own
validated stack and must not be conflated with container evidence)::

    docker run --rm \
        --device /dev/kfd --device /dev/dri \
        --group-add <video_gid> --group-add <render_gid> \
        -v "$PWD/models:/workspace/models" \
        -v "$PWD/.work-docker:/workspace/out" \
        --entrypoint bash qwen3-tts-rocm:gfx1151-e2e -c \
        '/workspace/.venv/bin/python /workspace/scripts/docker_gpu_e2e.py \
         --json-out /workspace/out/docker-gpu-e2e.json \
         --wav /workspace/out/gen.wav'

Proves the GPU runtime path, not just image construction: ROCm torch + HIP
version + GPU name + gfx arch from INSIDE the container, a finite bf16
matmul, a finite SDPA call, a torchaudio import, a real
``qwen3_tts_rocm.loader.load("custom-voice-0.6b")`` against the mounted
models tree, one official-API ``generate_custom_voice`` synthesis, waveform
assertions (expected sample rate, finite, non-silent, sane duration), a WAV
write, and peak-memory readouts. Emits one JSON summary (``--json-out``)
and exits nonzero on the first failed expectation — no softened checks.

The synthesis call convention mirrors the GPU suites: keyword-first
official API, ``max_new_tokens=512`` latency guardrail, sanity thresholds
identical to ``qwen3_tts_rocm.testing.assert_wav_sane`` (min RMS 1e-3).

This script is intentionally part of the image build context: committing it
before ``docker build`` embeds the exact probe the evidence run executed,
so a verifier re-running the image reruns byte-identical checks.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="In-container GPU E2E probe (docker image runtime validation)")
    ap.add_argument("--alias", default="custom-voice-0.6b",
                    help="registry alias to load through the repository loader "
                         "(default: custom-voice-0.6b, the Task 2 minimum)")
    ap.add_argument("--expect-gpu", default="Radeon 8060S",
                    help="substring the GPU name must contain (default: Radeon 8060S)")
    ap.add_argument("--expect-arch", default="gfx1151",
                    help="required gcnArchName (default: gfx1151)")
    ap.add_argument("--expect-hip", default="7.14",
                    help="required HIP version prefix (default: 7.14)")
    ap.add_argument("--expect-sr", type=int, default=24000,
                    help="expected sample rate of the 12Hz codec output (default: 24000)")
    ap.add_argument("--wav", default="/workspace/out/gen.wav",
                    help="where to write the generated WAV")
    ap.add_argument("--json-out", default="/workspace/out/docker-gpu-e2e.json",
                    help="machine-readable summary JSON path")
    ap.add_argument("--text", default="今天的天气真不错，适合去公园散步。",
                    help="synthesis text (default: the zh sentence pinned across the GPU suites)")
    return ap.parse_args(argv)


class _Fail(Exception):
    pass


#: Accumulated check records; filled only by _check, dumped into the JSON.
_checks: list[dict] = []


def _check(cond: bool, label: str, detail: str = "") -> None:
    rec = {"check": label, "ok": bool(cond), "detail": detail}
    _checks.append(rec)
    print(f"[e2e] {'OK  ' if cond else 'FAIL'} {label}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        raise _Fail(label)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = _checks
    gen: dict = {}
    t_start = time.perf_counter()

    # --- torch / ROCm environment inside the container -----------------------
    import torch

    _check("+rocm" in torch.__version__, "torch_is_rocm_build", torch.__version__)
    hip = str(torch.version.hip)
    _check(hip.startswith(args.expect_hip), "hip_version", hip)
    _check(torch.cuda.is_available(), "cuda_available")
    p = torch.cuda.get_device_properties(0)
    _check(args.expect_gpu in p.name, "gpu_name", p.name)
    arch = str(getattr(p, "gcnArchName", ""))
    _check(arch == args.expect_arch, "gfx_arch", arch)

    # --- finite bf16 matmul ---------------------------------------------------
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    _check(bool(torch.isfinite(c).all()), "bf16_matmul_finite",
           f"shape={tuple(c.shape)} min={float(c.min()):.3f} max={float(c.max()):.3f}")

    # --- finite SDPA ----------------------------------------------------------
    q = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    sd = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    _check(bool(torch.isfinite(sd).all()), "sdpa_finite",
           f"shape={tuple(sd.shape)}")

    # --- torchaudio -----------------------------------------------------------
    import torchaudio
    checks.append({"check": "torchaudio_import", "ok": True, "detail": torchaudio.__version__})
    print(f"[e2e] OK   torchaudio_import — {torchaudio.__version__}", flush=True)

    # --- repository loader + real synthesis (official API only) ---------------
    from qwen3_tts_rocm import loader

    t0 = time.perf_counter()
    model = loader.load(args.alias)
    gen["load_seconds"] = round(time.perf_counter() - t0, 2)
    _check(model is not None, "loader_load", f"alias={args.alias} load_seconds={gen['load_seconds']}")
    spks = model.get_supported_speakers()
    _check(len(spks) >= 9, "speakers_surface", f"n={len(spks)} first={spks[0]}")

    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=args.text, language="Auto", speaker=spks[0], max_new_tokens=512)
    torch.cuda.synchronize()
    gen["generate_seconds"] = round(time.perf_counter() - t0, 2)
    _check(len(wavs) == 1, "one_waveform", f"n={len(wavs)}")

    wav = wavs[0]
    # The official API returns numpy float waveforms; normalize to torch for
    # uniform assertions (found live in the first E2E attempt, 2026-09-24:
    # assuming torch tensors crashed the probe AFTER a successful render).
    import numpy as np

    if isinstance(wav, np.ndarray):
        wav_t = torch.from_numpy(wav.astype("float32"))
    else:
        wav_t = wav.detach().cpu().float()
    dur = float(wav_t.shape[-1]) / sr
    rms = float(wav_t.pow(2).mean().sqrt())
    finite = bool(torch.isfinite(wav_t).all())
    _check(sr == args.expect_sr, "sample_rate", f"{sr} (expected {args.expect_sr})")
    _check(finite, "waveform_finite")
    _check(rms >= 1e-3, "non_silent", f"rms={rms:.4f} (>= 1e-3)")
    _check(0.5 <= dur <= 60.0, "duration_sane", f"{dur:.2f}s in [0.5, 60]")
    gen.update({"audio_seconds": round(dur, 2),
                # Repo-binding definition (scripts/benchmark.py): RTF = wall / audio, lower is better.
                "rtf": round(gen["generate_seconds"] / dur, 2)
                if gen.get("generate_seconds") else None,
                "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)})

    # --- WAV write ------------------------------------------------------------
    import os
    os.makedirs(os.path.dirname(args.wav) or ".", exist_ok=True)
    try:
        import soundfile as sf
        sf.write(args.wav, wav_t.numpy(), sr)
        writer = "soundfile"
    except ImportError:
        torchaudio.save(args.wav, wav_t.unsqueeze(0), sr)
        writer = "torchaudio"
    size = os.path.getsize(args.wav)
    _check(size > 0, "wav_written", f"{args.wav} {size} bytes via {writer}")
    gen["wav_bytes"] = size
    gen["wav_writer"] = writer

    loader.unload(model)

    doc = {
        "meta": {
            "probe": "scripts/docker_gpu_e2e.py",
            "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": hip,
            "gpu": p.name,
            "gcn_arch": arch,
            "alias": args.alias,
            "wall_seconds": round(time.perf_counter() - t_start, 2),
        },
        "checks": checks,
        "generation": gen,
    }
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[e2e] wrote {args.json_out}", flush=True)
    print("DOCKER-GPU-E2E-OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except _Fail as e:
        print(f"DOCKER-GPU-E2E-FAILED: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
