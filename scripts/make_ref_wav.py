#!/usr/bin/env python3
"""One-shot generator for src/qwen3_tts_rocm/demo/assets/ref_en.wav.

Ships a tiny, license-clean ENGLISH reference clip for voice-clone tests
(Task 13) and tokenizer roundtrip tests (Task 14): a fully SYNTHETIC,
speech-like utterance, so no third-party recording ever enters the repo.

Synthesis recipe (deterministic, seed=20260827, numpy only)
-----------------------------------------------------------
* 24000 Hz float timeline, ~2.8 s total, split into 6 syllable slots.
* Per slot i a voiced "syllable":
    1. source  -- harmonic glottal buzz at f0 sliding 135 -> 105 Hz across the
       utterance (+ slow +-2 Hz vibrato), amplitude ~1/k per harmonic k;
    2. formants -- each harmonic is weighted by three Gaussian resonances
       F1~500 Hz / F2~1500 Hz / F3~2600 Hz whose centres jitter per syllable
       (vowel-colour variation), bandwidths ~120/180/240 Hz;
    3. envelope -- raised-cosine attack/decay (~35 ms edges) so syllables are
       cleanly separated, plus a global fade-in/out on the whole clip;
    4. consonant feel -- a short bandpassed white-noise burst (~40 ms) mixed
       in front of every second syllable onset (fricative flavour).
* Mix is peak-normalised to 0.85 and written as 16-bit PCM WAV via soundfile.

NOT a plain sine and NOT real speech: the cluster of f0 harmonics under a
moving formant envelope reads as speech-like babble to feature extractors,
which is all the clone sanity gates need.  Perceptual voice quality is NOT a
goal; see tests/test_voice_clone_workflow.py docstring for the limitation note.

Usage:
    .venv/bin/python scripts/make_ref_wav.py [output.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SR = 24_000
SEED = 20260827
TOTAL_S = 2.8
N_SYLLABLES = 6

_DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "src" / "qwen3_tts_rocm" / "demo" / "assets" / "ref_en.wav"
)


def _raised_cosine_envelope(n: int, edge: int) -> np.ndarray:
    """1.0 in the middle, raised-cosine ramps of *edge* samples on both ends."""
    env = np.ones(n, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge, endpoint=False))
    env[:edge] *= ramp
    env[n - edge:] *= ramp[::-1]
    return env


def synthesize_ref_wav(sr: int = SR, seconds: float = TOTAL_S,
                       n_syllables: int = N_SYLLABLES,
                       seed: int = SEED) -> tuple[np.ndarray, int]:
    """Build the speech-like reference waveform; returns ``(wav, sr)``."""
    rng = np.random.default_rng(seed)
    total_n = round(seconds * sr)

    # Syllable layout: voiced cores separated by short silent gaps.
    gap_n = round(0.03 * sr)
    core_total = total_n - gap_n * (n_syllables + 1)
    weights = rng.uniform(0.8, 1.3, size=n_syllables)
    cores = np.maximum(
        (core_total * weights / weights.sum()).round().astype(int), round(0.12 * sr)
    )

    f0_start, f0_end = 135.0, 105.0          # gentle male-ish declination
    formant_sets = [                          # per-syllable vowel colouring
        (480 + rng.uniform(-90, 90), 1480 + rng.uniform(-250, 250),
         2580 + rng.uniform(-250, 250))
        for _ in range(n_syllables)
    ]
    bandwidths = (120.0, 180.0, 240.0)
    n_harmonics = 24

    out = np.zeros(total_n, dtype=np.float64)

    cursor = gap_n
    progress_at = lambda pos: pos / max(total_n - 1, 1)
    for i, core_len in enumerate(cores):
        p0 = progress_at(cursor)
        t_loc = np.arange(core_len, dtype=np.float64) / sr

        # Sliding vibrato-ing pitch track over this syllable slice.
        f0 = (f0_start + (f0_end - f0_start) * p0) * (
            1.0 + 0.02 * np.sin(2.0 * np.pi * 5.5 * t_loc)
        )
        phase = 2.0 * np.pi * np.cumsum(f0) / sr

        # Harmonic buzz: sum_k sin(k*phase)/k with per-syllable jitter.
        source = np.zeros(core_len, dtype=np.float64)
        ks = np.arange(1, n_harmonics + 1)
        gains = 1.0 / ks ** 0.7
        for k, g in zip(ks, gains):
            source += g * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

        # Gaussian formant resonances applied to the harmonic series.
        freqs = ks[:, None] * f0[None, :]              # (K, T) harmonic Hz
        response = np.zeros_like(freqs)
        for (fc, bw) in zip(formant_sets[i], bandwidths):
            response += np.exp(-((freqs - fc) ** 2) / (2.0 * bw**2))
        voiced = (response * source[None, :]).sum(axis=0)  # sum weighted harmonics

        # Consonant-flavoured noise burst ahead of every 2nd syllable.
        if i % 2 == 1:
            burst_len = min(round(0.04 * sr), core_len // 3)
            noise = rng.standard_normal(burst_len)
            noise = np.convolve(noise, np.ones(8) / 8.0, mode="same")  # soften highs
            voiced[:burst_len] += 0.45 * noise
            voiced[: burst_len // 2] *= np.linspace(0.0, 1.0, burst_len // 2)

        edge = min(round(0.035 * sr), core_len // 4)
        voiced *= _raised_cosine_envelope(core_len, edge)
        out[cursor : cursor + core_len] += voiced
        cursor += core_len + gap_n

    # Global fades and peak normalisation to a comfortable headroom.
    out *= _raised_cosine_envelope(total_n, round(0.02 * sr))
    peak = float(np.abs(out).max())
    if peak > 0:
        out *= 0.85 / peak
    return out.astype(np.float32), sr


def main(argv: list[str]) -> int:
    out_path = Path(argv[1]) if len(argv) > 1 else _DEFAULT_OUT
    import soundfile as sf

    wav, sr = synthesize_ref_wav()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, sr, subtype="PCM_16")

    rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float64)))))
    print(f"wrote {out_path}")
    print(f"  samples={wav.shape[0]} sr={sr} dur={len(wav)/sr:.3f}s "
          f"peak={float(np.abs(wav).max()):.4f} rms={rms:.4f} "
          f"bytes~={out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
