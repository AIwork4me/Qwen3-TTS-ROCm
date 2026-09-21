#!/usr/bin/env python
"""Quality benchmark: ASR intelligibility (CER/WER via jiwer+whisper) and
speaker similarity (resemblyzer) on the fixed manifest. Separate from the
performance benchmark; never merged into one score.

Task 14 / brief 15. What this harness measures, per fixed manifest case:

* ``zh`` cases  -> CER  (character error rate, jiwer on normalized characters)
* ``en`` cases  -> WER  (word error rate, jiwer on normalized words)
* every case    -> ASR transcript via whisper ``tiny`` (the metric's input;
                   transcripts are recorded VERBATIM — the harness measures,
                   humans interpret; a Chinese TTS transcript may be imperfect
                   and CER is exactly the number that says so)
* ``reference-clone`` case additionally -> speaker similarity between the
  reference clip and the generated audio (resemblyzer ``VoiceEncoder``
  embedding cosine, CPU)

Metric-candidate verdicts that shaped this script (full log + license probes:
``evidence/quality-benchmark-2026-09-21.txt``, Step 14.1 gates):

* jiwer 4.0.0            ACCEPT (Apache-2.0)   -> CER/WER math
* resemblyzer 0.1.4      ACCEPT (Apache-2.0; the brief's "MIT" was corrected
                          at execution — GitHub API + PyPI classifier both say
                          Apache) -> speaker embeddings (weights ship inside
                          the wheel; no runtime download)
* openai-whisper 20250625 ACCEPT (MIT)         -> ASR; ``tiny`` probed on the
                          gfx1151 GPU (probe PASS: transcribe rc 0; GPU 5.00 s
                          vs CPU 4.81 s on the 2.8 s smoke clip). The harness
                          uses cuda when torch sees it, else cpu, and records
                          the resolved device in the JSON meta.
* pystoi 0.4.1           ACCEPT (MIT) but NOT computed here: STOI needs a
                          clean-reference/degraded-signal PAIR and the fixed
                          manifest is text-conditioned (no clean reference
                          waveform exists). Validated + shipped in the
                          ``quality`` extras for signal-pair (tokenizer
                          roundtrip) evaluation; noted in the output JSON.
* pesq 0.0.4             REJECT (license): the wrapper is MIT-labelled but
                          bundles the ITU-T P.862 reference C code whose own
                          IPR notice (Psytechnics/OPTICOM owners; "cannot ...
                          put to any commercial use") is incompatible with
                          OSS distribution. NOT installed, NOT in
                          pyproject.toml; the omission is recorded in the
                          output JSON, never hidden.

Generation policy (mirrors scripts/benchmark.py so quality and performance
rows describe the same generation behavior): ``do_sample=True``,
``temperature=1.0``, ``max_new_tokens=512`` (the project latency guardrail),
language mapped zh->"Chinese" / en->"English", first supported speaker for
plain CustomVoice cases; the clone case runs on the ``base`` alias through
``generate_voice_clone`` (ICL mode: ref_audio + ref_text from the manifest).
RNG seeded per case with ``seed + case_index`` (recorded per row); GPU
sampling is not bit-deterministic, the seed is provenance, not a guarantee.

Text normalization before jiwer (fixed, unit-tested in
tests/test_quality_eval.py):

* CER: remove ALL whitespace and every Unicode punctuation character
  (category ``P*``) from both strings; characters are then compared as-is
  (case sensitive — Chinese has no case; an English-letter casing error in a
  zh transcript is a real character error and stays one).
* WER: lowercase, remove Unicode punctuation, collapse runs of whitespace;
  tokens are whitespace-separated words.

Usage::

    .venv/bin/python scripts/quality_eval.py \
        --manifest tests/data/quality_benchmark_manifest.json \
        --out evidence/quality-benchmark-<date>.json

Note for tests: importing this module is intentionally LIGHTWEIGHT (stdlib
only at import time); jiwer/whisper/resemblyzer/torch/loader are imported
lazily inside the helpers so CPU unit tests can pin cer/wer normalization and
manifest handling without paying model loads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import unicodedata
from importlib.resources import files
from math import gcd
from pathlib import Path
from time import perf_counter
from typing import Any

__all__ = [
    "ASR_MODEL_NAME",
    "LANG_FOR",
    "MAX_NEW_TOKENS",
    "asr_transcribe",
    "cer",
    "load_manifest",
    "main",
    "normalize_cer_text",
    "normalize_wer_text",
    "speaker_sim",
    "wer",
]

#: Whisper checkpoint feeding CER/WER (probed on the gfx1151 GPU, Step 14.1).
ASR_MODEL_NAME = "tiny"

#: manifest lang key -> official language name for generate_* calls.
LANG_FOR = {"zh": "Chinese", "en": "English"}

#: Project latency guardrail shared with scripts/benchmark.py (12 Hz codec;
#: 512 code steps cap any render near ~42 s of audio).
MAX_NEW_TOKENS = 512

#: Targets: TTS output rate (Qwen3-TTS tokenizer decode) and whisper input.
TTS_SR = 24000
ASR_SR = 16000

# ---------------------------------------------------------------------------
# Text normalization + CER/WER (jiwer 4.x; ACCEPTed candidate).
# ---------------------------------------------------------------------------


def _strip_punct_and_ws(text: str) -> str:
    """Remove whitespace and Unicode punctuation (categories Z*/P*)."""
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith(("P", "Z", "C"))
    )


def normalize_cer_text(text: str) -> str:
    """CER normalization: drop whitespace + punctuation, keep characters."""
    return _strip_punct_and_ws(text)


def normalize_wer_text(text: str) -> str:
    """WER normalization: lowercase, drop punctuation, collapse whitespace."""
    lowered = "".join(
        ch if not unicodedata.category(ch).startswith("P") else " "
        for ch in text.lower()
    )
    return " ".join(lowered.split())


def cer(ref: str, hyp: str) -> float:
    """Character error rate on normalized text (zh cases).

    Empty normalized reference would make the rate undefined; that is a
    manifest bug and raises instead of hiding behind a fake number.
    """
    import jiwer

    ref_n, hyp_n = normalize_cer_text(ref), normalize_cer_text(hyp)
    if not ref_n:
        raise ValueError(f"CER reference normalizes to empty: {ref!r}")
    return float(jiwer.cer(ref_n, hyp_n))


def wer(ref: str, hyp: str) -> float:
    """Word error rate on normalized text (en cases)."""
    import jiwer

    ref_n, hyp_n = normalize_wer_text(ref), normalize_wer_text(hyp)
    if not ref_n:
        raise ValueError(f"WER reference normalizes to empty: {ref!r}")
    return float(jiwer.wer(ref_n, hyp_n))


# ---------------------------------------------------------------------------
# Audio plumbing (soundfile + scipy already in the validated stack — no new
# audio I/O dependencies were added for this harness).
# ---------------------------------------------------------------------------


def _read_audio(path: Path | str) -> tuple[Any, int]:
    """(float32 mono waveform, sr) from disk via soundfile."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    import numpy as np

    arr = np.asarray(wav)
    if arr.ndim > 1:  # stereo file: fold to mono
        arr = arr.mean(axis=1)
    return arr.astype(np.float32), int(sr)


def _resample_to_16k(wav: Any, sr: int) -> Any:
    """Deterministic polyphase resample to whisper's 16 kHz (scipy)."""
    import numpy as np
    from scipy.signal import resample_poly

    arr = np.asarray(wav, dtype=np.float32)
    if sr == ASR_SR:
        return arr
    if sr <= 0:
        raise ValueError(f"invalid sample rate {sr}")
    g = gcd(ASR_SR, int(sr))
    return resample_poly(arr, ASR_SR // g, int(sr) // g).astype(np.float32)


def _as_wav_sr(wav: Path | Any, sr: int | None) -> tuple[Any, int]:
    """Accept a file path or an in-memory waveform; normalize to (wav, sr)."""
    if isinstance(wav, (str, Path)):
        return _read_audio(Path(wav))
    if sr is None:
        raise ValueError("in-memory waveform needs its sample rate via sr=")
    return wav, int(sr)


# ---------------------------------------------------------------------------
# Speaker similarity (resemblyzer 0.1.4; ACCEPTed; CPU; weights in-wheel).
# ---------------------------------------------------------------------------

_ENCODER: dict[str, Any] = {}


def _get_encoder() -> Any:
    """Cached resemblyzer VoiceEncoder pinned to CPU (tiny LSTM; GPU adds
    nothing here and keeps the metric independent of the TTS device)."""
    if "enc" not in _ENCODER:
        from resemblyzer import VoiceEncoder

        _ENCODER["enc"] = VoiceEncoder(device="cpu", verbose=False)
    return _ENCODER["enc"]


def speaker_sim(a: Path | Any, b: Path | Any,
                *, sr: int | None = None) -> float:
    """Cosine similarity of resemblyzer embeddings of two utterances.

    ``a``/``b`` are wav paths or in-memory float waveforms (then ``sr`` is
    required). Audio is resampled to 16 kHz; the returned value is recorded
    verbatim — the harness measures, humans interpret.
    """
    import numpy as np

    enc = _get_encoder()
    embeds = []
    for item in (a, b):
        wav, item_sr = _as_wav_sr(item, sr)
        wav16 = _resample_to_16k(wav, item_sr)
        embeds.append(enc.embed_utterance(wav16))
    x, y = embeds
    return float(
        np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    )


# ---------------------------------------------------------------------------
# ASR (openai-whisper 20250625; ACCEPTed; tiny; device resolved once).
# ---------------------------------------------------------------------------

_WHISPER: dict[str, Any] = {}


def _get_asr_model() -> Any:
    """Cached whisper ``tiny`` on cuda when torch sees a device, else cpu."""
    if "model" not in _WHISPER:
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _WHISPER["device"] = device
        _WHISPER["model"] = whisper.load_model(ASR_MODEL_NAME, device=device)
    return _WHISPER["model"]


def asr_transcribe(wav: Path | Any, lang: str,
                   *, sr: int | None = None) -> str:
    """Transcribe one utterance with whisper tiny (language pinned, greedy).

    ``temperature=0`` disables the decode-time temperature fallback ladder so
    the metric input is deterministic for a fixed model + audio.
    """
    model = _get_asr_model()
    audio, audio_sr = _as_wav_sr(wav, sr)
    audio16 = _resample_to_16k(audio, audio_sr)
    result = model.transcribe(audio16, language=lang, temperature=0)
    return str(result["text"])


# ---------------------------------------------------------------------------
# Manifest (Step 14.2: tests/data/quality_benchmark_manifest.json).
# ---------------------------------------------------------------------------


def load_manifest(path: Path | str) -> dict:
    """Load + validate the fixed manifest (schema pinned by unit tests).

    TypeError for structural type violations, ValueError for semantic ones
    (duplicate ids, unknown lang/role, clone case without its ref fields).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("seed"), int):
        raise TypeError(f"{path}: manifest 'seed' must be an int")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: manifest needs a non-empty 'cases' list")
    seen: set[str] = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"{path}: case #{i} is not an object")
        for key in ("id", "text", "lang"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"{path}: case #{i} needs a non-empty {key!r} string")
        if case["id"] in seen:
            raise ValueError(f"{path}: duplicate case id {case['id']!r}")
        seen.add(case["id"])
        if case["lang"] not in LANG_FOR:
            raise ValueError(
                f"{path}: case {case['id']!r} lang {case['lang']!r} not in {sorted(LANG_FOR)}"
            )
        role = case.get("role")
        if role is not None and role != "reference-clone":
            raise ValueError(f"{path}: case {case['id']!r} unknown role {role!r}")
        if role == "reference-clone":
            for key in ("ref", "ref_text"):
                if not isinstance(case.get(key), str) or not case[key]:
                    raise ValueError(
                        f"{path}: reference-clone case {case['id']!r} needs {key!r}"
                    )
    return data


def _clone_ref_path(case: dict) -> Path:
    """Resolve the manifest's clone ``ref``; falls back to the bundled asset
    (same file) when the repo is installed as a package and the relative
    path does not exist from cwd."""
    ref = Path(case["ref"])
    if ref.exists():
        return ref
    bundled = Path(str(files("qwen3_tts_rocm") / "demo" / "assets" / "ref_en.wav"))
    if bundled.exists() and bundled.name == ref.name:
        return bundled
    raise FileNotFoundError(
        f"clone reference {ref} not found (cwd={Path.cwd()}); expected the "
        f"bundled asset {bundled}"
    )


# ---------------------------------------------------------------------------
# Environment metadata (guarded provenance probes; never block the run).
# ---------------------------------------------------------------------------


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=False, cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001,S110 - provenance probes never block
        pass
    return "unknown"


def collect_meta(args: argparse.Namespace) -> dict:
    """JSON 'meta' block: env facts + exact args + policy pins."""
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import version as pkg_version

    def _probe(fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return f"probe failed: {exc}"

    import torch

    gpu = "unavailable"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = (
            f"{props.name} ({getattr(props, 'gcnArchName', '?')}, "
            f"torch-visible memory {props.total_memory / 1024**3:.1f} GiB)"
        )
    return {
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": f"{torch.__version__} (HIP {getattr(torch.version, 'hip', None)})",
        "gpu": gpu,
        "versions": {
            "jiwer": _probe(lambda: pkg_version("jiwer")),
            "resemblyzer": _probe(lambda: pkg_version("resemblyzer")),
            "openai-whisper": _probe(lambda: pkg_version("openai-whisper")),
            "pystoi": _probe(lambda: pkg_version("pystoi")),
            "pesq": "not installed (REJECTED at Step 14.1 license gate)",
        },
        "args": {"manifest": str(args.manifest), "out": str(args.out)},
        "sampling_policy": {
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "speaker": "first supported speaker (generate_custom_voice)",
            "rng_seed_per_case": "manifest seed + case index (torch.manual_seed)",
        },
        "asr": {"model": ASR_MODEL_NAME, "temperature": 0,
                "device": "cuda if torch.cuda.is_available() else cpu"},
    }


# ---------------------------------------------------------------------------
# Omitted-metric notes (rejections + scope, recorded not hidden).
# ---------------------------------------------------------------------------

OMITTED_METRICS: dict[str, str] = {
    "pesq": (
        "REJECTED at the Step 14.1 license gate: the pesq PyPI wrapper is "
        "MIT-labelled but bundles the ITU-T P.862 reference C implementation "
        "whose own IPR notice (owners British Telecommunications/Psytechnics "
        "and Royal KPN/OPTICOM; 'cannot ... sell, hire, loan, distribute, "
        "dispose or put to any commercial use') is incompatible with OSS "
        "distribution. Not installed; see "
        "evidence/quality-benchmark-2026-09-21.txt."
    ),
    "stoi": (
        "pystoi ACCEPTed at the Step 14.1 gates (MIT; synthetic-pair smoke "
        "pass) but not computed for these cases: STOI requires a "
        "clean-reference/degraded-signal pair and the fixed manifest is "
        "text-conditioned (no clean reference waveform exists). Shipped in "
        "the quality extras for signal-pair (tokenizer roundtrip) evaluation."
    ),
}


# ---------------------------------------------------------------------------
# Main driver: generate -> sanity gate -> transcribe -> metric per case.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="quality_eval",
        description="TTS quality benchmark (CER/WER + speaker similarity) "
                    "on the fixed manifest (Task 14 / brief 15)",
    )
    ap.add_argument(
        "--manifest", default="tests/data/quality_benchmark_manifest.json",
        help="fixed benchmark manifest (default: %(default)s)",
    )
    ap.add_argument("--out", required=True, help="output JSON path")
    return ap.parse_args(argv)


def _generate_plain(model: Any, case: dict) -> tuple[Any, int]:
    """One CustomVoice render for a plain (non-clone) case."""
    return model.generate_custom_voice(
        text=case["text"], language=LANG_FOR[case["lang"]],
        speaker=model.get_supported_speakers()[0],
        do_sample=True, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS,
    )


def _generate_clone(model: Any, case: dict) -> tuple[Any, int]:
    """One ICL voice-clone render for the reference-clone case."""
    ref_wav, ref_sr = _read_audio(_clone_ref_path(case))
    return model.generate_voice_clone(
        text=case["text"], language=LANG_FOR[case["lang"]],
        ref_audio=(ref_wav, ref_sr), ref_text=case["ref_text"],
        do_sample=True, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    import torch

    from qwen3_tts_rocm import loader, testing

    meta = collect_meta(args)
    print("== qwen3-tts-rocm quality benchmark (Task 14 / brief 15) ==")
    print(f"# date={meta['date']}")
    print(f"# host={meta['host']} gpu={meta['gpu']}")
    print(f"# torch={meta['torch']}")
    print(f"# manifest={manifest_path} seed={manifest['seed']} "
          f"cases={[c['id'] for c in manifest['cases']]}")

    # One model per alias for the whole run (custom-voice for plain cases,
    # base for the clone case), unloaded once at the end — GPU serialization
    # friendly and identical semantics to per-case loads.
    loaded: dict[str, Any] = {}
    rows: list[dict] = []
    try:
        for index, case in enumerate(manifest["cases"]):
            alias = ("base" if case.get("role") == "reference-clone"
                     else "custom-voice")
            if alias not in loaded:
                t0 = perf_counter()
                loaded[alias] = loader.load(alias)
                print(f"[quality] alias={alias} loaded in "
                      f"{perf_counter() - t0:.1f}s", flush=True)
            model = loaded[alias]

            torch.manual_seed(int(manifest["seed"]) + index)
            t1 = perf_counter()
            if case.get("role") == "reference-clone":
                wavs, sr = _generate_clone(model, case)
            else:
                wavs, sr = _generate_plain(model, case)
            gen_s = perf_counter() - t1
            wav = wavs[0]
            testing.assert_wav_sane(wav, sr_expected=sr)
            audio_s = len(wav) / int(sr)

            transcript = asr_transcribe(wav, case["lang"], sr=int(sr))
            if case["lang"] == "zh":
                metric_name, metric_value = "cer", cer(case["text"], transcript)
            else:
                metric_name, metric_value = "wer", wer(case["text"], transcript)

            row: dict[str, Any] = {
                "id": case["id"],
                "lang": case["lang"],
                "metric": metric_name,
                metric_name: round(metric_value, 4),
                "text": case["text"],
                "transcript": transcript,
                "audio_seconds": round(audio_s, 3),
            }
            if case.get("role") == "reference-clone":
                row["speaker_sim_ref_vs_output"] = round(
                    speaker_sim(_clone_ref_path(case), wav, sr=int(sr)), 4
                )
            rows.append(row)
            print(
                f"[quality] case={case['id']} gen={gen_s:.1f}s "
                f"audio={audio_s:.2f}s {metric_name}={metric_value:.4f} "
                f"transcript={transcript!r}", flush=True,
            )
    finally:
        for model in loaded.values():
            loader.unload(model)

    out = {
        "meta": meta,
        "omitted_metrics": OMITTED_METRICS,
        "seed": manifest["seed"],
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[quality] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
