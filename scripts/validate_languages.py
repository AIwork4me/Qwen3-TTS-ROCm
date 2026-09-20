#!/usr/bin/env python3
"""Multilingual language-validation harness for qwen3-tts-rocm (Task 2).

Runs the full officially supported language surface of Qwen3-TTS through
real end-to-end generation on the Radeon GPU and archives a machine-readable
capability matrix.  Layout (24 matrix cells + 4 auxiliary reference cells):

* ``custom-voice`` (1.7B) -- ``generate_custom_voice`` once per manifest
  language (10 cells), plus one reference generation per DISTINCT clone-pair
  reference language (4 cells, recorded as the ``clone_reference_audio``
  section) -- the reference audio must be synthesized while the CustomVoice
  model is resident because the Base model cannot generate from scratch.
* ``voice-design`` (1.7B) -- ``generate_voice_design`` once per language with
  that language's natural-language style description (10 cells).
* ``base`` (1.7B) -- per cross-lingual clone pair (4 representative pairs):
  ``create_voice_clone_prompt`` from the reference audio, then
  ``generate_voice_clone`` into the target language.

Model residence plan: one model resident at a time (custom-voice ->
voice-design -> base, ``loader.unload`` between phases); the synthesized
reference waveforms are plain numpy arrays and survive the unload.

Claim discipline (binding)
--------------------------
The ONLY capability claim this harness produces is "end-to-end generation
completed on Radeon", established purely from waveform sanity: finite values,
non-silent (rms > 1e-3), valid sample rate, bounded duration.  It makes NO
pronunciation-quality claim anywhere, and cross-lingual clone coverage is
explicitly representative (4 pairs), not exhaustive.

Cross-lingual clone transcript rule (binding invariant): the ``ref_text``
handed to ``create_voice_clone_prompt`` is EXACTLY the manifest sentence that
was synthesized into the reference audio by ``generate_custom_voice``.  Both
sides are taken from the same :class:`Sample` object, so the identity is
structural, and the exact string is recorded in every clone row (``ref_text``)
for audit.

Language handling
-----------------
Manifest language names are canonical keys ("Chinese", ...); each resolves
case-insensitively onto the runtime identifiers of the installed package's
``get_supported_languages()`` (lowercase, incl. "auto"; the verbatim list
lives in evidence/ground-truth-2026-09-20.md section 5a).  The resolved
runtime identifier is recorded in every row.  Drift between manifest and
installed package is a HARD error -- in either direction (a manifest language
missing from the package, or a package language the manifest does not cover)
-- never a silent skip; the run aborts with exit code 2.

Call convention: official APIs only, keyword-first, unmodified; every
generation passes ``max_new_tokens=512`` (the project latency guardrail: at
12 Hz this caps a render near ~42 s of audio while bounding degenerate-loop
wall time).  All other sampling knobs stay at the official
``generate_config.json`` defaults.  A cell that fails records its error row
and the run continues; the exit code reflects overall pass/fail (0 = every
cell passed, 1 = at least one failure, 2 = hard harness/config error).

Usage::

    .venv/bin/python scripts/validate_languages.py 2>&1 | tee evidence/multilingual-matrix.txt

Outputs a JSON document (default ``evidence/multilingual-matrix.json``) with
``meta`` (timestamp, GPU name, gfx, ROCm/torch/qwen-tts versions, git HEAD,
upstream SHA, model alias per section, the runtime language list, the claim
text), ``results`` (one row per cell: language, resolved language_id, wall_s,
audio_s, checks, pass, error) and ``summary`` (counts + per-language matrix
the README table renders from).  The stdout transcript teed to
``.txt`` is the verbatim run record.

Note for tests: importing this module is intentionally LIGHTWEIGHT (stdlib
only at import time); torch/qwen_tts/numpy/loader are imported lazily inside
functions so unit tests can pin the pure helpers anywhere (mirrors
scripts/benchmark.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # editors only; never executed (lazy-import discipline)
    import numpy as np

__all__ = [
    "MAX_DURATION_S",
    "MAX_NEW_TOKENS",
    "UPSTREAM_QWEN3_TTS_SHA",
    "ClonePair",
    "Manifest",
    "Sample",
    "check_coverage",
    "evaluate_waveform",
    "load_manifest",
    "main",
    "resolve_language",
    "summarize_results",
]

#: Upstream QwenLM/Qwen3-TTS HEAD SHA at the Task 0 ground-truth audit
#: (2026-09-20; evidence/ground-truth-2026-09-20.md).  Recorded verbatim into
#: the JSON meta so every archived matrix names the upstream it measured
#: against without re-deriving it at run time.
UPSTREAM_QWEN3_TTS_SHA = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

#: The project latency guardrail applied to EVERY generation in this harness.
MAX_NEW_TOKENS = 512

#: Default evidence JSON (cwd-relative, next to the teed .txt transcript).
DEFAULT_JSON_OUT = "evidence/multilingual-matrix.json"

#: Shipped manifest fixture (repo-relative so the script runs from anywhere).
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "tests" / "data" / (
    "multilingual_samples.json")

#: Silence gate: RMS below this means the render is effectively silent.
SILENCE_RMS_THRESHOLD = 1e-3

#: Duration ceiling for a sane render: max_new_tokens=512 code steps at 12 Hz
#: cap audio near 512/12 = 42.7 s; 60 s adds ~40% margin without accepting
#: runaway output.
MAX_DURATION_S = 60.0

#: Canonical manifest language -> short display code (README "ref: en, fr").
LANG_CODES: dict[str, str] = {
    "Chinese": "zh", "English": "en", "Japanese": "ja", "Korean": "ko",
    "German": "de", "French": "fr", "Russian": "ru", "Portuguese": "pt",
    "Spanish": "es", "Italian": "it",
}

#: Result-section keys: the three matrix sections plus the auxiliary
#: reference-audio generations (which must also pass for a green run).
MATRIX_SECTIONS = ("custom_voice", "voice_design", "cross_lingual_clone")
REFERENCE_SECTION = "clone_reference_audio"

#: The one and only capability claim this harness is allowed to make.
CLAIM = (
    "end-to-end generation completed on Radeon (waveform sanity: finite, "
    "non-silent, valid sample rate, bounded duration); NOT a "
    "pronunciation-quality claim; cross-lingual clone coverage is "
    "representative (4 pairs), not exhaustive"
)

#: Caption printed under the summary table (same wording the README uses).
CAPTION = (
    "✅ = end-to-end generation completed on Radeon (waveform sanity: finite, "
    "non-silent, valid sample rate, bounded duration) — see "
    "`evidence/multilingual-matrix.json`. This is NOT a pronunciation-quality "
    "claim. Cross-lingual clone coverage is representative (4 pairs), not "
    "exhaustive."
)


# ---------------------------------------------------------------------------
# Manifest (pure; unit-tested in tests/test_validate_languages.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One per-language payload: canonical language name + authored sentence."""

    language: str
    text: str


@dataclass(frozen=True)
class ClonePair:
    """One representative cross-lingual clone pair (ref != target)."""

    ref_language: str
    target_language: str


@dataclass(frozen=True)
class Manifest:
    """Parsed + schema-validated multilingual sample manifest.

    Iterating yields the :class:`Sample` entries (so ``list(manifest)`` is the
    ``list[Sample]`` the interface contract names); ``voice_design_descriptions``
    maps every sample language to its natural-language style description and
    ``cross_lingual_clone_pairs`` holds the representative clone pairs.
    """

    samples: tuple[Sample, ...]
    voice_design_descriptions: dict[str, str]
    cross_lingual_clone_pairs: tuple[ClonePair, ...]

    @property
    def languages(self) -> tuple[str, ...]:
        """Canonical language names in manifest order."""
        return tuple(s.language for s in self.samples)

    def sample_for(self, language: str) -> Sample:
        """The sample for *language*; KeyError (naming the manifest) on miss."""
        for sample in self.samples:
            if sample.language == language:
                return sample
        raise KeyError(
            f"no sample for language {language!r} in manifest "
            f"{list(self.languages)}"
        )

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def load_manifest(path: str | Path) -> Manifest:
    """Parse + schema-validate the multilingual sample manifest (pure).

    Validates: required top-level keys; non-empty ``samples`` list of objects
    with non-empty string ``language``/``text``; no duplicate languages;
    ``voice_design_descriptions`` keys exactly covering the sample languages
    with non-empty string values; ``cross_lingual_clone_pairs`` entries with
    exactly-known ``ref_language``/``target_language`` (both sample languages,
    and different from each other -- cross-lingual means cross-lingual).

    Raises ValueError naming the offending index/key on any violation.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read multilingual sample manifest at {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise TypeError(
            f"manifest {p}: top level must be a JSON object, "
            f"got {type(data).__name__}"
        )
    for key in ("samples", "voice_design_descriptions", "cross_lingual_clone_pairs"):
        if key not in data:
            raise ValueError(f"manifest {p}: missing required top-level key {key!r}")

    raw_samples = data["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise TypeError(
            f"manifest {p}: 'samples' must be a non-empty list, "
            f"got {type(raw_samples).__name__}"
        )
    samples: list[Sample] = []
    for i, entry in enumerate(raw_samples):
        if not isinstance(entry, dict):
            raise TypeError(
                f"manifest {p}: sample {i} must be an object, "
                f"got {type(entry).__name__}"
            )
        missing = [k for k in ("language", "text") if k not in entry]
        if missing:
            raise ValueError(
                f"manifest {p}: sample {i} is missing required key(s) {missing}"
            )
        language, text = entry["language"], entry["text"]
        if not isinstance(language, str) or not language.strip():
            raise ValueError(
                f"manifest {p}: sample {i} 'language' must be a non-empty string"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"manifest {p}: sample {i} 'text' must be a non-empty string"
            )
        samples.append(Sample(language=language.strip(), text=text.strip()))

    languages = [s.language for s in samples]
    duplicates = sorted({x for x in languages if languages.count(x) > 1})
    if duplicates:
        raise ValueError(
            f"manifest {p}: duplicate language entries {duplicates}; "
            "exactly one sample per language"
        )

    descriptions = data["voice_design_descriptions"]
    if not isinstance(descriptions, dict):
        raise TypeError(
            f"manifest {p}: 'voice_design_descriptions' must be an object, "
            f"got {type(descriptions).__name__}"
        )
    for key, value in descriptions.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"manifest {p}: voice_design_descriptions[{key!r}] must be "
                "a non-empty string"
            )
    missing_desc = [x for x in languages if x not in descriptions]
    unknown_desc = [k for k in descriptions if k not in languages]
    if missing_desc or unknown_desc:
        raise ValueError(
            f"manifest {p}: voice_design_descriptions keys must exactly cover "
            f"the sample languages; missing={missing_desc} unknown={unknown_desc}"
        )

    raw_pairs = data["cross_lingual_clone_pairs"]
    if not isinstance(raw_pairs, list):
        raise TypeError(
            f"manifest {p}: 'cross_lingual_clone_pairs' must be a list, "
            f"got {type(raw_pairs).__name__}"
        )
    known = set(languages)
    pairs: list[ClonePair] = []
    for i, entry in enumerate(raw_pairs):
        if not isinstance(entry, dict):
            raise TypeError(
                f"manifest {p}: cross_lingual_clone_pairs[{i}] must be an object"
            )
        missing = [k for k in ("ref_language", "target_language") if k not in entry]
        if missing:
            raise ValueError(
                f"manifest {p}: cross_lingual_clone_pairs[{i}] is missing "
                f"required key(s) {missing}"
            )
        ref, target = entry["ref_language"], entry["target_language"]
        for role, value in (("ref_language", ref), ("target_language", target)):
            if value not in known:
                raise ValueError(
                    f"manifest {p}: cross_lingual_clone_pairs[{i}] "
                    f"{role}={value!r} is not one of the sample languages "
                    f"{languages}"
                )
        if ref == target:
            raise ValueError(
                f"manifest {p}: cross_lingual_clone_pairs[{i}] is not "
                f"cross-lingual (ref_language == target_language == {ref!r})"
            )
        pairs.append(ClonePair(ref_language=ref, target_language=target))

    return Manifest(
        samples=tuple(samples),
        voice_design_descriptions=dict(descriptions),
        cross_lingual_clone_pairs=tuple(pairs),
    )


# ---------------------------------------------------------------------------
# Language resolution + waveform sanity + aggregation (pure).
# ---------------------------------------------------------------------------


def resolve_language(canonical: str, supported: list[str]) -> str:
    """Case-insensitive canonical -> runtime identifier match (pure).

    Returns the runtime identifier from *supported* verbatim (so the recorded
    ``language_id`` is exactly what the installed package reports).  Raises
    KeyError naming BOTH sides on a miss: manifest/package drift must be a
    loud error, never a silent skip.
    """
    wanted = str(canonical).strip().lower()
    for identifier in supported:
        if str(identifier).strip().lower() == wanted:
            return str(identifier)
    raise KeyError(
        f"manifest language {canonical!r} has no case-insensitive match in the "
        f"installed model's get_supported_languages() list "
        f"{sorted(str(x) for x in supported)}; this is manifest/package drift "
        "and a hard error -- extend tests/data/multilingual_samples.json or "
        "align the installed qwen-tts version"
    )


def check_coverage(manifest_languages: list[str], supported: list[str]) -> list[str]:
    """Runtime languages (minus 'auto') missing from the manifest (pure).

    Empty list = the manifest covers every officially supported language.
    Non-empty = drift in the other direction (the installed package grew a
    language the manifest does not exercise) and main() hard-errors on it.
    """
    known = {str(x).strip().lower() for x in manifest_languages}
    return [
        str(x) for x in supported
        if str(x).strip().lower() != "auto" and str(x).strip().lower() not in known
    ]


def evaluate_waveform(wav: np.ndarray, sr: int) -> dict:
    """Waveform sanity gate -- the ONLY capability check (pure).

    Checks: finite (non-empty, all values finite), non-silent (RMS >
    :data:`SILENCE_RMS_THRESHOLD`), valid sample rate (sr > 0), bounded
    duration (0 < len/sr <= :data:`MAX_DURATION_S`).  Returns a dict of the
    four booleans plus ``rms``/``duration_s`` diagnostics and the combined
    ``pass`` key.  ``None`` diagnostics (impossible-to-compute cases such as
    an invalid sample rate) stay visible instead of masquerading as numbers.
    """
    import numpy as np

    arr = np.asarray(wav)
    sr_i = int(sr)
    n = int(arr.size)

    valid_sr = sr_i > 0
    finite = bool(n > 0 and bool(np.isfinite(arr).all()))
    if finite:
        rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
        non_silent = bool(rms > SILENCE_RMS_THRESHOLD)
    else:
        rms = None
        non_silent = False
    if valid_sr:
        duration_s = n / sr_i
        bounded_duration = bool(0.0 < duration_s <= MAX_DURATION_S)
    else:
        duration_s = None
        bounded_duration = False

    checks: dict = {
        "finite": finite,
        "non_silent": non_silent,
        "valid_sr": bool(valid_sr),
        "bounded_duration": bounded_duration,
        "rms": None if rms is None else round(rms, 6),
        "duration_s": None if duration_s is None else round(duration_s, 3),
    }
    checks["pass"] = bool(finite and non_silent and valid_sr and bounded_duration)
    return checks


def summarize_results(sections: dict[str, list[dict]]) -> dict:
    """Aggregate result rows into counts + the per-language README matrix (pure).

    Matrix cells = custom_voice + voice_design + cross_lingual_clone rows;
    the auxiliary ``clone_reference_audio`` generations are counted separately
    but still gate ``all_passed`` (a broken reference invalidates its pairs).
    ``per_language`` keeps manifest (first-seen) order; each entry carries the
    CustomVoice / VoiceDesign pass flags and the cross-lingual clone hits with
    their reference language codes (e.g. ``ref_code`` "en", "fr").
    """
    section_stats = {
        key: {
            "cells": len(rows),
            "passed": sum(1 for r in rows if r.get("pass")),
        }
        for key, rows in sections.items()
    }
    matrix_rows = [r for key in MATRIX_SECTIONS for r in sections.get(key, [])]
    reference_rows = sections.get(REFERENCE_SECTION, [])

    order: list[str] = []
    for key in MATRIX_SECTIONS:
        for row in sections.get(key, []):
            if row["language"] not in order:
                order.append(row["language"])

    def _flag(section: str, language: str) -> bool | None:
        for row in sections.get(section, []):
            if row["language"] == language:
                return bool(row["pass"])
        return None

    per_language = []
    for language in order:
        clones = [
            {
                "ref_language": row["ref_language"],
                "ref_code": LANG_CODES.get(row["ref_language"], row["ref_language"]),
                "pass": bool(row["pass"]),
            }
            for row in sections.get("cross_lingual_clone", [])
            if row["language"] == language
        ]
        per_language.append({
            "language": language,
            "custom_voice": _flag("custom_voice", language),
            "voice_design": _flag("voice_design", language),
            "cross_lingual_clone": clones,
        })

    total = len(matrix_rows)
    passed = sum(1 for r in matrix_rows if r.get("pass"))
    ref_total = len(reference_rows)
    ref_passed = sum(1 for r in reference_rows if r.get("pass"))
    return {
        "sections": section_stats,
        "matrix_cells_total": total,
        "matrix_cells_passed": passed,
        "reference_cells_total": ref_total,
        "reference_cells_passed": ref_passed,
        "all_passed": bool(total > 0 and passed == total and ref_passed == ref_total),
        "per_language": per_language,
    }


# ---------------------------------------------------------------------------
# Environment metadata (guarded; heavy imports lazy).
# ---------------------------------------------------------------------------


def _git_head() -> str:
    """``git rev-parse HEAD`` of this repo; never raises ('unknown' fallback)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
            cwd=Path(__file__).resolve().parents[1],
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001,S110 - provenance probes never block the run
        pass
    return "unknown"


def collect_env(args: argparse.Namespace, manifest: Manifest) -> dict:
    """JSON 'meta' section: pinned platform facts + run configuration."""
    import torch

    try:
        import importlib.metadata as importlib_metadata

        qwen_tts_version = importlib_metadata.version("qwen-tts")
    except Exception:  # noqa: BLE001 - diagnostics degrade, never block
        qwen_tts_version = "unknown"

    gpu_name, gfx = "unavailable", "unavailable"
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            arch = str(getattr(props, "gcnArchName", "") or "")
            gfx = arch.split(":")[0] or "unknown"
    except Exception as exc:  # noqa: BLE001 - diagnostics degrade, never block
        gpu_name = f"probe failed: {exc}"

    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "gpu_name": gpu_name,
        "gfx": gfx,
        "torch_version": str(torch.__version__),
        "rocm_hip_version": str(getattr(torch.version, "hip", None)),
        "qwen_tts_version": qwen_tts_version,
        "git_head": _git_head(),
        "upstream_qwen3_tts_sha": UPSTREAM_QWEN3_TTS_SHA,
        "manifest": str(args.manifest),
        "manifest_languages": list(manifest.languages),
        "clone_pairs": [
            {"ref_language": p.ref_language, "target_language": p.target_language}
            for p in manifest.cross_lingual_clone_pairs
        ],
        "max_new_tokens": int(args.max_new_tokens),
        "claim": CLAIM,
        "cross_lingual_clone_transcript_rule": (
            "ref_text passed to create_voice_clone_prompt is exactly the "
            "manifest sentence used to synthesize the reference audio via "
            "generate_custom_voice"
        ),
        "supported_languages_runtime": None,  # filled after the first model load
        "sections": {},  # filled per phase below
    }


# ---------------------------------------------------------------------------
# Generation plumbing (GPU; heavy imports lazy, failures become failed rows).
# ---------------------------------------------------------------------------


def _result_row(*, section: str, language: str, language_id: str, wall_s: float,
                checks: dict | None, extra: dict | None = None,
                error: str | None = None) -> dict:
    """One evidence row: language, timings, checks, pass, error (+extras)."""
    if checks is None:
        audio_s, embedded, passed = None, None, False
    else:
        audio_s = checks.get("duration_s")
        embedded = {k: v for k, v in checks.items() if k != "pass"}
        passed = bool(checks["pass"])
    row: dict = {
        "section": section,
        "language": language,
        "language_id": language_id,
        "wall_s": round(float(wall_s), 3),
        "audio_s": audio_s,
        "checks": embedded,
        "pass": passed,
        "error": error,
    }
    if extra:
        row.update(extra)
    return row


def _print_cell(tag: str, language: str, language_id: str, row: dict) -> None:
    """One progress line per cell so the teed transcript is self-describing."""
    if row["pass"]:
        detail = f"audio={row['audio_s']}s"
    elif row["error"]:
        detail = f"error={str(row['error'])[:300]}"
    else:
        detail = f"checks={row['checks']}"
    print(
        f"[{tag}] language={language} id={language_id} "
        f"wall={row['wall_s']:.2f}s pass={row['pass']} {detail}",
        flush=True,
    )


def _generate_cell(model, method: str, call_kwargs: dict, *, section: str,
                   language: str, language_id: str, max_new_tokens: int,
                   extra: dict | None = None, label: str | None = None):
    """One timed keyword-first official call + waveform sanity gate.

    ``max_new_tokens`` is appended to the official kwargs on EVERY call (the
    project latency guardrail).  Returns ``(row, wav, sr)``; on failure the
    row carries the error text and ``wav``/``sr`` are ``None`` -- a failed
    cell never crashes the run.
    """
    t0 = time.perf_counter()
    try:
        wavs, sr = getattr(model, method)(
            **call_kwargs, max_new_tokens=max_new_tokens
        )
        wall = time.perf_counter() - t0
        if not wavs:
            raise ValueError(f"{method} returned an empty waveform list")
        checks = evaluate_waveform(wavs[0], int(sr))
        row = _result_row(section=section, language=language,
                          language_id=language_id, wall_s=wall, checks=checks,
                          extra=extra)
        wav, sr_out = wavs[0], int(sr)
    except Exception as exc:  # noqa: BLE001 - generation failures become rows
        wall = time.perf_counter() - t0
        row = _result_row(section=section, language=language,
                          language_id=language_id, wall_s=wall, checks=None,
                          extra=extra,
                          error=f"{type(exc).__name__}: {exc}")
        wav, sr_out = None, None
    _print_cell(label or section, language, language_id, row)
    return row, wav, sr_out


def run_custom_voice_matrix(model, manifest: Manifest, *, supported: list[str],
                            max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
    """CustomVoice matrix: one ``generate_custom_voice`` cell per language."""
    speaker = model.get_supported_speakers()[0]
    print(
        f"[custom-voice] speaker={speaker!r} (first of "
        "get_supported_speakers())",
        flush=True,
    )
    rows: list[dict] = []
    for sample in manifest:
        language_id = resolve_language(sample.language, supported)
        row, _, _ = _generate_cell(
            model, "generate_custom_voice",
            {"text": sample.text, "language": language_id, "speaker": speaker},
            section="custom_voice", language=sample.language,
            language_id=language_id, max_new_tokens=max_new_tokens, label="cv",
        )
        rows.append(row)
    return rows


def synthesize_clone_references(model, manifest: Manifest, *, supported: list[str],
                                max_new_tokens: int = MAX_NEW_TOKENS):
    """Reference audio for the clone pairs, synthesized on CustomVoice (GPU).

    One ``generate_custom_voice`` render per DISTINCT pair reference language
    (the Base model resident later cannot generate from scratch, so references
    must be produced in this phase and carried across the unload as plain
    numpy arrays).  Returns ``(rows, refs)`` where ``refs`` maps each ref
    language to ``(wav, sr, ref_text)`` -- *ref_text* is the exact manifest
    sentence that was synthesized, preserved for the transcript rule -- and
    only sanely-rendered references are kept.
    """
    distinct: list[str] = []
    for pair in manifest.cross_lingual_clone_pairs:
        if pair.ref_language not in distinct:
            distinct.append(pair.ref_language)
    print(
        f"[cv-ref] synthesizing clone-reference audio for {distinct} "
        "(transcript rule: ref_text == the exact manifest sentence below)",
        flush=True,
    )
    speaker = model.get_supported_speakers()[0]
    rows: list[dict] = []
    refs: dict[str, tuple] = {}
    for language in distinct:
        sample = manifest.sample_for(language)
        language_id = resolve_language(language, supported)
        row, wav, sr = _generate_cell(
            model, "generate_custom_voice",
            {"text": sample.text, "language": language_id, "speaker": speaker},
            section=REFERENCE_SECTION, language=language,
            language_id=language_id, max_new_tokens=max_new_tokens,
            label="cv-ref", extra={"role": "clone_reference"},
        )
        rows.append(row)
        if row["pass"]:
            refs[language] = (wav, sr, sample.text)
    return rows, refs


def run_voice_design_matrix(model, manifest: Manifest, *, supported: list[str],
                            max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
    """VoiceDesign matrix: one ``generate_voice_design`` cell per language."""
    rows: list[dict] = []
    for sample in manifest:
        language_id = resolve_language(sample.language, supported)
        instruct = manifest.voice_design_descriptions[sample.language]
        row, _, _ = _generate_cell(
            model, "generate_voice_design",
            {"text": sample.text, "instruct": instruct, "language": language_id},
            section="voice_design", language=sample.language,
            language_id=language_id, max_new_tokens=max_new_tokens, label="vd",
        )
        rows.append(row)
    return rows


def run_cross_lingual_clone(model, manifest: Manifest, *, supported: list[str],
                            refs: dict[str, tuple],
                            max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
    """Cross-lingual clone pairs on the Base model (GPU, representative).

    Per pair (transcript rule, binding): the reference audio was synthesized
    from the manifest sentence ``ref_text``; here it goes into
    ``create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=ref_text)``
    unchanged, and the returned prompt drives ``generate_voice_clone`` on the
    target-language sentence.  ``wall_s`` covers prompt creation + generation
    (split out as ``prompt_s`` / ``generate_s``); the exact ``ref_text`` is
    recorded in the row for audit.
    """
    rows: list[dict] = []
    for pair in manifest.cross_lingual_clone_pairs:
        target = manifest.sample_for(pair.target_language)
        target_id = resolve_language(pair.target_language, supported)
        ref_id = resolve_language(pair.ref_language, supported)
        extra = {"ref_language": pair.ref_language, "ref_language_id": ref_id}

        entry = refs.get(pair.ref_language)
        if entry is None:
            row = _result_row(
                section="cross_lingual_clone", language=pair.target_language,
                language_id=target_id, wall_s=0.0, checks=None, extra=extra,
                error=(
                    f"reference audio for {pair.ref_language!r} did not render "
                    "sanely in the custom-voice phase; pair skipped"
                ),
            )
            _print_cell("clone", pair.target_language, target_id, row)
            rows.append(row)
            continue

        wav, sr_ref, ref_text = entry
        t0 = time.perf_counter()
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=(wav, sr_ref), ref_text=ref_text
            )
            prompt_s = time.perf_counter() - t0
            tg0 = time.perf_counter()
            wavs, sr = model.generate_voice_clone(
                text=target.text,
                language=target_id,
                voice_clone_prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            generate_s = time.perf_counter() - tg0
            wall = time.perf_counter() - t0
            if not wavs:
                raise ValueError("generate_voice_clone returned an empty waveform list")
            checks = evaluate_waveform(wavs[0], int(sr))
            row = _result_row(
                section="cross_lingual_clone", language=pair.target_language,
                language_id=target_id, wall_s=wall, checks=checks,
                extra={
                    **extra,
                    "prompt_s": round(prompt_s, 3),
                    "generate_s": round(generate_s, 3),
                    "ref_text": ref_text,
                },
            )
        except Exception as exc:  # noqa: BLE001 - generation failures become rows
            wall = time.perf_counter() - t0
            row = _result_row(
                section="cross_lingual_clone", language=pair.target_language,
                language_id=target_id, wall_s=wall, checks=None, extra=extra,
                error=f"{type(exc).__name__}: {exc}",
            )
        _print_cell("clone", pair.target_language, target_id, row)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main driver: one model resident at a time, evidence out, exit code = verdict.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="validate_languages",
        description="Multilingual Radeon validation matrix for qwen3-tts-rocm (Task 2)",
    )
    ap.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"multilingual sample manifest (default: {DEFAULT_MANIFEST})",
    )
    ap.add_argument(
        "--json-out", type=Path, default=Path(DEFAULT_JSON_OUT),
        help=f"machine-readable evidence JSON (default: {DEFAULT_JSON_OUT})",
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help="official sampling cap for EVERY render (default: %(default)s)",
    )
    return ap.parse_args(argv)


def _drift_gate(manifest: Manifest, supported: list[str]) -> str | None:
    """Hard-error message for manifest <-> installed-package drift, else None."""
    try:
        for language in manifest.languages:
            resolve_language(language, supported)
    except KeyError as exc:
        return str(exc)
    missing = check_coverage(list(manifest.languages), supported)
    if missing:
        return (
            f"installed model supports language(s) {missing} that the manifest "
            f"{list(manifest.languages)} does not exercise; extend the manifest "
            "(drift is a hard error, never a silent skip)"
        )
    return None


def _mark(flag: bool | None) -> str:
    return "—" if flag is None else ("✅" if flag else "❌")


def _clone_cell(clones: list[dict]) -> str:
    if not clones:
        return "—"
    codes = ", ".join(str(c["ref_code"]) for c in clones)
    return f"{_mark(all(c['pass'] for c in clones))} (ref: {codes})"


def _print_summary(summary: dict) -> None:
    print("\n== Multilingual capability matrix (summary) ==", flush=True)
    print("| Language | CustomVoice | VoiceDesign | Clone (cross-lingual) |")
    print("|---|---|---|---|")
    for entry in summary["per_language"]:
        print(
            f"| {entry['language']} | {_mark(entry['custom_voice'])} "
            f"| {_mark(entry['voice_design'])} | {_clone_cell(entry['cross_lingual_clone'])} |"
        )
    print(f"\n{CAPTION}", flush=True)
    print(
        f"[summary] matrix cells: {summary['matrix_cells_passed']}/"
        f"{summary['matrix_cells_total']} passed; clone-reference generations: "
        f"{summary['reference_cells_passed']}/{summary['reference_cells_total']} "
        f"passed; wall total {summary.get('wall_seconds_total', 0)}s",
        flush=True,
    )
    verdict = "PASS" if summary["all_passed"] else "FAIL"
    print(f"[summary] RESULT: {verdict}", flush=True)


def _dump_evidence(path: Path, evidence: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[evidence] wrote {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Heavy/lazy imports happen here, NOT at module import time.
    from qwen3_tts_rocm import loader, models

    started = time.perf_counter()
    env = collect_env(args, manifest)
    print("== qwen3-tts-rocm multilingual validation matrix (Task 2) ==")
    print(f"# timestamp={env['timestamp']}")
    print(f"# gpu={env['gpu_name']} gfx={env['gfx']}")
    print(
        f"# torch={env['torch_version']} hip={env['rocm_hip_version']} "
        f"qwen-tts={env['qwen_tts_version']}"
    )
    print(f"# git_head={env['git_head']}")
    print(f"# upstream_qwen3_tts_sha={env['upstream_qwen3_tts_sha']}")
    print(
        f"# manifest={env['manifest']} ({len(manifest)} languages, "
        f"{len(manifest.cross_lingual_clone_pairs)} clone pairs)"
    )
    print(f"# max_new_tokens={env['max_new_tokens']} (every generation)")
    print(f"# claim: {CLAIM}")

    results: dict[str, list[dict]] = {}

    def _section_meta(alias: str, load_s: float, supported: list[str],
                      note: str | None = None) -> dict:
        meta = {
            "model_alias": alias,
            "model_repo": models.REPOS[alias],
            "load_seconds": round(load_s, 3),
            "supported_languages": supported,
        }
        if note:
            meta["note"] = note
        return meta

    def _load_phase(alias: str):
        """Load one alias + run the drift gate; never leaks a loaded model."""
        print(f"[phase] loading alias={alias} ...", flush=True)
        t0 = time.perf_counter()
        try:
            model = loader.load(alias)
        except Exception as exc:  # noqa: BLE001 - infra failure aborts loudly
            return None, time.perf_counter() - t0, None, (
                f"loader.load({alias!r}) failed: {type(exc).__name__}: {exc}"
            )
        load_s = time.perf_counter() - t0
        supported = [str(x) for x in model.get_supported_languages()]
        print(
            f"[phase] loaded alias={alias} took={load_s:.1f}s "
            f"supported_languages={supported}",
            flush=True,
        )
        drift = _drift_gate(manifest, supported)
        if drift is None:
            return model, load_s, supported, None
        loader.unload(model)
        return None, load_s, supported, drift

    def _dump_abort(reason: str) -> int:
        env["aborted_reason"] = reason
        summary = summarize_results(results)
        summary["wall_seconds_total"] = round(time.perf_counter() - started, 1)
        print(f"[summary] RESULT: ABORTED ({reason})", flush=True)
        _dump_evidence(args.json_out, {"meta": env, "results": results, "summary": summary})
        return 2

    # --- Phase 1: custom-voice (10 matrix cells + clone-reference audio) ----
    model, load_s, supported, problem = _load_phase("custom-voice")
    if model is None:
        return _dump_abort(problem)
    env["supported_languages_runtime"] = supported
    env["sections"]["custom_voice"] = _section_meta("custom-voice", load_s, supported)
    try:
        results["custom_voice"] = run_custom_voice_matrix(
            model, manifest, supported=supported, max_new_tokens=args.max_new_tokens
        )
        ref_rows, refs = synthesize_clone_references(
            model, manifest, supported=supported, max_new_tokens=args.max_new_tokens
        )
        results[REFERENCE_SECTION] = ref_rows
        env["sections"][REFERENCE_SECTION] = _section_meta(
            "custom-voice", load_s, supported,
            note="reference audio for the cross-lingual clone pairs, "
            "synthesized on custom-voice (transcript rule: ref_text == the "
            "exact manifest sentence)",
        )
    finally:
        print("[phase] unloading alias=custom-voice", flush=True)
        loader.unload(model)

    # --- Phase 2: voice-design (10 matrix cells) ----------------------------
    model, load_s, supported, problem = _load_phase("voice-design")
    if model is None:
        return _dump_abort(problem)
    env["sections"]["voice_design"] = _section_meta("voice-design", load_s, supported)
    try:
        results["voice_design"] = run_voice_design_matrix(
            model, manifest, supported=supported, max_new_tokens=args.max_new_tokens
        )
    finally:
        print("[phase] unloading alias=voice-design", flush=True)
        loader.unload(model)

    # --- Phase 3: base (4 cross-lingual clone cells) ------------------------
    model, load_s, supported, problem = _load_phase("base")
    if model is None:
        return _dump_abort(problem)
    env["sections"]["cross_lingual_clone"] = _section_meta("base", load_s, supported)
    try:
        results["cross_lingual_clone"] = run_cross_lingual_clone(
            model, manifest, supported=supported, refs=refs,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        print("[phase] unloading alias=base", flush=True)
        loader.unload(model)

    summary = summarize_results(results)
    summary["wall_seconds_total"] = round(time.perf_counter() - started, 1)
    _print_summary(summary)
    _dump_evidence(
        args.json_out, {"meta": env, "results": results, "summary": summary}
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
