# tests/test_validate_languages.py
"""Task 2: unit pin for scripts/validate_languages.py's PURE helpers.

Scope note (mirrors tests/test_benchmark.py): the matrix driver itself
(model loads, generations, JSON emit) runs live on the Radeon GPU and its
verbatim transcript is archived under evidence/multilingual-matrix.txt --
no test doubles are attempted for the GPU path.  What is trivially pinnable
and worth protecting against silent regression is everything every reported
cell flows through: manifest schema validation, canonical->runtime language
resolution (drift must be a LOUD error, never a silent skip), waveform
sanity arithmetic, and the aggregation that the README matrix is rendered
from.  The script is loaded via importlib from its path because
``scripts/`` is not an installed package; the module stays stdlib-only at
import time by design, so this import costs nothing (no torch, no GPU).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_languages.py"
_spec = importlib.util.spec_from_file_location("validate_languages_script", _SCRIPT)
vl = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("validate_languages_script", vl)
_spec.loader.exec_module(vl)

_REAL_MANIFEST = Path(__file__).resolve().parents[1] / "tests" / "data" / (
    "multilingual_samples.json")

#: The verbatim runtime list recorded in evidence/ground-truth-2026-09-20.md
#: section 5a (wrapper level, sorted): the installed qwen-tts 0.1.1 returns
#: exactly these identifiers from get_supported_languages() on every TTS
#: checkpoint shipped here.
RUNTIME_LANGS = [
    "auto", "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
]


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    """Dump *payload* as the manifest JSON a load_manifest() call reads."""
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def _minimal_payload() -> dict:
    """Internally consistent two-language manifest used as the error-case base."""
    return {
        "samples": [
            {"language": "Chinese", "text": "第一句"},
            {"language": "English", "text": "Second sentence."},
        ],
        "voice_design_descriptions": {
            "Chinese": "年轻女性，声音清亮",
            "English": "A calm male voice",
        },
        "cross_lingual_clone_pairs": [
            {"ref_language": "English", "target_language": "Chinese"},
        ],
    }


# ---------------------------------------------------------------------------
# load_manifest: schema + internal-consistency validation
# ---------------------------------------------------------------------------


def test_real_fixture_loads_consistently():
    """The shipped manifest: 10 languages, 10 descriptions, 4 clone pairs."""
    m = vl.load_manifest(_REAL_MANIFEST)
    assert len(m) == 10
    assert [s.language for s in m] == [
        "Chinese", "English", "Japanese", "Korean", "German", "French",
        "Russian", "Portuguese", "Spanish", "Italian",
    ]
    assert all(s.text.strip() for s in m.samples)
    assert set(m.voice_design_descriptions) == set(m.languages)
    assert len(m.cross_lingual_clone_pairs) == 4
    for pair in m.cross_lingual_clone_pairs:
        assert pair.ref_language != pair.target_language  # cross-lingual


def test_load_manifest_rejects_missing_sample_keys(tmp_path):
    """A sample entry missing 'text' fails loudly, naming index and key."""
    payload = _minimal_payload()
    del payload["samples"][0]["text"]
    with pytest.raises(ValueError, match="sample.*0.*text"):
        vl.load_manifest(_write_manifest(tmp_path, payload))


def test_load_manifest_rejects_unknown_language_references(tmp_path):
    """Clone pairs / descriptions naming a language with no sample are errors."""
    payload = _minimal_payload()
    payload["cross_lingual_clone_pairs"] = [
        {"ref_language": "Klingon", "target_language": "Chinese"},
    ]
    with pytest.raises(ValueError, match="Klingon"):
        vl.load_manifest(_write_manifest(tmp_path, payload))

    payload = _minimal_payload()
    payload["voice_design_descriptions"].pop("English")
    with pytest.raises(ValueError, match="English"):
        vl.load_manifest(_write_manifest(tmp_path, payload))


def test_load_manifest_rejects_structural_and_duplicate_errors(tmp_path):
    """Non-list samples, duplicate languages and mono-lingual pairs all fail."""
    payload = _minimal_payload()
    payload["samples"] = "not-a-list"
    with pytest.raises(TypeError, match="samples"):
        vl.load_manifest(_write_manifest(tmp_path, payload))

    payload = _minimal_payload()
    payload["samples"].append({"language": "Chinese", "text": "重复语言"})
    with pytest.raises(ValueError, match="duplicate"):
        vl.load_manifest(_write_manifest(tmp_path, payload))

    payload = _minimal_payload()
    payload["cross_lingual_clone_pairs"] = [
        {"ref_language": "English", "target_language": "English"},
    ]
    with pytest.raises(ValueError, match="cross"):
        vl.load_manifest(_write_manifest(tmp_path, payload))


# ---------------------------------------------------------------------------
# resolve_language / check_coverage: manifest <-> installed-package drift
# ---------------------------------------------------------------------------


def test_resolve_language_matches_case_insensitively():
    """Canonical names map onto the runtime identifiers; runtime spelling wins."""
    assert vl.resolve_language("Chinese", RUNTIME_LANGS) == "chinese"
    assert vl.resolve_language("JAPANESE", RUNTIME_LANGS) == "japanese"
    assert vl.resolve_language("portuguese", RUNTIME_LANGS) == "portuguese"
    # The runtime identifier is returned verbatim, not the canonical input.
    assert vl.resolve_language("German", ["Auto", "German"]) == "German"


def test_resolve_language_drift_raises_naming_both_sides():
    """A manifest language missing from the package is a hard error, not a skip."""
    with pytest.raises(KeyError) as excinfo:
        vl.resolve_language("Chinese", ["auto", "english"])
    msg = str(excinfo.value)
    assert "Chinese" in msg and "english" in msg  # both sides named


def test_check_coverage_flags_uncovered_runtime_languages():
    """Package languages absent from the manifest (minus 'auto') are reported."""
    assert vl.check_coverage(["Chinese", "English"], RUNTIME_LANGS) == [
        "french", "german", "italian", "japanese", "korean",
        "portuguese", "russian", "spanish",
    ]
    assert vl.check_coverage(vl.load_manifest(_REAL_MANIFEST).languages,
                             RUNTIME_LANGS) == []


# ---------------------------------------------------------------------------
# evaluate_waveform: the only capability gate (waveform sanity, nothing more)
# ---------------------------------------------------------------------------


def _sine(seconds: float = 0.5, sr: int = 24000, freq: float = 440.0,
          amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_evaluate_waveform_passes_sane_sine_burst():
    res = vl.evaluate_waveform(_sine(), 24000)
    assert res["pass"] is True
    assert all(res[k] for k in ("finite", "non_silent", "valid_sr",
                                "bounded_duration"))
    assert res["duration_s"] == pytest.approx(0.5, abs=1e-6)
    assert res["rms"] == pytest.approx(0.3 / np.sqrt(2), rel=1e-3)


def test_evaluate_waveform_flags_nan():
    wav = _sine()
    wav[100] = np.nan
    res = vl.evaluate_waveform(wav, 24000)
    assert res["pass"] is False and res["finite"] is False


def test_evaluate_waveform_flags_silent():
    res = vl.evaluate_waveform(np.zeros(24000, dtype=np.float32), 24000)
    assert res["pass"] is False
    assert res["finite"] is True and res["non_silent"] is False
    assert res["rms"] <= 1e-3


def test_evaluate_waveform_flags_invalid_sample_rate():
    for sr in (0, -16000):
        res = vl.evaluate_waveform(_sine(seconds=0.1), sr)
        assert res["pass"] is False and res["valid_sr"] is False


def test_evaluate_waveform_flags_overlong_and_empty():
    sr = 8000
    too_long = _sine(seconds=vl.MAX_DURATION_S + 1.0, sr=sr, freq=220.0)
    res = vl.evaluate_waveform(too_long, sr)
    assert res["pass"] is False and res["bounded_duration"] is False

    res = vl.evaluate_waveform(np.zeros(0, dtype=np.float32), 24000)
    assert res["pass"] is False


# ---------------------------------------------------------------------------
# summarize_results: the README matrix renders from this aggregation
# ---------------------------------------------------------------------------


def _row(language: str, passed: bool, **extra) -> dict:
    row = {"language": language, "pass": passed, "error": None}
    row.update(extra)
    return row


def test_summarize_results_counts_and_per_language_shape():
    sections = {
        "custom_voice": [_row("Chinese", True), _row("English", True)],
        "voice_design": [_row("Chinese", True), _row("English", False)],
        "clone_reference_audio": [_row("English", True)],
        "cross_lingual_clone": [
            _row("Chinese", True, ref_language="English"),
        ],
    }
    s = vl.summarize_results(sections)
    assert s["matrix_cells_total"] == 5 and s["matrix_cells_passed"] == 4
    assert s["reference_cells_total"] == 1 and s["reference_cells_passed"] == 1
    assert s["all_passed"] is False
    zh = s["per_language"][0]
    assert zh["language"] == "Chinese"
    assert zh["custom_voice"] is True and zh["voice_design"] is True
    assert zh["cross_lingual_clone"] == [
        {"ref_language": "English", "ref_code": "en", "pass": True},
    ]
    en = s["per_language"][1]
    assert en["voice_design"] is False and en["cross_lingual_clone"] == []
