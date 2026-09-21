# tests/test_quality_eval.py
"""Task 14 (brief 15): CPU unit tests for scripts/quality_eval.py.

Pinned here, all hand-computable without any GPU / TTS model download:

* ``cer`` / ``wer`` on hand-computed examples (normalization behavior is the
  contract: case/punctuation/whitespace handling verified against jiwer 4.0.0
  semantics probed in the Step 14.1 gates,
  evidence/quality-benchmark-2026-09-21.txt);
* ``speaker_sim`` on synthetic monotone sines written to ``tmp_path`` via
  numpy + soundfile: identical tones -> cosine ~1.0, different frequencies ->
  distinctly lower (resemblyzer runs on CPU and its weights ship inside the
  wheel, so no GPU and no network are involved);
* the fixed manifest's schema (``load_manifest``) — the good file in
  tests/data plus rejections for every invalid shape;
* the omitted-metric documentation contract: PESQ's rejection is recorded in
  the harness output vocabulary, never silently dropped.

The script is loaded via importlib from its path because ``scripts/`` is not
an installed package (same pattern as tests/test_benchmark.py); the module is
stdlib-only at import time so this costs nothing.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quality_eval.py"
_spec = importlib.util.spec_from_file_location("quality_eval_script", _SCRIPT)
qev = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("quality_eval_script", qev)
_spec.loader.exec_module(qev)

#: The fixed manifest delivered with this task (Step 14.2).
MANIFEST = Path(__file__).resolve().parents[1] / "tests" / "data" / \
    "quality_benchmark_manifest.json"

#: jiwer is part of the optional ``[quality]`` extra too — same convention as
#: resemblyzer below: the cer/wer tests skip visibly (with the fix) when the
#: extras group is not installed, instead of erroring the plain CPU suite.
#: (Added by the 2026-09-21 claims audit: Task 14 shipped these tests with an
#: unguarded ``import jiwer`` inside ``cer``/``wer``, which turned CI red on
#: every plain ``.[dev]`` environment — 12 ModuleNotFoundError failures per
#: Python job, runs 35588552252 / 35590816441 / 35593965878.)
try:
    import jiwer  # noqa: F401

    _HAVE_JIWER = True
except Exception:  # noqa: BLE001 - extras not installed -> visible skip
    _HAVE_JIWER = False


requires_jiwer = pytest.mark.skipif(
    not _HAVE_JIWER,
    reason="jiwer (quality extras) not installed: "
           "pip install -e '.[quality]' to run cer/wer tests",
)

#: Resemblyzer is an optional ``[quality]`` extra — the speaker_sim test is
#: skipped (visibly, with the fix) when the extras group is not installed,
#: instead of erroring the plain CPU suite.
try:
    import resemblyzer  # noqa: F401

    _HAVE_RESEMBLYZER = True
except Exception:  # noqa: BLE001 - extras not installed -> visible skip
    _HAVE_RESEMBLYZER = False


# ---------------------------------------------------------------------------
# cer / wer: hand-computed examples
# ---------------------------------------------------------------------------

@requires_jiwer
def test_cer_hand_computed_substitution():
    """cer(你好世界, 你号世界) == 0.25 — one substitution of four chars."""
    assert math.isclose(qev.cer("你好世界", "你号世界"), 0.25)


@requires_jiwer
def test_cer_identical_is_zero():
    assert qev.cer("今天天气很好", "今天天气很好") == 0.0


@requires_jiwer
def test_cer_normalizes_punctuation_and_whitespace():
    """Sentence-final 。 and ASR-inserted spaces must not count as errors."""
    assert qev.cer("今天天气很好。", "今天天气很好") == 0.0
    assert qev.cer("今天天气很好", "今天 天气 很 好") == 0.0
    assert qev.cer("你好，世界！", "你好世界") == 0.0


@requires_jiwer
def test_cer_empty_hypothesis_is_one():
    """Empty transcript = every char deleted = CER 1.0 (worst, not a crash)."""
    assert qev.cer("今天天气很好", "") == 1.0


@requires_jiwer
def test_cer_one_insertion():
    """A stray extra character is one insertion over the ref length: 1/5."""
    assert math.isclose(qev.cer("天气很好", "天气很好好"), 0.25)


@requires_jiwer
def test_cer_empty_reference_raises():
    with pytest.raises(ValueError):
        qev.cer("。。。", "你好")  # ref normalizes to empty -> manifest bug


@requires_jiwer
def test_wer_hand_computed_substitution():
    assert math.isclose(qev.wer("hello world", "hello there world"), 0.5)


@requires_jiwer
def test_wer_identical_is_zero():
    assert qev.wer("the quick brown fox", "the quick brown fox") == 0.0


@requires_jiwer
def test_wer_normalizes_case_and_punctuation():
    assert qev.wer("The weather is nice today.", "the weather is nice today") == 0.0
    assert qev.wer("Hello, world!", "hello world") == 0.0


@requires_jiwer
def test_wer_one_substitution_of_five_words():
    assert math.isclose(
        qev.wer("The weather is nice today.", "The weather was nice today."), 0.2
    )


@requires_jiwer
def test_wer_empty_hypothesis_is_one():
    assert qev.wer("The weather is nice today.", "") == 1.0


@requires_jiwer
def test_wer_empty_reference_raises():
    with pytest.raises(ValueError):
        qev.wer("!!", "hello")  # ref normalizes to empty


def test_normalize_helpers_pure_shapes():
    assert qev.normalize_cer_text("你好， 世界！") == "你好世界"
    assert qev.normalize_wer_text("Hello,  World!\nhi.") == "hello world hi"


# ---------------------------------------------------------------------------
# speaker_sim: synthetic tones (CPU, no downloads)
# ---------------------------------------------------------------------------

def _tone_wav(path: Path, freq: float, seconds: float = 2.0,
              sr: int = 16000) -> Path:
    """Write a monotone sine to *path* (numpy + soundfile, tmp_path-based)."""
    import soundfile as sf

    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    wav = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wav, sr)
    return path


@pytest.mark.skipif(
    not _HAVE_RESEMBLYZER,
    reason="resemblyzer (quality extras) not installed: "
           "pip install -e '.[quality]' to run speaker_sim tests",
)
class TestSpeakerSimOnTones:
    """Identical sines embed identically (~cos 1.0); different frequencies
    embed distinctly lower. CPU-only: resemblyzer runs on cpu here and ships
    its weights inside the wheel (no model download, no GPU)."""

    def test_identical_tones_similarity_is_one(self, tmp_path):
        a = _tone_wav(tmp_path / "a_220.wav", 220.0)
        b = _tone_wav(tmp_path / "b_220.wav", 220.0)
        sim = qev.speaker_sim(a, b)
        assert sim > 0.99

    def test_different_frequencies_similarity_is_lower(self, tmp_path):
        a = _tone_wav(tmp_path / "a_220.wav", 220.0)
        c = _tone_wav(tmp_path / "c_2500.wav", 2500.0)
        sim_same = qev.speaker_sim(a, _tone_wav(tmp_path / "a2_220.wav", 220.0))
        sim_diff = qev.speaker_sim(a, c)
        assert sim_diff < 0.75
        assert sim_diff < sim_same

    def test_path_and_memory_inputs_agree(self, tmp_path):
        """Path input and the identical in-memory waveform agree."""
        a = _tone_wav(tmp_path / "a_220.wav", 220.0)
        import soundfile as sf

        wav, sr = sf.read(str(a), dtype="float32", always_2d=False)
        assert qev.speaker_sim(a, wav, sr=sr) > 0.99


# ---------------------------------------------------------------------------
# Manifest schema (Step 14.2 contract)
# ---------------------------------------------------------------------------

class TestManifest:
    def test_fixed_manifest_loads_with_expected_cases(self):
        data = qev.load_manifest(MANIFEST)
        assert data["seed"] == 20260921
        assert [c["id"] for c in data["cases"]] == [
            "zh-short", "zh-mid", "en-short", "en-mid", "clone-ref",
        ]

    def test_fixed_manifest_clone_case_points_at_real_asset(self):
        data = qev.load_manifest(MANIFEST)
        clone = data["cases"][-1]
        assert clone["role"] == "reference-clone"
        assert clone["lang"] == "zh"
        assert isinstance(clone["ref_text"], str) and clone["ref_text"]
        ref = Path(__file__).resolve().parents[1] / clone["ref"]
        assert ref.exists(), f"clone ref asset missing: {ref}"

    def test_fixed_manifest_texts_frozen(self):
        """The texts are the benchmark's fixed payload — pinned verbatim."""
        data = qev.load_manifest(MANIFEST)
        texts = {c["id"]: c["text"] for c in data["cases"]}
        assert texts["zh-short"] == "今天天气很好。"
        assert texts["zh-mid"] == "这个周末我们打算去山上徒步，顺便拍一些照片。"
        assert texts["en-short"] == "The weather is nice today."
        assert texts["en-mid"] == ("We are planning to go hiking this "
                                   "weekend and take some photos.")
        assert texts["clone-ref"] == "欢迎使用语音克隆功能。"


def _write(tmp_path: Path, payload: dict | list) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def test_manifest_rejects_missing_seed(tmp_path):
    cases = [{"id": "zh-short", "text": "你好", "lang": "zh"}]
    with pytest.raises(TypeError, match="seed"):
        qev.load_manifest(_write(tmp_path, {"cases": cases}))


def test_manifest_rejects_non_object_case(tmp_path):
    payload = {"seed": 1, "cases": ["not-an-object"]}
    with pytest.raises(TypeError, match="not an object"):
        qev.load_manifest(_write(tmp_path, payload))


def test_manifest_rejects_empty_cases(tmp_path):
    with pytest.raises(ValueError, match="cases"):
        qev.load_manifest(_write(tmp_path, {"seed": 1, "cases": []}))


def test_manifest_rejects_duplicate_ids(tmp_path):
    case = {"id": "zh-short", "text": "你好", "lang": "zh"}
    with pytest.raises(ValueError, match="duplicate"):
        qev.load_manifest(_write(tmp_path, {"seed": 1, "cases": [case, case]}))


def test_manifest_rejects_unknown_lang(tmp_path):
    case = {"id": "fr-short", "text": "bonjour", "lang": "fr"}
    with pytest.raises(ValueError, match="lang"):
        qev.load_manifest(_write(tmp_path, {"seed": 1, "cases": [case]}))


def test_manifest_rejects_clone_without_ref(tmp_path):
    case = {"id": "clone-ref", "text": "你好", "lang": "zh",
            "role": "reference-clone", "ref_text": "hello"}
    with pytest.raises(ValueError, match="'ref'"):
        qev.load_manifest(_write(tmp_path, {"seed": 1, "cases": [case]}))


def test_manifest_rejects_unknown_role(tmp_path):
    case = {"id": "odd", "text": "你好", "lang": "zh", "role": "mystery"}
    with pytest.raises(ValueError, match="role"):
        qev.load_manifest(_write(tmp_path, {"seed": 1, "cases": [case]}))


# ---------------------------------------------------------------------------
# Omitted-metric documentation contract (rejections recorded, not hidden)
# ---------------------------------------------------------------------------

def test_pesq_rejection_is_documented_in_output_vocabulary():
    """The REJECT verdict must appear in the harness's omitted-metrics notes."""
    note = qev.OMITTED_METRICS.get("pesq", "")
    assert "REJECT" in note
    assert "P.862" in note or "ITU" in note


def test_stoi_scope_note_distinguishes_scope_from_rejection():
    """pystoi is ACCEPTed but out of scope for text-conditioned cases."""
    note = qev.OMITTED_METRICS.get("stoi", "")
    assert "ACCEPT" in note
    assert "REJECT" not in note.split("ACCEPTed")[0]
