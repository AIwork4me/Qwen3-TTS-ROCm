"""CPU tests for scripts/make_finetune_dataset.py pure helpers (Task 4).

These cover the dataset generator's model-free surface: manifest validation
(12 samples, 6 Chinese + 6 English, unique ids/texts) and the upstream-schema
JSONL record builder (audio/text/ref_audio keys, verbatim transcripts).
No GPU, no model weights, no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "make_finetune_dataset.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_finetune_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_script()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(
        (_REPO / "tests" / "data" / "finetune_manifest.json").read_text("utf-8")
    )


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

def test_load_manifest_accepts_shipped_manifest(gen, manifest):
    data = gen.load_manifest(_REPO / "tests" / "data" / "finetune_manifest.json")
    assert len(data["samples"]) == 12
    assert data == manifest


def test_manifest_shape_6zh_6en_unique(gen, manifest):
    langs = [s["language"].lower() for s in manifest["samples"]]
    assert langs.count("chinese") == 6 and langs.count("english") == 6
    ids = [s["id"] for s in manifest["samples"]]
    texts = [s["text"] for s in manifest["samples"]]
    assert len(set(ids)) == 12
    assert len(set(texts)) == 12


def test_manifest_texts_distinct_from_multilingual_samples(manifest):
    """Task-4 corpus must not reuse any tests/data/multilingual_samples.json text."""
    multi = json.loads(
        (_REPO / "tests" / "data" / "multilingual_samples.json").read_text("utf-8")
    )
    multi_texts = {s["text"] for s in multi["samples"]}
    for s in manifest["samples"] + [manifest["reference"]]:
        assert s["text"] not in multi_texts


def test_load_manifest_rejects_bad_shapes(gen, tmp_path):
    def write(obj):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    with pytest.raises(ValueError):
        gen.load_manifest(write({"samples": [], "reference": {"text": "x"}}))  # != 12
    with pytest.raises(ValueError):
        gen.load_manifest(write({  # wrong language mix
            "samples": [{"id": f"z{i}", "language": "Chinese", "text": str(i)}
                        for i in range(12)],
            "reference": {"text": "x"},
        }))
    with pytest.raises(ValueError):
        gen.load_manifest(write({  # duplicate ids
            "samples": [{"id": "dup", "language": l, "text": t}
                        for l, t in zip(["Chinese"] * 6 + ["English"] * 6,
                                        map(str, range(12)))],
            "reference": {"text": "x"},
        }))
    with pytest.raises(ValueError):
        gen.load_manifest(write({"samples": [{"id": "a"}] * 12}))  # no reference


# ---------------------------------------------------------------------------
# build_jsonl_records
# ---------------------------------------------------------------------------

def test_build_jsonl_records_upstream_schema(gen, manifest):
    samples = manifest["samples"]
    wavs = [f"/tmp/utt_{s['id']}.wav" for s in samples]
    records = gen.build_jsonl_records(samples, wavs, "/tmp/ref_speaker.wav")
    assert len(records) == 12
    for rec, sample, wav in zip(records, samples, wavs):
        # exactly upstream prepare_data.py's expected keys, verbatim transcript
        assert set(rec) == {"audio", "text", "ref_audio"}
        assert rec["audio"] == wav
        assert rec["text"] == sample["text"]  # transcript == manifest text
        assert rec["ref_audio"] == "/tmp/ref_speaker.wav"


def test_build_jsonl_records_shared_ref_audio(gen, manifest):
    records = gen.build_jsonl_records(
        manifest["samples"],
        [f"/x/{s['id']}.wav" for s in manifest["samples"]],
        "/shared/ref.wav",
    )
    assert {r["ref_audio"] for r in records} == {"/shared/ref.wav"}


def test_build_jsonl_records_length_mismatch(gen, manifest):
    with pytest.raises(ValueError):
        gen.build_jsonl_records(manifest["samples"], ["/a.wav"], "/ref.wav")


def test_build_jsonl_records_rejects_empty_text(gen):
    with pytest.raises(ValueError):
        gen.build_jsonl_records(
            [{"id": "x", "language": "English", "text": "   "}], ["/a.wav"], "/r.wav"
        )
