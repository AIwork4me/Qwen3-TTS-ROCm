#!/usr/bin/env python3
"""Build the Task 4 self-generated fine-tuning corpus (execution smoke only).

Renders every utterance in ``tests/data/finetune_manifest.json`` with the
OFFICIAL Qwen3-TTS CustomVoice model loaded through our smart-default loader
(``qwen3_tts_rocm.loader.load("custom-voice")``), so the training corpus is
100% self-generated speech — no third-party recording is ever touched.  The
output is exactly the JSONL format upstream's ``finetuning/prepare_data.py``
expects (ground-truth audit §4: one JSON object per line with keys ``audio``
/ ``text`` / ``ref_audio``, and "use the same ref_audio for all samples").

Discipline (from the task brief):

* ``torch.manual_seed(1234)`` before EACH render (mirrors the test-suite
  ``conftest._reseed``) for variance control;
* every render capped at ``max_new_tokens=512`` (official kwarg);
* transcript rule: the JSONL ``text`` is VERBATIM the manifest text used to
  generate the wav — no transcript guessing is ever done;
* every wav passes ``qwen3_tts_rocm.testing.assert_wav_sane`` before it is
  written into the manifest.

This is an EXECUTION-VALIDATION dataset (smoke), not a quality corpus: 12
short utterances + 1 reference clip, all synthesised from the same official
speaker.  No speaker-similarity or convergence claims are made anywhere.

Usage:
    .venv/bin/python scripts/make_finetune_dataset.py \
        [--manifest tests/data/finetune_manifest.json] \
        [--output-dir .work-finetune/data] \
        [--speaker serena]

Outputs (under --output-dir):
    ref_speaker.wav     shared reference clip (ref_audio for every sample)
    utt_<id>.wav        one wav per manifest sample
    train_raw.jsonl     upstream prepare_data.py input schema
    dataset_report.txt  per-utterance duration/rms table (evidence-friendly)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_DEFAULT = _REPO_ROOT / "tests" / "data" / "finetune_manifest.json"
_OUTPUT_DEFAULT = _REPO_ROOT / ".work-finetune" / "data"

#: Official sampling cap used for every render (brief-mandated).
MAX_NEW_TOKENS = 512

#: torch RNG reset before each render (mirrors tests/conftest._reseed).
_RESEED = 1234


# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable; no model involved)
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> dict:
    """Load and validate the fine-tuning manifest JSON.

    Returns the parsed dict.  Raises ValueError when the shape is wrong:
    exactly 12 samples (6 Chinese + 6 English), unique non-empty ids, unique
    non-empty texts, plus one ``reference`` entry.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "samples" not in data or "reference" not in data:
        raise ValueError(
            f"manifest {path} must be a JSON object with 'samples' and 'reference'"
        )
    samples = data["samples"]
    if not isinstance(samples, list) or len(samples) != 12:
        raise ValueError(f"manifest must carry exactly 12 samples, got {len(samples)}")
    langs = [s.get("language") for s in samples]
    n_zh = sum(1 for x in langs if str(x).lower() == "chinese")
    n_en = sum(1 for x in langs if str(x).lower() == "english")
    if (n_zh, n_en) != (6, 6):
        raise ValueError(
            f"manifest must carry 6 Chinese + 6 English samples, got {n_zh}+{n_en}"
        )
    ids = [s.get("id") for s in samples]
    if len(set(ids)) != 12 or any(not i for i in ids):
        raise ValueError(f"sample ids must be 12 unique non-empty strings, got {ids}")
    texts = [s.get("text") for s in samples]
    if len(set(texts)) != 12 or any(not t or not str(t).strip() for t in texts):
        raise ValueError("sample texts must be 12 unique non-empty strings")
    ref = data["reference"]
    if not isinstance(ref, dict) or not str(ref.get("text", "")).strip():
        raise ValueError("manifest 'reference' must be an object with non-empty 'text'")
    return data


def build_jsonl_records(samples: list[dict], wav_paths: list[str | Path],
                         ref_audio: str | Path) -> list[dict]:
    """Zip rendered wavs into upstream ``prepare_data.py`` JSONL records.

    The record schema is exactly upstream's input contract (ground truth §4):
    ``{"audio": <wav path>, "text": <transcript>, "ref_audio": <ref wav>}``.
    The transcript rule is enforced structurally: the text comes verbatim
    from the same manifest entry whose id names the wav, so the pair can
    never drift apart.  Raises ValueError on any length/mismatch problem.
    """
    if len(samples) != len(wav_paths):
        raise ValueError(
            f"samples ({len(samples)}) and wav_paths ({len(wav_paths)}) must align"
        )
    records = []
    for sample, wav in zip(samples, wav_paths):
        text = str(sample["text"])
        if not text.strip():
            raise ValueError(f"sample {sample.get('id')!r} has empty text")
        records.append({
            "audio": str(wav),
            "text": text,
            "ref_audio": str(ref_audio),
        })
    return records


# ---------------------------------------------------------------------------
# GPU render driver
# ---------------------------------------------------------------------------

def _reseed() -> None:
    """torch RNG reset before each render (variance control, brief rule)."""
    import torch

    torch.manual_seed(_RESEED)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the self-generated fine-tuning corpus (official model)"
    )
    parser.add_argument("--manifest", type=Path, default=_MANIFEST_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DEFAULT)
    parser.add_argument("--speaker", type=str, default="serena",
                        help="official CustomVoice speaker id (default: serena)")
    parser.add_argument("--language-mode", type=str, default="explicit",
                        choices=["explicit", "auto"],
                        help="pass each sample's language, or 'Auto' for all")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from qwen3_tts_rocm import loader, testing

    model = loader.load("custom-voice")
    speakers = model.get_supported_speakers()
    if str(args.speaker).lower() not in speakers:
        raise SystemExit(
            f"speaker {args.speaker!r} not in official speakers {speakers}"
        )

    def render(text: str, language: str) -> tuple[list, int]:
        _reseed()
        lang = language if args.language_mode == "explicit" else "Auto"
        t0 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=args.speaker, language=lang,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"[render] took={time.perf_counter() - t0:.1f}s sr={sr} "
              f"lang={lang} chars={len(text)}")
        return wavs, sr

    report_lines = [
        f"manifest: {args.manifest}",
        (
            f"speaker: {args.speaker}  language-mode: {args.language_mode}  "
            f"max_new_tokens: {MAX_NEW_TOKENS}  reseed-per-render: {_RESEED}"
        ),
    ]

    # 1) shared reference clip (ref_audio for every training sample)
    ref = manifest["reference"]
    print(f"[ref] rendering shared reference clip: {ref['text']}")
    wavs, sr = render(ref["text"], ref["language"])
    testing.assert_wav_sane(wavs[0], sr_expected=sr)
    ref_path = out_dir / "ref_speaker.wav"
    testing.write_wav(ref_path, wavs[0], sr)

    # 2) the 12 manifest utterances
    wav_paths: list[Path] = []
    for sample in manifest["samples"]:
        text, sid = str(sample["text"]), str(sample["id"])
        print(f"[utt] {sid} ({sample['language']}): {text}")
        wavs, sr = render(text, str(sample["language"]))
        testing.assert_wav_sane(wavs[0], sr_expected=sr)
        path = out_dir / f"utt_{sid}.wav"
        testing.write_wav(path, wavs[0], sr)
        wav_paths.append(path)
        report_lines.append(
            f"{sid}\t{sample['language']}\t{len(wavs[0]) / sr:.2f}s\t{path.name}"
        )

    # 3) upstream-schema JSONL (verbatim transcripts)
    records = build_jsonl_records(manifest["samples"], wav_paths, ref_path)
    jsonl_path = out_dir / "train_raw.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    (out_dir / "dataset_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8",
    )

    print(f"[done] {len(records)} records -> {jsonl_path}")
    print(f"[done] shared ref_audio -> {ref_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
