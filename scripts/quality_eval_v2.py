#!/usr/bin/env python3
"""Quality benchmark v2 (v0.2.1 Task 7): 10-language content correctness +
clone speaker-similarity controls, with full metric provenance.

Extends the v1 harness discipline (scripts/quality_eval.py, 2026-09-21) —
whose helper functions it imports unchanged — with:

* **All 10 officially supported languages** (fixed versioned manifest
  ``tests/data/quality_v2_manifest.json``): one same-intent sentence per
  language rendered through the official 1.7B CustomVoice API
  (``language=`` per case), transcribed by ASR, scored CER (zh/ja/ko/ru —
  character scripts) or WER (en/de/fr/pt/es/it — space-delimited scripts),
  each sample recording: intended text, generated WAV (archived), ASR
  transcript VERBATIM, normalization method, error rate, language, model,
  seed, audio duration.
* **ASR adequacy decision, explicit**: v1 used whisper ``tiny``; for the
  10-language grid v2 uses whisper ``small`` (openai-whisper), GPU-probed
  on gfx1151 BEFORE any scoring run (probe transcript recorded). Scores are
  NEVER compared across ASR models — the v1 (tiny) evidence stays its own
  dated record.
* **Clone speaker-similarity controls** (1.7B Base): same-reference
  positives (two independent clone renders vs the reference clip and each
  other) and a different-speaker negative (preset CustomVoice render),
  resemblyzer cosine similarities recorded verbatim. NO universal quality
  threshold is claimed — the numbers are the evidence; distributions are
  described, not gated.
* **No composite score, no MOS claim, no instruction/emotion adherence
  score** (recorded as not objectively scored).

Usage::

    .venv/bin/python scripts/quality_eval_v2.py \
        --manifest tests/data/quality_v2_manifest.json \
        --wav-dir evidence/quality-v2-wavs \
        --json-out evidence/quality-v2-gfx1151-2026-09-24.json

Importing this module is lightweight; heavy imports happen lazily.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_eval import (
    _read_audio,
    _resample_to_16k,
    cer,
    normalize_cer_text,
    normalize_wer_text,
    speaker_sim,
    wer,
)

#: Languages whose scripts are scored character-level (CJK + Cyrillic);
#: the rest are word-level.
CER_LANGS = {"zh", "ja", "ko", "ru"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quality benchmark v2 (Task 7)")
    ap.add_argument("--manifest", default="tests/data/quality_v2_manifest.json")
    ap.add_argument("--asr-model", default="small",
                    help="openai-whisper checkpoint name (default: small; v1 used tiny — never mix)")
    ap.add_argument("--wav-dir", default="evidence/quality-v2-wavs")
    ap.add_argument("--json-out", default="evidence/quality-v2.json")
    return ap.parse_args(argv)


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance probe never blocks
        return None


def _get_asr(name: str):
    """Load the whisper checkpoint once (downloading on first use)."""
    import torch
    import whisper

    model = whisper.load_model(name, device="cuda" if torch.cuda.is_available() else "cpu")
    return model


def _transcribe(model, wav_path: Path, lang: str) -> str:
    wav, sr = _read_audio(wav_path)
    wav16 = _resample_to_16k(wav, sr)
    result = model.transcribe(wav16, language=lang, temperature=0.0,
                              beam_size=1, condition_on_previous_text=False)
    return str(result["text"]).strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    import torch

    # --- ASR adequacy: GPU probe BEFORE any scoring --------------------------
    import whisper

    from qwen3_tts_rocm import loader

    asr_device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_t0 = time.perf_counter()
    asr = whisper.load_model(args.asr_model, device=asr_device)
    probe_load_s = round(time.perf_counter() - probe_t0, 1)
    from importlib.resources import files

    ref_path = Path(str(files("qwen3_tts_rocm") / "demo" / "assets" / "ref_en.wav"))
    rwav, rsr = _read_audio(ref_path)
    probe_t0 = time.perf_counter()
    probe_text = str(asr.transcribe(_resample_to_16k(rwav, rsr), language="en",
                                    temperature=0.0, beam_size=1,
                                    condition_on_previous_text=False)["text"]).strip()
    probe_s = round(time.perf_counter() - probe_t0, 1)
    print(f"[qv2] ASR probe: model={args.asr_model} device={asr_device} "
          f"load={probe_load_s}s transcribe={probe_s}s transcript={probe_text!r}",
          flush=True)

    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(), "manifest_version": manifest["manifest_version"],
        "asr": {"model": args.asr_model, "package": "openai-whisper",
                "device": asr_device, "probe": {"clip": "ref_en.wav",
                                                "transcript": probe_text,
                                                "load_seconds": probe_load_s,
                                                "transcribe_seconds": probe_s}},
        "seed_base": manifest["seed_base"],
        "metric_provenance": {
            "cer": "jiwer character error rate on normalize_cer_text (punct/ws-stripped, lowercased latin)",
            "wer": "jiwer word error rate on normalize_wer_text (lowercased, punct-stripped)",
            "speaker_sim": "resemblyzer VoiceEncoder embedding cosine (CPU), quality_eval.speaker_sim",
        },
        "claims_policy": ["no composite quality score", "no MOS claim",
                          "instruction/emotion adherence NOT objectively scored",
                          "v1 (whisper-tiny) scores never compared to v2 (whisper-small)"],
    }
    results: list[dict] = []

    # --- Section A: 10-language content correctness ---------------------------
    cv = loader.load("custom-voice")
    spk = cv.get_supported_speakers()[manifest["speaker_index"]]
    try:
        for i, case in enumerate(manifest["languages"]):
            lang_key = case["whisper_lang"]
            seed = manifest["seed_base"] + i
            wav_path = wav_dir / f"lang_{lang_key}.wav"
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            wavs, sr = cv.generate_custom_voice(
                text=case["text"], language=case["language"], speaker=spk,
                max_new_tokens=manifest["max_new_tokens"])
            gen_s = round(time.perf_counter() - t0, 1)
            import soundfile as sf

            sf.write(wav_path, wavs[0], sr)
            transcript = _transcribe(asr, wav_path, lang_key)
            is_cer = lang_key in CER_LANGS
            norm = normalize_cer_text if is_cer else normalize_wer_text
            score_fn = cer if is_cer else wer
            err = score_fn(norm(case["text"]), norm(transcript))
            dur = round(float(wavs[0].shape[-1]) / sr, 2)
            rec = {"section": "languages", "language": case["language"],
                   "whisper_lang": lang_key, "metric": "CER" if is_cer else "WER",
                   "normalization": "normalize_cer_text" if is_cer else "normalize_wer_text",
                   "intended": case["text"], "transcript": transcript,
                   "error_rate": round(err, 4), "audio_seconds": dur,
                   "seed": seed, "model": "custom-voice (1.7B)",
                   "speaker": str(spk), "generate_seconds": gen_s,
                   "wav": str(wav_path)}
            results.append(rec)
            print(f"[qv2] {case['language']:10s} {'CER' if is_cer else 'WER'}="
                  f"{err:.4f} dur={dur}s gen={gen_s}s transcript={transcript[:60]!r}",
                  flush=True)
    finally:
        loader.unload(cv)

    # --- Section B: clone speaker-similarity controls --------------------------
    base = loader.load("base")
    try:
        from benchmark import BASE_REF_TEXT, _load_base_ref_audio

        ref_wav, ref_sr = _load_base_ref_audio()
        sims: dict[str, float] = {}
        ctrl_wavs: dict[str, Path] = {}
        for j, case in enumerate(manifest["clone_controls"]):
            seed = manifest["seed_base"] + 100 + j
            wav_path = wav_dir / f"clone_{case['case']}.wav"
            torch.manual_seed(seed)
            if case["origin"].startswith("ref_en.wav"):
                wavs, sr = base.generate_voice_clone(
                    text=case["text"], language="English",
                    ref_audio=(ref_wav, ref_sr), ref_text=BASE_REF_TEXT,
                    max_new_tokens=manifest["max_new_tokens"])
            else:
                # negative control: different speaker by construction ->
                # rendered on the CustomVoice family (Base has no preset speakers)
                cv2 = loader.load("custom-voice")
                torch.manual_seed(seed)
                wavs, sr = cv2.generate_custom_voice(
                    text=case["text"], language="English", speaker=spk,
                    max_new_tokens=manifest["max_new_tokens"])
                loader.unload(cv2)
            import soundfile as sf

            sf.write(wav_path, wavs[0], sr)
            ctrl_wavs[case["case"]] = wav_path
            sim_ref = round(speaker_sim(ref_path, wav_path), 4)
            sims[f"ref_vs_{case['case']}"] = sim_ref
            print(f"[qv2] clone control {case['case']}: sim(ref, out)={sim_ref}",
                  flush=True)
        sims["positive_a_vs_positive_b"] = round(
            speaker_sim(ctrl_wavs["positive_a"], ctrl_wavs["positive_b"]), 4)
        sims["positive_a_vs_negative"] = round(
            speaker_sim(ctrl_wavs["positive_a"], ctrl_wavs["negative"]), 4)
        results.append({"section": "clone_similarity",
                        "metric": "resemblyzer cosine",
                        "similarities": sims,
                        "controls": {c["case"]: c["origin"]
                                     for c in manifest["clone_controls"]},
                        "note": "no universal threshold claimed — distributions only"})
        print(f"[qv2] cross sims: a_vs_b={sims['positive_a_vs_positive_b']} "
              f"a_vs_negative={sims['positive_a_vs_negative']}", flush=True)
    finally:
        loader.unload(base)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[qv2] wrote {out}", flush=True)

    print("| language | metric | error | dur s |")
    print("|---|---|---|---|")
    for r in results:
        if r["section"] == "languages":
            print(f"| {r['language']} | {r['metric']} | {r['error_rate']} | {r['audio_seconds']} |")
    print("QUALITY-V2-OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
