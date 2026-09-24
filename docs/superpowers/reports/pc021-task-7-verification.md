# Task 7 verification — quality benchmark v2 (v0.2.1 production closure)

- Verifier: independent subagent, repo `/home/amd/Desktop/Qwen3-TTS-ROCm`, branch `main`, HEAD `8cad1b4` (implementation commits `1107014` + `8cad1b4`). No tracked files modified; writes were this report + `/tmp/qv2_verify/` scratch.
- Method: assumed the implementer wrong; re-transcribed the archived WAVs with whisper-small on the gfx1151 GPU myself (identical decode parameters to the harness: `language=whisper_lang, temperature=0.0, beam_size=1, condition_on_previous_text=False`), re-scored CER/WER with the v1 repo helpers via jiwer, recomputed all five resemblyzer clone similarities from the raw WAVs, and audited JSON structure, manifest, docs, script, and WAV archive.

## Verdict

**PASS.** Every number I could recompute from raw files reproduces exactly: all 10 language transcripts match the stored ones verbatim under deterministic decoding, all 10 error rates recompute identically (including German 0.5455, whose arithmetic I confirmed by hand), the probe transcript 'Buzzing' reproduces, and all five clone similarities reproduce with delta 0.000000. Two minor documentation-completeness discrepancies were found (the human-readable .txt omits the Italian row despite claiming "verbatim [qv2] lines"; the clone-control seeds and the three clone-WAV paths are not recorded in the JSON, only recoverable via the committed script's `seed_base+100+j` formula and `wav_dir/clone_{case}.wav` convention). Neither affects any measured value; the JSON is complete and authoritative for the language grid. Also noted: the verification brief's own German decomposition "(6 substitutions + 0)/11" is imprecise — the true alignment is 4 substitutions + 2 deletions + 0 insertions = 6 errors over 11 reference words (0.545455 → 0.5455); the stored rate is correct.

## Check 1 — JSON structure: PASS

Programmatic assertions over `evidence/quality-v2-gfx1151-2026-09-24.json`: 11 results = 10 language rows + 1 `clone_similarity` section; every language row carries all required fields (`intended`, `transcript`, `error_rate`, `audio_seconds`, `seed`, `model`, `metric`, `normalization`, `wav`, plus `language`/`whisper_lang`/`speaker`/`generate_seconds`); meta carries `manifest_version=quality-v2-2026-09-24`, `seed_base=20260926`, `asr{model=small, package=openai-whisper, device=cuda, probe{clip, transcript, load_seconds, transcribe_seconds}}`, `metric_provenance{cer, wer, speaker_sim}`, and a 4-entry `claims_policy` (no composite quality score; no MOS claim; instruction/emotion adherence NOT objectively scored; v1 whisper-tiny scores never compared to v2 whisper-small). Key output: `META-OK: manifest_version=quality-v2-2026-09-24 seed_base=20260926 asr=small/cuda claims=4`, `results=11 (languages=10, clone_sections=1)`, `all 10 language rows carry all 10 required fields`.

## Check 2 — RECOMPUTE FROM RAW FILES (core gate): PASS, 10/10 cells + probe

I re-transcribed ALL 10 archived language WAVs (requirement was ≥3 including German + one CER + one zero-WER) with whisper-small on cuda (`AMD Radeon 8060S`, torch 2.12.0+rocm7.14.0; model loaded in 2.6 s from the already-downloaded `~/.cache/whisper/small.pt` — the implementer's 166.5 s load is labeled "one-time checkpoint download" in the evidence, consistent). Verbatim log:

```
[verify] whisper small loaded on cuda in 2.6s
[verify] PROBE ref_en.wav -> 'Buzzing' in 2.1s (stored: 'Buzzing')
[verify] Chinese    match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
[verify] English    match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
[verify] Japanese   match=True my_err=0.0500 stored_err=0.0500 rescored_on_stored_tx=0.0500
[verify] Korean     match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
[verify] German     match=True my_err=0.5455 stored_err=0.5455 rescored_on_stored_tx=0.5455
[verify] French     match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
[verify] Russian    match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
[verify] Portuguese match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
[verify] Spanish    match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
[verify] Italian    match=True my_err=0.0000 stored_err=0.0000 rescored_on_stored_tx=0.0000
```

`my_err` = recomputed from my own transcript; `rescored_on_stored_tx` = jiwer score of the stored intended/transcript pair through the repo helpers (`scripts/quality_eval.py`: `normalize_cer_text`/`normalize_wer_text` + `cer`/`wer`). Zero divergences: my transcripts equal the stored ones byte-for-byte (deterministic decoding held), and every stored error rate is exactly the jiwer value of its stored strings. Scratch: `/tmp/qv2_verify/recompute.log`, `/tmp/qv2_verify/recompute.py`.

## Check 3 — German WER arithmetic by hand: PASS (stored value correct; brief's decomposition imprecise)

Normalized strings via `normalize_wer_text`: ref `'das wetter ist heute schön lass uns im park spazieren gehen'` (11 words), hyp `'das wetter ist heute schon lessons in park pizzierengain'` (9 words). jiwer `process_words` decomposition: `S=4 D=2 I=0 hits=5` → `total errors = 6; ref words = 11; 6/11 = 0.545455` → rounds to 0.5455, matching the stored `error_rate` exactly. The optimal alignment matches das/wetter/ist/heute/park (5 hits), substitutes schön→schon, lass→lessons, uns→in, spazieren→pizzierengain (4), and deletes im, gehen (2). The verification brief's "(6 substitutions + 0)/11" is therefore wrong on the composition (it is 4 S + 2 D, not 6 S) but right on the total; the stored metric is correct either way.

## Check 4 — Clone-sim recompute from raw WAVs: PASS, exact

`speaker_sim` from `scripts/quality_eval.py` (resemblyzer VoiceEncoder, CPU) on `src/qwen3_tts_rocm/demo/assets/ref_en.wav` and the archived control WAVs:

```
SIM ref_vs_positive_a:      mine=0.6619 stored=0.6619 delta=0.000000
SIM ref_vs_positive_b:      mine=0.6296 stored=0.6296 delta=0.000000
SIM ref_vs_negative:        mine=0.5776 stored=0.5776 delta=0.000000
SIM positive_a_vs_positive_b: mine=0.6794 stored=0.6794 delta=0.000000
SIM positive_a_vs_negative:   mine=0.6001 stored=0.6001 delta=0.000000
```

Positives exceed negatives in every pairing (0.6296/0.6619/0.6794 vs 0.5776/0.6001), and no threshold is claimed anywhere.

## Check 5 — Manifest discipline: PASS with one minor gap

Versioned manifest `tests/data/quality_v2_manifest.json` committed (confirmed via `git ls-files`, in commit `1107014`), `manifest_version` echoed into the JSON meta. Language seeds verified programmatically: each language row's seed equals `seed_base + i` (20260926..20260935). The clone-control seeds follow `seed_base + 100 + j` in the script (`scripts/quality_eval_v2.py`, Section B) — correct formula, but the JSON `clone_similarity` section does not record per-control seeds (recoverable from the committed script + manifest; minor provenance gap, see D2). The 10 manifest languages `[Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian]` match `evidence/multilingual-matrix.json`'s `manifest_languages` (the program's upstream-validated language ground truth, upstream sha recorded there) exactly, in the same order.

## Check 6 — Docs audit: PASS

`docs/quality-v2.md`: results table matches the JSON row-for-row (all 10 error rates and durations, e.g. Russian 3.20 vs JSON 3.2), all five similarity values match, "no universal quality threshold is claimed", "No composite 'quality score' exists by design, and no MOS is claimed anywhere", a dedicated "What these metrics measure — and what they do NOT" section (do-NOT-measure: human perceptual quality, naturalness, expressiveness, prosody, absolute speaker identity, instruction/emotion adherence "not objectively scored"), and the German cell framed as ASR-agreement with explicit "This metric does not say which side erred". README.md row (line 70), README_CN.md row (line 62), and evidence/README.md index row (line 96) all carry consistent numbers (9/10 at 0.00–0.05, German 0.55/0.5455 outlier, positives 0.63–0.68 > negatives 0.58–0.60, no thresholds/composite/MOS). CHANGELOG entry (lines 347–355) matches. No v1 (whisper-tiny) score is quoted in any v2 doc — v1 is mentioned only to say its numbers are never compared; the v1 evidence files (`evidence/quality-benchmark-2026-09-21.*`) were last touched by pre-Task-7 commit `cfef2e7` and are untouched by the Task 7 commits.

## Check 7 — Script audit: PASS

`scripts/quality_eval_v2.py` imports `_read_audio`, `_resample_to_16k`, `cer`, `normalize_cer_text`, `normalize_wer_text`, `speaker_sim`, `wer` from `scripts/quality_eval.py` and never assigns into that module (grep for `quality_eval.` shows only the import block, a docstring mention, and a provenance string — no monkey-patching). Generation is official-API-only: `loader.load("custom-voice").generate_custom_voice(...)` and `loader.load("base").generate_voice_clone(...)`, and `src/qwen3_tts_rocm/loader.py` documents that `load()` returns exactly what the official `qwen_tts.Qwen3TTSModel.from_pretrained` returns ("the native object, never a wrapper"). Code order: the GPU probe (whisper-small load + `ref_en.wav` transcription with recorded transcript) at lines ~109–129 strictly precedes the Section A scoring loop at line ~151. No composite score is computed anywhere — the only occurrences of "composite"/"score" are the claims-policy strings prohibiting them; the end-of-run table prints per-row values only.

## Check 8 — WAV archive: PASS with one gap

13 files in `evidence/quality-v2-wavs/` (10 `lang_*.wav` + 3 `clone_*.wav`), all committed (confirmed via `git ls-files`, commit `8cad1b4`), sizes 130,604–380,204 bytes (within the plausible 100–500 KB band). WAV header durations match every recorded `audio_seconds` exactly (verified for all 10, e.g. `lang_de.wav dur=3.12s (json 3.12)`). Gap: only the 10 language WAVs are referenced by path in the JSON (each row's `wav` field); the 3 clone control WAV paths appear nowhere in the JSON or .txt (they follow the script's `wav_dir/clone_{case}.wav` convention) — see D2.

## Discrepancies (all minor; none affect any measured value)

- **D1 (cosmetic, evidence .txt):** `evidence/quality-v2-gfx1151-2026-09-24.txt` Section A is headed "verbatim [qv2] lines" but omits the Italian row — the file has 14 `[qv2]` lines (1 probe + 9 languages + 4 clone) where 15 belong. The JSON (authoritative) contains the full Italian record, and I independently verified it from the raw WAV: transcript matches byte-for-byte, WER 0.0000, duration 3.20 s. The Spanish line's missing final period in the .txt is NOT a discrepancy — it is the script's `transcript[:60]` truncation, which I reproduced exactly. Suggested fix for a future docs pass: add the missing `[qv2] Italian ...` line.
- **D2 (provenance completeness, clone section):** the JSON `clone_similarity` section records similarities and control origins but neither the per-control seeds (`seed_base+100+j` = 20261026/27/28, formula only in the script) nor the three clone-WAV archive paths. Everything is recoverable from committed artifacts (script + manifest + wav-dir convention) and I recomputed all five similarities from the archived files exactly, so this is a completeness nit rather than a defect.
- **D3 (brief correction, not an implementer fault):** the verification brief's expected German decomposition "(6 substitutions + 0)/11" is inaccurate; the actual jiwer alignment is 4 substitutions + 2 deletions + 0 insertions = 6 errors over 11 reference words. The implementer's stored 0.5455 is correct.
- **D4 (observation):** `docs/quality-v2.md` (end) and the evidence .txt (line 30) pre-reference this verifier report's path at commit time, before the report existed. The reference resolves as of this writing.

## Observations

- Determinism held perfectly: whisper-small greedy/beam-1 decoding reproduced every transcript byte-for-byte weeks-of-runtime apart, and resemblyzer reproduced all five cosines to the fourth decimal (delta 0.000000) — the archived WAVs are exactly the audio that produced the stored metrics.
- The ASR adequacy decision is properly executed per the task rules: whisper-small named exactly, GPU-probed on gfx1151 before any scoring (probe transcript 'Buzzing' reproduced by me in 2.1 s), and no score is compared against the v1 tiny-based run anywhere in the new docs.
- Honesty framing is good: the German outlier is presented as an ASR-agreement measurement with no blame attribution, and single-render cells are labeled "measurements, not statistics".
