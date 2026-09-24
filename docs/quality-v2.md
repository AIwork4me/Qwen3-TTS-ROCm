# Quality benchmark v2 — what these metrics measure (gfx1151)

**Run date:** 2026-09-24 · **Script:** [`scripts/quality_eval_v2.py`](../scripts/quality_eval_v2.py) ·
**Manifest:** [`tests/data/quality_v2_manifest.json`](../tests/data/quality_v2_manifest.json) (versioned `quality-v2-2026-09-24`) ·
**Evidence:** [`evidence/quality-v2-gfx1151-2026-09-24.txt`](../evidence/quality-v2-gfx1151-2026-09-24.txt) ·
machine-readable: [`evidence/quality-v2-gfx1151-2026-09-24.json`](../evidence/quality-v2-gfx1151-2026-09-24.json) ·
generated audio: [`evidence/quality-v2-wavs/`](../evidence/quality-v2-wavs)

The v1 quality benchmark (2026-09-21, [`scripts/quality_eval.py`](../scripts/quality_eval.py))
established the metric machinery on a 5-case zh/en manifest with whisper
**tiny**. v2 extends the *breadth* (all 10 officially supported languages +
clone-similarity controls) and upgrades the ASR leg to whisper **small**
(GPU-probed on gfx1151 before any scoring: load 166.5 s one-time,
transcribe 2.3 s). **Scores are never compared across ASR models** — v1's
tiny-based numbers remain their own dated record and are not quoted here.

## Results (one seeded render per cell; measurements, not statistics)

| Language | Metric | Error rate | Audio s |
|---|---|---|---|
| Chinese | CER | 0.0000 | 2.72 |
| English | WER | 0.0000 | 3.68 |
| Japanese | CER | 0.0500 | 3.04 |
| Korean | CER | 0.0000 | 4.08 |
| German | WER | 0.5455 | 3.12 |
| French | WER | 0.0000 | 4.24 |
| Russian | CER | 0.0000 | 3.20 |
| Portuguese | WER | 0.0000 | 3.52 |
| Spanish | WER | 0.0000 | 4.16 |
| Italian | WER | 0.0000 | 3.20 |

9/10 languages transcribe at 0.00–0.05 error under the pinned ASR model.
The German cell is the honest outlier: whisper-small heard *"Das Wetter ist
heute schon. Lessons in Park Pizzierengain."* — the intended *"schön … lass
uns … spazieren gehen"* partly mis-transcribed. **This metric does not say
which side erred** (TTS pronunciation, ASR weakness, or both); the raw
transcripts and WAVs are archived precisely so a human can listen and a
stronger ASR could be swapped in a future, separately-versioned run.

Clone speaker-similarity controls (1.7B Base, resemblyzer embedding
cosine): same-reference positives `ref→a` **0.6619**, `ref→b` **0.6296**,
`a↔b` **0.6794**; different-speaker negatives `ref→neg` **0.5776**,
`a↔neg` **0.6001**. Positives exceed negatives in every pairing —
directionally consistent with cloning. Margins are small; **no universal
quality threshold is claimed** (none would be justified by this data).

## What these metrics measure — and what they do NOT

**Measure:** agreement between an ASR transcript and the intended text
under a pinned model (whisper-small, temperature 0, beam 1) and pinned
normalizations (CER: punctuation/whitespace-stripped characters — zh/ja/
ko/ru; WER: lowercased, punctuation-stripped words — the rest); embedding
cosine between reference and generated audio under resemblyzer's encoder.

**Do NOT measure:** human perceptual quality, naturalness, expressiveness,
prosody; speaker identity in any absolute sense (cosine has no calibrated
"same person" threshold here); instruction/emotion adherence (**not
objectively scored** — no defensible metric was available); anything about
languages beyond the ten tested or models beyond 1.7B CustomVoice/Base on
this host. **No composite "quality score" exists** by design, and no MOS is
claimed anywhere.

Single seeded render per cell: these are *measurements*, not distributions
(see the RTF benchmark v2 for statistically-treated performance numbers).

## Reproduce

```bash
.venv/bin/python scripts/quality_eval_v2.py \
  --manifest tests/data/quality_v2_manifest.json \
  --wav-dir evidence/quality-v2-wavs \
  --json-out /tmp/quality-v2.json
```

The v0.2.1 Task 7 verifier recomputed a sample of CER/WER values directly
from the archived WAVs
([report](superpowers/reports/pc021-task-7-verification.md)).
