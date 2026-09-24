# gfx1151 Production Closure — v0.2.1

**Program:** Qwen3-TTS-ROCm v0.2.1 — gfx1151 Production Closure (plan:
[`docs/superpowers/plans/2026-09-24-gfx1151-production-closure-v0.2.1.md`](superpowers/plans/2026-09-24-gfx1151-production-closure-v0.2.1.md))
· **Executed:** 2026-09-24 → 2026-09-25 (one continuous session) ·
**North Star:** official Qwen3-TTS on AMD Radeon — capability by capability,
benchmark by benchmark, with zero upstream patches.

## 1. Baseline SHA

`cd46a8152a56e0d475512ba8e86c80b46ae0dbe2` (v0.2.0-era HEAD; frozen in
[`evidence/gfx1151-production-closure-baseline-2026-09-24.txt`](../evidence/gfx1151-production-closure-baseline-2026-09-24.txt):
313 CPU + 38 GPU tests, capability matrix as-was, upstream #372/#373 OPEN,
vllm-omni main 3d43571).

## 2. Final SHA

`72676c39dfd9e8c81c6b64ab49451c6265274825` + this closure commit (all
program commits are on `main`, pushed; per-commit evidence messages in
`git log cd46a81..HEAD`).

## 3. Task-by-task results (each gated by an independent verifier)

| # | Task | Result | Verifier verdict (report) |
|---|---|---|---|
| 0 | Baseline freeze | evidence archived, no claims changed | PASS after fix round 1 ([pc021-task-0](superpowers/reports/pc021-task-0-verification.md)) |
| 1 | First green full-weekly GPU CI | run 35964051504, 40/40 steps green, 38 nodes 314 s, verify_gpu OK, benchmark 12 cells RTF 1.27–1.40, artifact persisted | PASS first round ([pc021-task-1](superpowers/reports/pc021-task-1-verification.md)) |
| 2 | Docker real-GPU E2E | no-cache build at 488a028; in-container 16/16 checks incl. 0.6B synthesis (3.84 s @ 24 kHz) + 4-node GPU pytest slice | PASS after fix round 1 (17/17→16/16 count erratum) ([pc021-task-2](superpowers/reports/pc021-task-2-verification.md)) |
| 3 | Official API batch inference | 5 families × B∈{1,2,4,8}: 20/20 green, max validated B=8 each; 2 new reusable-prompt batch GPU tests | PASS after fix round 1 (runbook table) ([pc021-task-3](superpowers/reports/pc021-task-3-verification.md)) |
| 4 | Benchmark v2 | 5 aliases × 4 cells × n=5 seeded runs; phases separated; 480/480 stored stats recomputed exact by verifier | PASS first round ([pc021-task-4](superpowers/reports/pc021-task-4-verification.md)) |
| 5 | Long-text + soak | ladder 8/8 tiers (≤892 chars / 58 s audio, natural EOS); 60-min soak 815/815 zero failures (RSS +0.02 GiB); 10/10 recycle | PASS after fix round 1 (8192-default label, drift disclosure) ([pc021-task-5](superpowers/reports/pc021-task-5-verification.md)) |
| 6 | 0.6B fine-tuning E2E | ×2 independent runs green (12 steps → fingerprinted checkpoints → official reload → sane synthesis); **second upstream defect root-caused** (sft_12hz.py omits mandatory text_projection — shape error by construction on 0.6B) | PASS after fix round 1 ([pc021-task-6](superpowers/reports/pc021-task-6-verification.md)) |
| 7 | Quality benchmark v2 | 10 languages via whisper-small: 9/10 at 0.00–0.05, German 0.5455 honest outlier; clone-sim controls (positives > negatives, no thresholds) | PASS first round (all transcripts/scores reproduced from raw WAVs) ([pc021-task-7](superpowers/reports/pc021-task-7-verification.md)) |
| 8 | vLLM-Omni current truth | current stack (vllm 0.30.0+rocm723 + omni 0.30.0rc1), upstream end2end.py @ 7e5897b: all three 1.7B families offline green | PASS first round (rerun WAV sha-identical) ([pc021-task-8](superpowers/reports/pc021-task-8-verification.md)) |
| 9 | Online serving + true streaming | 9A/9B/9C serving green (Base inline clone after 3 honest iterations); **true streaming measured client-side** (TTFA 0.249 s / 0.222 s); upstream WS client upstream-broken (verbatim) | PASS after fix round 1 ([pc021-task-9](superpowers/reports/pc021-task-9-verification.md)) |
| 10 | Concurrency characterization | default collapses past c=2; stage-1 max_num_seqs=1 stable through c=8 (TTFA 0.74 s, 34.9 req/min, 1.50 audio-s/wall-s); zero failures/OOM | PASS first round ([pc021-task-10](superpowers/reports/pc021-task-10-verification.md)) |

## 4. Exact tested hardware / software

AMD Ryzen AI Max+ PRO 395 · Radeon 8060S (`gfx1151`, unified memory) ·
ROCm 7.14.0 (HIP 7.14.60850) · torch 2.12.0+rocm7.14.0 · Python 3.12.3 ·
official `qwen-tts 0.1.1` (primary path, unmodified) · vLLM leg: vllm
0.30.0+rocm723 + vllm-omni 0.30.0rc1 + onnxruntime-rocm 1.22.2.post3 in an
isolated gitignored venv. Final regression at closure: **317 CPU + 40 GPU
= 357/357 green, ruff clean**
([transcript](../evidence/final-regression-v0.2.1-2026-09-25.txt)).

## 5. Capability matrix — before vs after

**Before (baseline, 2026-09-24):** 13 matrix rows — generation/cloning/
tokenizer/multilingual E2E; fine-tuning 1.7B execution-only; vLLM-Omni 🟡
offline-feasibility-only (one example, 0.28.0 snapshot); streaming 🚫
(qwen-tts API measured non-incremental).

**After (closure):** 19 rows — additions/upgrades, all evidence-linked:

| New/upgraded capability | Status |
|---|---|
| First green full-weekly GPU CI (all 38-then-40 nodes + benchmark, artifact-persisted) | ✅ run 35964051504 |
| Docker GPU-runtime E2E (was build-validated) | ✅ 16/16 in-container incl. 0.6B synthesis |
| Official API batch inference (list-of-texts, 5 families) | ✅ B≤8 all green (this host/config) |
| Benchmark v2 (n=5 seeded, phases separated, all 5 aliases) | ✅ RTF medians 1.04–1.52, stdev ≤0.11 |
| Long-text + sustained stability | ✅ 8/8 tiers to 58 s audio; 815/815 in 60 min; 10/10 recycle |
| Fine-tuning now covers BOTH 1.7B and 0.6B (execution-only, unchanged claim discipline) | ✅ ×2 independent 0.6B runs |
| Quality v2: 10-language ASR content correctness + clone-sim controls | ✅ measured (no thresholds/MOS) |
| vLLM-Omni: offline on CURRENT stack, all three 1.7B families | ✅ (was 0.28.0/CustomVoice-only) |
| vLLM-Omni: OpenAI-compatible serving (speech/voices, 3 task types) | ✅ E2E |
| vLLM-Omni: TRUE streaming (HTTP PCM) | ✅ measured, TTFA 0.22–0.25 s — while the qwen-tts Python API row stays honestly 🚫 non-incremental |
| vLLM-Omni: concurrency guidance | ✅ measured (default vs stage-1 seq=1) |

## 6. Remaining known gaps (deliberately not claimed)

- **Upstream WebSocket streaming client** broken against the current
  vllm-omni server build (client/server route+protocol drift; four verbatim
  attempts archived). HTTP PCM is the working streaming path.
- **0.6B vLLM variants** untested (program deprioritized them by design).
- **Upstream fine-tuning blockers persist**: PR #373 OPEN; the 0.6B
  text-projection omission (issue drafted in
  `.work-upstream/issue-draft-sft-06b-text-projection.md`, **not filed —
  user-gated**). Workarounds stay confined to gitignored clones, disclosed.
- **Base serving concurrency matrix** not run (small-matrix ruling);
  long-prompt concurrency uncharacterized.
- **Memory observations on the vLLM leg**: dedicated-VRAM is not a
  utilization signal on this APU (GTT pool); server-RSS sampling invalid in
  Task 10 — neither claimed.
- Soak latency drifted ~15 % upward across the hour (zero failures, cause
  uncharacterized); no leak-free claim anywhere.
- Quality: single seeded render per language cell (measurements, not
  statistics); German WER 0.5455 outlier recorded unblamed.

## 7. No-overclaim audit

Final-regression-day grep audit across README/README_CN/docs/evidence
index/CHANGELOG for the program's vocabulary (validated / supported /
streaming / batch / Docker / quality / fine-tuning / vLLM / production /
gfx1151 / Radeon): every "validated" instance traces to evidence; no
leak-free / maximum-length / universal-best / MOS / composite-score claims
exist (only their explicit negations); the qwen-tts-API vs vLLM-Omni
streaming separation (Rule 3) holds in every public surface; upstream
facts are never restated as local Radeon claims. Two parity-table rows
(EN/CN fine-tuning) were brought up to the 1.7B+0.6B scope during this
audit. Program Rule 1 status flips in this cycle: 11 matrix upgrades, each
gated on verbatim transcripts/machine JSON and an independent verifier.

## 8. Release readiness verdict

**READY for v0.2.1** pending the final release-verification subagent's
independent PASS (report:
[`superpowers/reports/pc021-final-release-verification.md`](superpowers/reports/pc021-final-release-verification.md))
and the user's explicit approval to tag/publish (which this program does
NOT take on its own). Version bump + release notes are prepared only after
that PASS.
