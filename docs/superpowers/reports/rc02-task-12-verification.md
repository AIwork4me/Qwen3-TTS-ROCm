# Task 12 independent verification — vLLM-Omni vs qwen-tts controlled A/B (gfx1151)

Verifier role: independent release verifier (Task 12 gate = brief Task 14 benchmark
audit). Mandate: try to falsify — cherry-picking, unfair comparison, RTF arithmetic.

## 1. Verdict

**PASS** (4 minor, non-blocking problems recorded in §10; none falsifies an
acceptance criterion).

## 2. Exact commit / working-tree SHA reviewed

- Commit: `b54cc7690fdaabf7fde368b667ac5e990ca4cfa4` (`feat(benchmark): qwen-tts vs
  vLLM-Omni controlled A/B on gfx1151`), parent `13615df`.
- `git rev-parse HEAD` = `b54cc7690fdaabf7fde368b667ac5e990ca4cfa4`;
  `git status --short` empty (tree clean) — working files == commit.
- `origin/main` = `b54cc7690fdaabf7fde368b667ac5e990ca4cfa4`; `git branch -r
  --contains b54cc76` = `origin/main` → **pushed**.
- Evidence records `git_head_at_execution = 13615df` (execution preceded the commit;
  consistent).
- Commit contains exactly 4 files: `docs/vllm-omni-rocm.md` (+195),
  `evidence/README.md` (+2), `evidence/vllm-vs-qwen-tts-2026-09-21.json` (+2535),
  `evidence/vllm-vs-qwen-tts-2026-09-21.txt` (+10772). Nothing else smuggled in.

## 3. Acceptance criteria checklist (brief Step 12.6 / audit list 1–10)

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Same-workload fairness: identical prompts/params both sides, 24 rows/block, same checkpoint, token-cap asymmetry disclosed and immaterial | **PASS** | Raw `ab_qwen_rows_A1/A2.json` + `ab_vllm_rows_B.json`: each 8 prompts × 3 runs, prompt-run pairs complete, no dupes; unique (id, lang, bucket, char_len) tuples identical qwen vs vllm (8/8); both drivers read the same `ab_prompts.json` (speaker Vivian, instruct "", mnt 512); same checkpoint dir both sides — qwen `loader.load("custom-voice")` resolves `models/Qwen3-TTS-12Hz-1.7B-CustomVoice` (src/qwen3_tts_rocm/models.py REPOS), vllm HF_HOME snapshot symlink `readlink -f` → the same directory (verified on disk). Token cap: max audio in any of the 72 rows = 33.52 s (vllm B p4 r1) → 402.2 tokens @12 Hz < 512; qwen max 32.56 s → 390.7 < 512 (and 512 is enforced there). Asymmetry (field not consumed by vllm-omni qwen3_tts pipeline) disclosed in `fairness_notes[0]`, `unsupported_parity`, doc, README. NOTE: `fairness_notes[0]`'s inline bound "longest output <=~20 s" is wrong/stale vs the rows (see §10 P1); the doc/README state the correct 33.5 s ≈ 402 figure and the immateriality conclusion independently verified here from the rows. |
| 2 | Warmup policy equal (1 discarded gen per side per block) | **PASS** | `ab_qwen.py` / `ab_vllm.py`: identical structure — timed load → `reset_peak_memory_stats()` → 1 discarded generation on manifest p1 (same 512 cap) → measured runs; warmup lines present per block in transcript ("warmup wall=… (discarded…)"), all 3 blocks. |
| 3 | Environment isolation real (distinct venvs, HF_HOME only vllm side, no cross-imports) | **PASS** | Recorded `block_commands`: A1/A2 `.venv/bin/python`, B `HF_HOME=… HF_HUB_OFFLINE=1 .work-vllm/venv/bin/python`; `ab_blocks.sh` on disk issues exactly those. Transcript env lines: A blocks torch 2.12.0+rocm7.14.0/HIP 7.14.60850 (no HF_HOME key), B torch 2.12.0+git6bbd260/HIP 7.2.53211 + vllm 0.28.0 + vllm_omni 0.28.0 + HF_HOME. Disk check: `.venv/.../torch-2.12.0+rocm7.14.0.dist-info` vs `.work-vllm/venv/.../torch-2.12.0+git6bbd260.dist-info` (+vllm/vllm_omni only in the latter) — two genuinely distinct torch builds. No cross-imports: ab_qwen imports only repo stack; ab_vllm imports `end2end`+`vllm_omni` from the work venv (block-B warning tracebacks resolve under `.work-vllm/venv/lib`). Disclosed erratum (drivers' venv probe printed `/usr`, symlink-resolution display artifact) confirmed in transcript and correctly disclosed. |
| 4 | Memory method stated and applied consistently (sysfs whole-stack both sides; in-process 0.0 disclosed for vllm) | **PASS** | Identical 1 Hz `VramSampler` class in both drivers (same sysfs paths, same phase logic); `method.memory_measurement` states both methods and the GTT-carries-allocations APU caveat. B summary `peak_alloc_gib_post_load_reset = 0.0` with explicit orchestrator-only scope note. Whole-stack peaks (MiB→GiB recomputed): A1 measured 7119.7 = 6.95 GiB, A2 7220.2 = 7.05 GiB, B 26907.0 = 26.28 GiB → ratio 3.78× — matches doc "7.0/7.1 vs 26.3 GiB, ~3.8×"; GTT 5.6/5.7 vs 25.0 GiB ✓; "~24.5 GiB preallocated at load" = post_load GTT 25104.3 MiB = 24.52 GiB ✓. |
| 5 | RTF = wall/audio recomputed independently (≥6 rows) | **PASS (all 72)** | Recomputed `wall_s/audio_s` for all 72 rows in both the raw row files and the aggregated evidence: every row matches the recorded `rtf` within the tolerance implied by the driver computing RTF from the full-precision wall before rounding wall to 3 dp (5 rows differ by ≤0.0004, all within `0.0005/audio_s`; driver code confirms full-precision computation — the recorded RTF is the more accurate one). Also `audio_s == n_samples/sr` on every row; all sr = 24000. |
| 6 | No cherry-picked runs; 72/72; zero failures | **PASS** | Raw row files (24+24+24) field-identical to the 72 aggregated rows (0 mismatches across side/block/prompt/run/wall/audio/rtf/ok/error/waveform). Transcript `.work-vllm/ab_run_2026-09-21.log` byte-identical to committed `evidence/….txt` (diff clean), contains exactly 24 wall/audio/rtf run lines per block, zero `ok=False`, single continuous run (no earlier/later A/B attempts), all block exits 0, `overall_rc=0`. Aggregator code includes all rows (failures counted, never dropped). |
| 7 | A-B-A structure with drift disclosed | **PASS** | `block_order = [A1 qwen, B vllm, A2 qwen]`; deviation from brief's literal ABBA disclosed in evidence `question`, README, and report. Block medians recomputed from rows: A1 1.2907, B 1.1818, A2 1.3236 (match JSON); drift A1→A2 = +2.55% ≈ claimed +2.6%. Conservative comparison holds: vllm 1.1818 beats qwen's better block A1 1.2907 by 8.4% (doc "~8%"). Continuous window 08:50:57Z→09:09:25Z = 1108 s ✓. Block-boundary snapshots show GTT back to 146 MB idle baseline before each next block (post-A1 577 MB → pre-B 146 MB; post-B 148.5 MB → pre-A2 148.5 MB) — full unload proven. |
| 8 | Doc wording: 7 questions keyed to evidence; no quality claims; guidance follows from numbers; unproven list | **PASS** | `docs/vllm-omni-rocm.md` has the seven numbered questions, each citing JSON paths/rows. Grep for quality/MOS/naturalness/intelligibility/pronunciation/similarity: every hit is a disclaimer (out of scope, sanity-not-quality, Task 14). Verified all quoted numbers against rows: pooled medians 1.3033/1.1818, per-prompt median ranges 1.258–1.344 / 1.138–1.238, ratios 0.8462–0.9858 all <1 (vllm faster 8/8), vllm RTF range 1.1318–1.3574 ("1.13–1.36" ✓), load 4.772/4.903 vs 69.063 (14.1–14.5×, "14×" ✓), warmup 5.752/6.402 vs 16.933, startup deficit 75.1 s ("~75 s" ✓), RTF gap 0.1215 ("~0.12" ✓), break-even 75.1/(0.1215×7.72 median audio) ≈ 80 renders → "roughly 50–100" ✓, memory §4 above, 25 sequential generations on one engine (1 warmup + 24). "When to use" sections each trace to load/memory/RTF numbers; unproven list present (§7). |
| 9 | Aliased evidence: correct alias `custom-voice` = 1.7B | **PASS** | `src/qwen3_tts_rocm/models.py` REPOS: `"custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"`; the 0.6B is `custom-voice-0.6b`; there is NO `custom-voice-1.7b` alias (brief Step 12.2's alias would KeyError). Driver uses `custom-voice`; the deviation is disclosed in evidence env block, doc fairness note 5, and report. |
| 10 | Commit pushed, tree clean, CPU suite green | **PASS** | Push verified (§2). `.venv/bin/python -m pytest -m 'not gpu' -q` → **267 passed, 38 deselected** in 16.04 s (exit 0), exactly the expected counts. |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-12-brief.md`, `task-12-report.md`
- Review diff `review-13615df..b54cc76.diff` (structure: 4 files, headers verified)
- `evidence/vllm-vs-qwen-tts-2026-09-21.json` (full read, all 72 rows)
- `evidence/vllm-vs-qwen-tts-2026-09-21.txt` (targeted: env lines, block
  boundaries, snapshots, run lines, tracebacks; byte-diff vs gitignored log)
- `docs/vllm-omni-rocm.md` (full read + quality-language grep)
- `evidence/README.md` (2 new index rows)
- `.work-vllm/ab_prompts.json`, `ab_qwen.py`, `ab_vllm.py`, `ab_blocks.sh`,
  `ab_aggregate.py`, `ab_qwen_rows_A1.json`, `ab_qwen_rows_A2.json`,
  `ab_vllm_rows_B.json`, `ab_run_2026-09-21.log`, `hf-home/` symlink chain
- `src/qwen3_tts_rocm/models.py` (alias registry)

## 5. Exact commands executed

- `git log --oneline -8`; `git status --short`; `git rev-parse HEAD`
- `git rev-parse origin/main`; `git log origin/main -1`; `git branch -r --contains b54cc76`
- `git show --stat --format='%H %s' b54cc76`
- `grep -n '^diff --git' review-13615df..b54cc76.diff`
- `grep -n 'custom-voice\|alias\|1.7B' src/qwen3_tts_rocm/models.py`
- `ls -la .work-vllm/`; `grep -n 'work-vllm' .gitignore`
- `diff -q .work-vllm/ab_run_2026-09-21.log evidence/vllm-vs-qwen-tts-2026-09-21.txt`
- `find .work-vllm/hf-home -maxdepth 4`; `readlink -f …/snapshots/main`;
  `ls models/Qwen3-TTS-12Hz-1.7B-CustomVoice/`
- Transcript greps: `env=`, `===== BLOCK|A/B RUN`, `vram_used=`, `Traceback`,
  `work-vllm/venv/lib`, `ok=False`, per-block run-line counts via awk/grep
- `ls .venv/…/site-packages` vs `ls .work-vllm/venv/…/site-packages` (torch dist-infos)
- `grep -niE 'quality|MOS|natural|sounds|intelligib|pronounc|similarity' docs/vllm-omni-rocm.md`
- `.venv/bin/python` verification script (stdlib json/statistics) over raw row
  files + evidence: counts/completeness, raw-vs-aggregate field compare, RTF
  recompute (72 rows), audio-vs-n_samples/sr, longest-output tokens, block and
  pooled medians, drift, per-prompt medians/ratios, prompt identity, GiB/ratio
  conversions, doc-claim arithmetic
- `.venv/bin/python -m pytest -m 'not gpu' -q`

## 6. Exit codes

- pytest: **0** (267 passed, 38 deselected).
- All other commands: 0 (diff -q silent = identical; greps that found nothing
  by design — e.g. `ok=False` — were expected-empty and reported as such).

## 7. Runtime evidence inspected

- 72 per-run rows in raw block files AND aggregated evidence (byte-level field
  compare, 0 mismatches): wall, audio, rtf, waveform sanity (finite,
  non-silent, 24 kHz, n_samples/sr consistent) for every run.
- Transcript: per-run wall/audio/rtf lines (24 per block), warmup lines,
  per-block SUMMARY lines, block boundary timestamps/exit codes, rocm-smi +
  sysfs snapshots at every boundary (GTT 146 MB idle baseline restored between
  blocks), stack env lines, block-B vllm_omni path resolution in warnings.
- Row-file summaries: load_s, warmup wall/audio, torch peak after load reset,
  sysfs phase peaks per block.
- HF_HOME hub symlink chain on disk → the exact checkpoint dir the qwen side loads.
- Two venvs on disk with distinct torch builds.

## 8. Regression tests

- CPU suite `.venv/bin/python -m pytest -m 'not gpu' -q`: 267 passed,
  38 deselected — matches the expected counts exactly. (GPU suite not run by
  this verifier per CPU-only verification scope; the A/B itself is the GPU
  runtime evidence, re-verified arithmetically above.)

## 9. Claims audit (attempted falsifications that failed)

- **Cherry-picking**: raw == aggregate, 72/72, zero failures, single continuous
  transcript with no extra attempts, failures-would-be-recorded code path
  (except-branch writes ok=False rows; none present). Not falsified — no cherry-picking found.
- **Unfair comparison**: identical manifest params and prompt set both sides
  (verified as sets and per row); timing window covers only the generation call
  on both sides (sanity + wav write excluded from wall in both drivers —
  verified in code); per-stack default sampling disclosed, RTF normalizes by
  produced audio; A-B-A bracketing plus the conservative "vllm vs qwen's better
  block" check (8.4%) rules out heat-drift as the explanation for the gap.
  Token-cap knob asymmetry disclosed and immaterial from the rows (402 < 512).
  Not falsified.
- **RTF arithmetic**: all 72 recomputed; matches within input-rounding
  tolerance (driver uses full-precision wall); medians, pooled medians, drift
  %, per-prompt ratios, load/warmup/memory GiB conversions, break-even
  arithmetic all reproduce. Not falsified.
- **Alias**: `custom-voice` is the 1.7B CustomVoice; no `custom-voice-1.7b`
  exists — implementer's correction of the brief is right and disclosed.
- **Quality claims**: none present (all quality language is disclaimers).

## 10. Problems found (all minor, non-blocking)

1. **Stale bound in `fairness_notes[0]` (evidence JSON)**: text says "longest
   output <=~20 s audio ~=<=240 tokens at 12 Hz < 512", but the rows show the
   longest output is 33.52 s ≈ 402 tokens. The conclusion (no side truncated,
   < 512) remains true and is stated correctly (33.5 s / 402) in
   `docs/vllm-omni-rocm.md` and `evidence/README.md`; only the JSON note's
   inline figure is wrong. Recommend fixing the string in a follow-up.
2. **Doc §6 "median ~8–9 s render"**: pooled median audio duration is 7.72 s
   (qwen side 8.28 s). Slightly overstated for the pooled set; the derived
   break-even (~80 renders) still sits inside the stated "roughly 50–100" range,
   and the "~1 s per render" saving holds (0.94 s at pooled median).
3. **Rounding-order nit**: `rtf_ratio_vllm_over_qwen` for p2 is computed from
   4-dp-rounded medians (0.8462) vs 0.8461 from exact medians — delta 0.0001,
   immaterial to every conclusion drawn from it.
4. **Erratum (already self-disclosed)**: drivers' `venv` env-line probe printed
   `/usr` for all blocks (symlink resolution). Real isolation was independently
   re-proven here from disk (distinct torch dist-infos) and the transcript env
   lines / vllm_omni paths, so the disclosure is accurate.

## 11. Why PASS is justified

Every acceptance criterion is backed by independent, re-executable evidence:
row-level fairness and completeness verified by recomputation from the raw
gitignored row files (which are field-identical to the committed evidence),
RTF arithmetic verified on all 72 rows (not just 6), all headline numbers in
the doc/README reproduce from the rows (medians, ratios, drift, load, warmup,
memory GiB and 3.8× ratio, break-even arithmetic), isolation is proven from
recorded commands plus on-disk venv contents and the transcript, the alias use
is correct against the registry source, the seven-question doc makes no quality
claims and its guidance follows from the measured numbers, the A-B-A deviation
is disclosed with drift quantified, the commit is pushed with a clean tree, and
the CPU suite passes with exactly the expected counts. The four problems found
are cosmetic/rounding-level inaccuracies (one stale string inside a fairness
note, one approximate median phrasing, one 0.0001 rounding-order delta, one
already-disclosed display erratum) — none changes any measured value,
comparison, or conclusion.
