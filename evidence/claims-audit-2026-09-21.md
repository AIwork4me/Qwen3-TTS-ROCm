# Claims audit — 2026-09-21 (v0.2 reference closure, Task 16 / brief 17)

Repo-wide sweep of the public-claims vocabulary so that **every public claim
in the repository is evidence-accurate for v0.2.0**. Downstream HEAD at audit
start: `9a91900` (Task 15 verifier commit).

**Method (Step 16.1).** Grep of

```
validated|supported|all models|all Radeon|zero patch|zero patches|fine-tuning|finetuning|vLLM|streaming|305|0\.2\.0|gfx1151|gfx1100
```

across `README.md`, `README_CN.md`, `CHANGELOG.md`, `docs/**`,
`evidence/README.md`, `src/**`, `tests/**`, `pyproject.toml` — **779 hits
(post-fix count; 778 pre-fix)**, every one judged against evidence. Live test
recount: `.venv/bin/python -m pytest --co -q` → **351 tests collected**
(`-m 'not gpu'` → 313 collected / 38 deselected; `-m gpu` → 38 collected) —
the "305" figure everywhere stale (305 was the Task-0 collection baseline;
Task 14 added 28 CPU tests, Task 15 added 18 more: 267 + 28 + 18 = 313 CPU).

**Fix commit (one docs commit, Step 16.2/16.4):** `docs: claims audit wave —
every public claim evidence-accurate for v0.2.0`.

---

## A. Test-count truing (305 → 351)

Evidence for the new counts: CPU half re-run green by the Task 15 verifier
(`docs/superpowers/reports/rc02-task-15-verification.md`: **313 passed,
38 deselected**, exit 0 — and re-run again during this audit: **313 passed,
38 deselected in 17.61 s**); GPU half carried by the p0 Task 5 verifier
(`docs/superpowers/reports/task-5-verification.md`: `pytest -m gpu -q` →
**38 passed** on the real gfx1151 host — the 38-GPU suite is unchanged since;
Tasks 14/15 added CPU tests only). This follows the repo's established
count-claim convention (CPU and GPU halves recorded as separate full runs,
as the previous 305 claim also did).

| Location | Before | After |
|---|---|---|
| `README.md` verified-results table | 305/305 — 267 CPU + 38 GPU | 351/351 — 313 CPU + 38 GPU |
| `README.md` CI paragraph + Test-suite bullet + re-run block comment | 267 / 305 | 313 / 351 (+ honest CI-skip/skip-era notes, see §B) |
| `README_CN.md` (same four spots, EN/CN lockstep) | 305/305, 267 | 351/351, 313 |
| `CHANGELOG.md` [Unreleased] truing entry | "suite as it stands … 305/305" | dated + explicit supersession pointer to the Task 16 entry (new) |
| `evidence/README.md` `gpu-suite-2026-09-20.txt` row | "current claim 305/305 = 267 CPU + 38 GPU" | "current claim 351/351 = 313 CPU + 38 GPU" |
| `evidence/upstream-372-root-cause.md` Q4(b) | "305-test suite" | "351-test suite — 305 at experiment time, trued by the 2026-09-21 claims audit" |
| `evidence/README.md` ground-truth row | 305 collect-only (frozen Task-0 baseline) | unchanged + supersession note (313 + 38 = 351) |

Left intentionally unchanged (dated records, accurate for their date):
`evidence/reference-closure-ground-truth-2026-09-21.md` (frozen read-only
Task-0 baseline — its 305 *is* the recorded fact), `docs/p0-parity-report.md`
(header declares "records the P0 program at its final commit `8815238`";
README named authoritative), all `docs/superpowers/**` program records, and
the dated CHANGELOG per-task bullets (250 → 282 → 290 progression —
historically accurate, disclosed in-file).

## B. CI-red root cause — the one code/test fix of this wave

**Finding:** the README/README_CN claim "the CPU-only CI matrix runs the
same … CPU tests" was contradicted by live CI: pushes `c07055a`, `cfef2e7`,
`bbc25db` (Task 14 → HEAD at audit start) were **red on all three Python
jobs** — `12 failed, 297 passed, 4 skipped` (runs `35588552252`,
`35590816441`, `35593965878`; gh logs: `ModuleNotFoundError: No module named
'jiwer'`). Cause: Task 14's `tests/test_quality_eval.py` exercises
`scripts/quality_eval.py`'s `cer`/`wer`, which `import jiwer` at call time,
while CI installs only `.[dev]` — and jiwer ships in the optional
`[quality]` extra. The file's own resemblyzer tests already had the
designed guard ("skipped (visibly, with the fix) when the extras group is
not installed, instead of erroring the plain CPU suite"); the jiwer tests
were missing it.

**Fix (minimal, convention-consistent, no dependency-graph change):** the
12 cer/wer tests now carry the same visible-skip guard
(`requires_jiwer`, reason `pip install -e '.[quality]' to run cer/wer
tests`). Verified both ways:

- with the extra installed (validation host): `tests/test_quality_eval.py`
  → **28 passed**; full CPU suite → **313 passed, 38 deselected** (claim
  unchanged);
- with a simulated absence (raising `jiwer` stub first on `PYTHONPATH`):
  **16 passed, 12 skipped**, skip reasons printed — i.e. plain CI
  environments now go green with *visible* skips (12 jiwer + 3 resemblyzer
  + 1 HIP-gated = 16 skips, 297 passed expected per Python job).

README/README_CN now describe the CI state precisely: 313 CPU tests
collected, 1 HIP-gated skip, plus the 15 `[quality]`-extras skips in a plain
`.[dev]` env, with the red-window disclosure and pointer to this file. The
last size-verified green CI run remains the 267-CPU era (run
`35545854932`, `266 passed, 1 skipped`); the first post-fix green run at
the 313 size is the push carrying this commit (to be confirmed on Actions —
not claimed as verified here before it exists).

## C. Version agreement (Step 16.3, verified programmatically)

| Surface | Value |
|---|---|
| `pyproject.toml` `version` | `0.2.0` |
| `src/qwen3_tts_rocm/__init__.py` `__version__` | `0.2.0` |
| `tests/test_scaffold.py` pin | `== "0.2.0"` (passes) |
| `CHANGELOG.md` headings | `[0.2.0]`, `[Unreleased]`, `[0.1.0]` |
| git tags | `v0.1.0` only — **0.2.0 is unreleased** |
| other live-doc `0.2.0` mentions | only `evidence/README.md`'s description of the ground-truth record (accurate) |
| tests' other version strings | synthetic fixtures (`test_check_upstream_drift.py:207`, `test_loader.py:309`) — not claims |

**Result: PASS — all agree on 0.2.0, none claims a 0.2.0 release.** One
convention fix: the CHANGELOG heading `## [0.2.0] - 2026-09-20` dated an
unreleased version; it now reads `## [0.2.0] - Unreleased (opened
2026-09-20; latest tagged release: v0.1.0)`.

## D. PR #373 / fine-tuning wording (re-verified live)

`gh`-equivalent API query on 2026-09-21 (this audit):
`state: open`, `merged: false`, `merged_at: null`, head `48b8644`, base
`022e286`. Every fine-tuning claim surface checked against the required
fixed wording — all conform:

- README capability row + comparison table, README_CN mirrors,
  `docs/finetuning-rocm.md` §"Upstream fix status": "ROCm E2E execution
  proven … submitted as Qwen3-TTS PR #373 (OPEN); current published
  `qwen-tts==0.1.1` still requires the documented temporary workaround" —
  "OPEN, not merged" stated; no claim of merge; merged-upstream path
  explicitly not claimed (finetuning-rocm.md: "there is no merged upstream
  to run").
- The **zero-patch distinction is enforced**: the headline "Zero upstream
  patches" / verified-table row "Patches to upstream `qwen-tts` — 0,
  enforced by a dedicated parity test" mean *zero committed downstream
  upstream-source patches* (pip-RECORD audits + parity test), and the
  fine-tuning row/doc say plainly that the official SFT workflow does NOT
  run unchanged on published 0.1.1 (in-clone temporary workaround, PR #373
  open). No surface conflates the two. No `all models` / `all Radeon`
  overclaim exists anywhere in scope (0 hits outside this audit's own
  vocabulary list).

## E. Deferred-minors ledger — verified, disposition each

1. **`evidence/upstream-372-root-cause.md` citations — STILL APPLIED,
   fixed.** Verified against the pinned upstream clone
   (`.upstream/Qwen3-TTS` @ `022e286b…`, clean): `grep -l flash_attention_2
   examples/*.py` matches **3 of 4** (`test_model_12hz_{base,custom_voice,
   voice_design}.py`; `test_tokenizer_12hz.py` has none) — Q2 now says
   "3 of the 4 `examples/*.py` (`test_tokenizer_12hz.py` contains no such
   literal)". The quoted mapping `attn_impl = "flash_attention_2" if
   args.flash_attn else None` is at `demo.py:606` (the `attn_implementation`
   pass is `demo.py:608-613`, kwarg on line 612) — Q2 now cites `:606` for
   the mapping with the 608–613 pass noted.
2. **Task 7 implementer-report timestamp (15:05:58 vs 15:06:32) — CHECKED,
   no action.** The mismatch lives only in the gitignored implementer
   report (`.superpowers/` is gitignored, verified). The committed evidence
   (`evidence/upstream-372-default-semantics.txt`) is internally consistent
   as one clean run (single header `15:06:32`, single closeout `15:06:43`),
   and the committed verifier report
   (`docs/superpowers/reports/rc02-task-7-verification.md`, P1 note)
   documents the ~34 s attribution discrepancy accurately. Nothing
   committed to fix.
3. **`docs/finetuning-rocm.md:103-105` — STILL APPLIED, fixed.** The
   hard-wrapped fixed-wording sentence is now a single line with backticks
   restored around `qwen-tts==0.1.1` (wording unchanged; renders
   identically, matches README/README_CN style).
4. **`evidence/vllm-vs-qwen-tts-2026-09-21.json` `fairness_notes[0]` stale
   figure — STILL APPLIED, fixed via erratum.** JSON is machine output and
   stays byte-unedited (no-hand-edit policy; precedent: the
   `finetune-smoke-2026-09-20.txt` errata in `evidence/README.md`).
   Verified from the JSON's own `rows`: longest output **33.52 s**
   (vLLM p4; qwen max 32.56 s) ≈ **402 tokens** @12 Hz — the note's
   "<=~20 s ≈ <=240 tokens" understated it; conclusion (no side reached
   any cap; nothing truncated) unchanged. Correction recorded as a new
   errata block in `evidence/README.md` (§"Errata — stale figure inside
   `vllm-vs-qwen-tts-2026-09-21.json` `fairness_notes`"); the companion
   `.txt` is a verbatim transcript and was not appended to.
5. **`docs/vllm-omni-rocm.md` §6 "median ~8–9 s render" — STILL APPLIED,
   fixed.** Recomputed from the JSON: pooled median audio **7.72 s**
   (qwen-side **8.28 s**; 72 rows). §6 now quotes those figures (0.12 ×
   7.72 ≈ 0.9 s per render; break-even 50–100 renders unchanged).

## F. Audit table — every grep hit (term → location → verdict → action)

779 hits post-fix across 70 files; every hit falls in exactly one row below
(counts from the Step 16.1 grep re-run after the fixes; the pre-fix count
was 778 — the audit's own CHANGELOG entry adds hits).

### F.1 Live public-claim surfaces

| File (hits) | Terms found | Verdict | Action |
|---|---|---|---|
| `README.md` (67) | validated/supported/zero patches/fine-tuning/vLLM/streaming/305/267/gfx1151 | 4 stale count claims; rest evidence-backed (zero-patch = committed-patch claim, distinction honored; streaming row 🚫 measured; vLLM 🟡 scoped; fine-tuning ✅-scoped with PR #373 OPEN wording) | Counts trued to 313/351 (3 spots + re-run comment); CI-skip era notes added (§A/§B); no claim weakened or strengthened |
| `README_CN.md` (32) | CN mirrors of the same | same | same four spots, EN/CN lockstep kept |
| `CHANGELOG.md` (50) | dated per-task bullets + 2 live Unreleased statements | dated bullets accurate for their dates; truing entry stale; `0.2.0` heading dated an unreleased version | supersession pointer added; heading marked Unreleased; new Task 16 Verified entry (counts, CI fix, minors, erratum) |
| `docs/finetuning-rocm.md` (12) | fine-tuning/validated/gfx1151/0.1.1 | wording exact (PR #373 OPEN, not merged; workaround scoped); formatting wart only | fixed-wording line unwrapped, backticks restored (minor 3) |
| `docs/vllm-omni-rocm.md` (34) | vLLM/streaming/supported/median figures | all figures re-derived from the JSON during this audit; one stale median | §6 median corrected to 7.72/8.28 s (minor 5); fairness note already carried correct 33.5 s/402 tokens |
| `docs/benchmarks.md` (8) | gfx1151/streaming/validated | factual, methodology-scoped | none |
| `docs/troubleshooting.md` (5) | gfx1151/supported/validated | factual (W3 override guidance; supported = the validated stack's path) | none |
| `docs/p0-parity-report.md` (34) | validated/fine-tuning/vLLM/streaming/290-era counts | dated final report; header declares `8815238` scope + README authoritative; superseded rows marked | none (dated record) |
| `docs/development/gpu-ci-runbook.md` (12), `docs/development/README.md` (1) | validated/gfx1151/fine-tuning | statuses accurate (BLOCKED ON RUNNER INFRASTRUCTURE; validated locally, never executed); GPU node inventory unchanged (38) | none |
| `evidence/README.md` (24) | counts in 2 rows + term mentions in row descriptions | 1 stale "current claim" count; 1 frozen baseline count needing a supersession note; new erratum needed | row trued to 351/313+38; Task-0 row annotated; errata block added for the vLLM JSON fairness figure (minor 4) |
| `evidence/upstream-372-root-cause.md` (ledger item; outside the brief's literal grep scope — 35 vocabulary hits when grepped directly) | 305/fine-tuning/demo.py citations | 1 stale suite count; 2 wrong citations (minor 1) | "351-test suite (305 at experiment time)"; `demo.py:606` + 3-of-4 examples (verified against pinned clone) |
| `pyproject.toml` (4) | version/gfx1151/validated(comment) | agrees (§C) | none |
| `src/qwen3_tts_rocm/__init__.py` (2), `loader.py` (3), `compat.py` (5), `env.py` (5), `testing.py` (7), `demo/backend.py` (7), `demo/ui.py` (3), `cli_demo.py` (3), `models.py` (1) — 36 total | validated-stack scoping in messages/docstrings, gfx1151 facts, `get_supported_*` API names, "23 minutes" guardrail note | all accurate and deliberately scoped (loader error text "validated AMD ROCm wheel stack" etc.); no overclaim | none |
| `tests/**` (59 in 20 files) | fixture paths (`finetuning/sft_12hz.py`), docstrings about scoped wording, `__version__` pin, synthetic version fixtures | accurate; `test_scaffold` pin agrees (§C) | `tests/test_quality_eval.py` — the CI-red fix (§B): visible-skip guard for the 12 jiwer tests |

### F.2 Dated program records (no live claims)

| File class (hits) | Verdict | Action |
|---|---|---|
| `docs/superpowers/reports/**` (19 files, 162 hits) | verifier transcripts — counts/states accurate at recording time (e.g. rc02-task-11's 38/305 collection records); repo convention keeps them verbatim | none |
| `docs/superpowers/plans/**` + `specs/**` (4 files, 179 hits) | program plans/specs incl. the vocabulary list itself; spec's "305/305" is the Task-0 baseline statement | none |
| `docs/development/2026-08-27-*` (3 files, 34 hits) | original 0.1.0 program records (dated) | none |
| `evidence/reference-closure-ground-truth-2026-09-21.md` | frozen read-only Task-0 baseline (305 collect-only is its recorded fact) | none (described with supersession note in `evidence/README.md`) |

### F.3 Machine output (never edited)

| File (hits) | Verdict | Action |
|---|---|---|
| `docs/upstream-baseline.json` (7) | drift-detection baseline, machine-generated, factual | none |
| `tests/data/finetune_manifest.json` (2), `tests/data/quality_benchmark_manifest.json` (1) | authored fixtures | none |
| `src/qwen3_tts_rocm.egg-info/PKG-INFO` (22) | untracked build artifact (`git ls-files` empty), regenerated from pyproject | none (out of committed scope) |

### F.4 Terms with zero in-scope findings

`all models` (0 hits outside this file/CHANGELOG vocabulary lists), `all Radeon` (0), `gfx1100` (0 outside the program docs' vocabulary list) — no such overclaim exists to fix.

## G. Post-fix verification (Step 16.3 gate)

- Vocabulary re-grep: 779 hits; every remaining `305` is a dated/superseded
  context (CHANGELOG dated bullets + supersession notes, frozen Task-0
  baseline description) — no live stale count remains.
- `.venv/bin/python -m pytest -m 'not gpu' -q` → **313 passed, 38
  deselected** (re-run during this audit).
- Simulated jiwer absence → 16 passed + 12 visible skips (CI behavior).
- Version agreement → PASS (§C).
- PR #373 → still open, unmerged (§D).
- No capability-matrix state and no version number was changed by this
  audit; no claim weakened below, or strengthened beyond, its evidence.
