# Radeon Reference Closure v0.2 — closure report (pre-release audit state)

Program: Radeon Reference Closure v0.2 (20 gated tasks, 2026-09-20 → 2026-09-21).
This document is the program's closure record at the **pre-release-audit**
state: it closes Tasks 0–17 and leaves Task 18 Step 18.2 (final independent
release verifier) and Task 19 (user-gated tag + release) to follow. Every
claim below is keyed to its evidence file; no claim is stronger than its
evidence. Where something is not proven, it says so.

Scope note: this is a *closure report*, not marketing copy. The single
question it answers is: what did the v0.2 program prove, on what hardware,
against which upstream state — and what remains open.

---

## 1. North Star

> **Official Qwen3-TTS on AMD Radeon — capability by capability, benchmark by
> benchmark, with zero upstream patches.**

Verbatim from the program charter; restated in [`README.md`](../README.md).

## 2. Final repository SHA

**`afad6b79656821ae5777f6eea3f44a7e9c41a56c`** (`afad6b7`, branch `main`) —
the pre-release-audit SHA this report audits. Recent chain (all on `main`):

| Commit | What it is |
|---|---|
| `afad6b7` | Task 17 verifier PASS (this report's base) |
| `d389786` | Final full regression on gfx1151 (CPU 313/313, GPU 38/38) |
| `20e148e` | Task 16 verifier PASS (claims audit) |
| `89d0015` | Claims-audit wave; CPU CI green at this SHA (§15) |

This document and the `dist/` stale-artifact cleanup (§17, item 7) are
committed on top as Task 18 Step 18.1, so the commit carrying this file has a
newer SHA by construction. Evidence: `git log --oneline` (re-runnable);
regression transcript `evidence/final-regression-2026-09-21.txt` (its
in-transcript start HEAD `20e148e` precedes its own commit `d389786` for the
same reason — transcript-before-commit, never edited after).

## 3. Upstream SHA

**`022e286b98fbec7e1e916cb940cdf532cd9f488e`** — pinned for the whole
program, on both sides:

- Local pristine clone `.upstream/Qwen3-TTS` HEAD = `022e286b…`,
  `status --porcelain` empty (Task 0 §2, re-checked by the Task 8 and Task 17
  verifiers, and again by the Task 17 final zero-patch audit).
- `git ls-remote https://github.com/QwenLM/Qwen3-TTS.git HEAD` = the same
  SHA — **re-verified live during this task (2026-09-21)**.
- The program's baseline record: `evidence/reference-closure-ground-truth-2026-09-21.md`
  §2; the committed drift baseline `docs/upstream-baseline.json`
  (`upstream_sha: 022e286b…`, `pypi_version: 0.1.1`).

Upstream has not moved during the program window. Ongoing movement is watched
by the drift checker (§13).

## 4. Upstream #372

The blocker: official fine-tuning fails out-of-the-box on the validated ROCm
stack. Issue: `QwenLM/Qwen3-TTS#372` (filed 2026-09-21 by this program;
state OPEN, 0 comments at Task 0 capture —
`evidence/upstream-issue-2026-09-21.txt`).

### 4.1 Original reproduction (control leg)

The failure was reproduced **twice from byte-pristine upstream at the pinned
SHA**, in fresh processes/dirs with fresh datasets, before any fix work:
`evidence/upstream-372-pristine-failure-run1.txt` and `-run2.txt`. Both carry
the verbatim failure — `ImportError: FlashAttention2 has been toggled on, but
it cannot be used … the package flash_attn seems to be not installed` — with
`sft-as-is-exit=1` (exit of the `finetuning/sft_12hz.py` process itself), and
both record in-transcript pristine invariants (clone SHA + empty porcelain)
before and after each run. Verified independently: Task 8 Challenge A
(re-hash, cross-transcript byte-compare, class-match against the 2026-09-20
capture).

### 4.2 Root cause

A **loader-configuration choice hard-coded in the fine-tuning script**, not a
model capability: `finetuning/sft_12hz.py:51` at the pinned SHA passes
`attn_implementation="flash_attention_2"` to
`Qwen3TTSModel.from_pretrained`; the wrapper forwards it into transformers,
whose init-time dispatch check (`_flash_attn_2_can_dispatch`,
transformers `modeling_utils.py:2422` raise site in 4.57.3) raises the
ImportError **before any training compute** — and the validated ROCm wheel
stack ships no flash-attn build, so the failure is unconditional on this
stack.

Full causal chain, the 8-question source audit, the one-token controlled
isolation experiment (diff recorded pre-run; treatment leg exit 0 at every
stage; no second blocker), rejected alternative hypotheses, and stated
uncertainties: **`evidence/upstream-372-root-cause.md`** (the citation target
for Tasks 3/8/9). The controlled experiment's key facts: identical invocation
shape, only the one token `flash_attention_2` → `sdpa` differing → dataset
prep, official `prepare_data.py`, **direct `sft_12hz.py` invocation (exit 0,
12 optimizer steps, both checkpoints saved)**, official-API reload, and sane
24 kHz synthesis all completed.

### 4.3 Local fix (and why the repo is still zero-patch)

The fix lives **only on a fork branch**, never in this repository:
`fix/finetuning-attn-implementation` = `0be0026` ("fix(finetuning): make
attention implementation configurable" — adds a `--attn_implementation` CLI
arg) + `48b8644` ("test(finetuning): cover default and explicit attention
implementation") on top of the pinned `022e286`. The **upstream default is
preserved** (`default="flash_attention_2"`), proven at three levels — source,
unit (5/5 on the FIX branch), and runtime (a default-flag load on this
flash-attn-less host fails byte-identically to pristine upstream:
`evidence/upstream-372-default-semantics.txt`). The diff is 2 files,
+10/−5 in `sft_12hz.py` plus a 105-line stdlib-only test file; the Task 8
verifier's grep for vendor-specific patterns (`amd|hip|rocm|radeon|…`) over
the whole diff returned zero hits — the patch is generic, not AMD-specific:
`evidence/upstream-372-diff-audit.txt` (Round E per-hunk audit).

Until that PR merges, this repository's own fine-tuning users follow the
documented temporary workaround (one-token `sdpa` override applied inside a
gitignored clone, restored pristine afterwards) —
[`docs/finetuning-rocm.md`](finetuning-rocm.md).

### 4.4 Repeated validation

- **Round A — targeted loads ×3** (`evidence/upstream-372-round-a-loads{1,2,3}.txt`):
  three fresh processes parsing `--attn_implementation sdpa` through the
  patched script's own parser into the real loader, each exit 0, resolved
  `config._attn_implementation = 'sdpa'`; a GPU-resident supplement leg
  records `param_device=cuda:0`, `mem_alloc_GiB=3.91` in all three.
- **Rounds B+C — two independent E2E trainings**
  (`evidence/upstream-372-e2e-run{1,2}.{txt,json}`): disjoint trees,
  distinct per-run dataset hashes, distinct epoch-1 checkpoint hashes, distinct
  loss sequences — genuinely separate trainings; both completed direct
  `sft_12hz.py --attn_implementation sdpa` runs (exit 0), checkpoint saves,
  official-API reloads (`RELOAD-AND-SYNTHESIS-OK`), and non-silent 24 kHz
  synthesis wavs (re-parsed by the verifier to the last digit).
- **Rounds D+E — default semantics + diff audit**
  (`evidence/upstream-372-default-semantics.txt`,
  `evidence/upstream-372-diff-audit.txt`): default-path behavior unchanged;
  per-hunk accounting of the minimal diff.

### 4.5 Verifier verdict

Independent chain-verifier (fresh, adversarial, no implementation role):
**`docs/superpowers/reports/rc02-task-8-upstream-372-verdict.md` — PASS**:
all eight challenges A–H PASS and **12/12 PR preconditions MET**, every one
backed by evidence the verifier re-derived or re-executed itself (§5 of that
report lists the exact commands). Three minor non-blocking prose observations,
none touching a criterion.

### 4.6 PR

**https://github.com/QwenLM/Qwen3-TTS/pull/373** — "fix(finetuning): make
attention implementation configurable", submitted 2026-09-21 after the
program's user gate (Task 9), head `AIwork4me:fix/finetuning-attn-implementation`,
base `022e286`, closing #372 on merge. Submission facts recorded verbatim in
the appendix of `evidence/upstream-372-root-cause.md`.

### 4.7 PR status (live)

**OPEN** — re-verified live during this task with
`gh pr view 373 --repo QwenLM/Qwen3-TTS --json state`
(`"state":"OPEN"`, 2026-09-21). Issue #372 is therefore **resolved only when
PR #373 merges**; see §9 for the exact downstream wording this forces.

## 5. Capability matrix

Source of truth: the capability matrix in [`README.md`](../README.md) (row
states + evidence pointers maintained there; this section summarizes and does
not override it). Row states at `afad6b7`:

- **✅ E2E validated (Radeon 8060S, real synthesis asserted sane):**
  CustomVoice generation 1.7B and 0.6B; VoiceDesign 1.7B; zero-shot Voice
  Clone (1.7B Base) and the 0.6B Base family; reusable clone prompt
  (save → load → reuse, both sizes); the Design → Clone → Reuse Voice Studio
  flow; the 12Hz tokenizer codec roundtrip; the multilingual matrix (all 10
  officially supported languages × CustomVoice/VoiceDesign + 4 representative
  cross-lingual clone pairs — `evidence/multilingual-matrix.{txt,json}`).
  Per-checkpoint evidence pointers: `evidence/gen-customvoice.txt`,
  `evidence/gen-voicedesign.txt`, `evidence/gen-voiceclone.txt`,
  `evidence/gpu-suite-2026-09-20.txt`, `evidence/voice-workflow-2026-09-20.txt`,
  `evidence/tokenizer-codec.txt`.
- **Fine-tuning: ✅ scoped — execution-only** (prep → 12 optimizer steps →
  checkpoint save → reload → sane synthesis; explicitly no quality or
  convergence claims). The row carries the full #372 chain pointers
  (`evidence/finetune-smoke-2026-09-20.txt` + the §4 files above). Exact
  status wording: §9.
- **Streaming: 🚫 not exposed upstream** — the official `qwen-tts` 0.1.1
  Python API has no incremental-audio path; measured on gfx1151
  (`evidence/streaming-2026-09-21.{txt,json}`: exactly one audio chunk at
  return on every run; TTFB == total wall). Stays 🚫 until upstream ships a
  genuine incremental-delivery API.
- **vLLM-Omni: 🟡 partial, upgraded by the controlled A/B.** The row remains
  🟡 (offline path only — no serving, no quality claims), but its evidence
  base was upgraded from "one feasibility example" to a measured A/B against
  the qwen-tts path (72 timed generations, RTF/load/memory — §11): feasibility
  `evidence/vllm-omni-feasibility-2026-09-21.{txt,json}`; A/B
  [`docs/vllm-omni-rocm.md`](vllm-omni-rocm.md) +
  `evidence/vllm-vs-qwen-tts-2026-09-21.{txt,json}`.
- **Instruction control on 0.6B CustomVoice: 🚫 not exposed upstream** (the
  wrapper silently ignores `instruct` — pinned by tests; Task 0 audit
  `evidence/ground-truth-2026-09-20.md`).

## 6. Hardware matrix

**`gfx1151` only.** Exactly one configuration is independently validated:
AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S (`gfx1151`, Strix Halo class,
94 GB LPDDR5X unified pool), ROCm 7.14.0 wheels
(`torch 2.12.0+rocm7.14.0`), Python 3.12 — [`README.md`](../README.md)
"Verified configuration".

Second architecture: **deferred — no second-architecture hardware available
at execution time (2026-09-21).** Other ROCm-capable AMD GPUs remain
🧪 not-yet-validated (community reports welcome); see §8 and §17.

## 7. gfx1151 results

- **Tests: 351 collected = 313 CPU + 38 real-GPU; 351/351 green on the
  validation host.** Final full regression transcript:
  `evidence/final-regression-2026-09-21.txt` (commit `d389786`); CPU
  313 passed / 38 deselected, GPU suite 38 passed, both exit 0 (details §15).
- **Benchmark RTF references** (median RTF, bf16/sdpa, `max_new_tokens=512`;
  full methodology + per-cell tables:
  [`docs/benchmarks.md`](benchmarks.md)):
  - CustomVoice 1.7B: **1.31 – 1.51** (`evidence/benchmark.json`)
  - VoiceDesign 1.7B: **1.27 – 1.62** (`evidence/benchmark.json`)
  - Voice Clone (1.7B Base): **1.71 – 1.88** (2026-08-27); the 2026-09-21
    replication measured **1.27 – 1.40** under non-heat-soaked conditions
    (`evidence/benchmark-2026-09-21.json`; cross-day tables in
    `docs/benchmarks.md`)
  - 0.6B family: `evidence/benchmark-06b-2026-09-20.json` (+ 2026-09-21
    replication `evidence/benchmark-06b-2026-09-21.json`)
- **Quality metrics — first results** (foundation scope; recorded verbatim,
  not interpreted — `evidence/quality-benchmark-2026-09-21.json`):
  CER 0.0 (zh-short), 0.2 (zh-mid); WER 0.0 (en-short, en-mid); clone case
  CER 2.8 (cross-lingual synthetic-babble reference — the explicit
  "measure, don't interpret" case) with speaker similarity ref↔output
  **0.6728**. Scope: a fixed 5-case manifest, whisper-tiny ASR — smoke-grade
  first numbers, not a quality verdict (§12).
- **Fine-tune smoke** (`evidence/finetune-smoke-2026-09-20.{txt,json}`):
  official SFT workflow executed on gfx1151 — dataset prep → official
  `prepare_data.py` → 12 optimizer steps → both checkpoints saved →
  official-API reload → finite non-silent 24 kHz synthesis; in-process peak
  18.02 GiB. Execution-only scope (no quality claims); superseded in strength
  by the §4 FIX-branch E2E pair but still the **published-package path**
  evidence.

## 8. Second Radeon architecture results

**Deferred — no second-architecture hardware available at execution time
(2026-09-21).** No second-architecture claim is made anywhere in this
repository; the compatibility table keeps all other GPUs at
🧪 not-yet-validated and asks for measured community reports. See §6 and
Known Gaps (§17, item 1).

## 9. Fine-tuning status

The fixed wording (also carried verbatim by the README row and the
comparison table):

> **ROCm E2E execution proven. Upstream portability fix submitted as
> Qwen3-TTS PR #373 (OPEN) — validated by a pristine double reproduction of
> the failure, a minimal-fix controlled isolation, 3× targeted loader
> validations, two independent E2E runs from the fix branch, preserved
> default `flash_attention_2` semantics, and an independent chain-verifier
> PASS. Until it merges, current published `qwen-tts==0.1.1` still requires
> the documented temporary workaround.**

Decomposed:

- **E2E execution proven** — on gfx1151, through the official
  `finetuning/` workflow, at smoke scale (12 samples / 12 optimizer steps),
  including checkpoint save, official-API reload, and sane synthesis. Proof
  stack: §4.4 (and the 2026-09-20 smoke, §7). Execution scope only: no
  convergence, speaker-similarity-after-training, or long-run claims.
- **PR #373 submitted** — URL and live state in §4.6/§4.7.
- **Published `qwen-tts==0.1.1` still requires the documented temporary
  workaround** — the PyPI package and the upstream repo at the pinned SHA
  still hard-code `flash_attention_2` at `sft_12hz.py:51`; until a merged fix
  ships in a release, users of this repo follow
  [`docs/finetuning-rocm.md`](finetuning-rocm.md) (one-token `sdpa` override
  inside a gitignored clone, restored pristine).

## 10. GPU CI status

**BLOCKED ON RUNNER INFRASTRUCTURE.** What exists (Task 11, verifier PASS):
the self-hosted workflow `.github/workflows/gpu-nightly.yml` (nightly short
suite = 32/38 GPU test nodes via 3 disjoint `-k` slices, plus manual
full-weekly matrix) and the activation runbook
[`docs/development/gpu-ci-runbook.md`](development/gpu-ci-runbook.md).
Filter-selection proofs: `evidence/gpu-ci-prep-validation.txt`.

What does not: the workflow has **never executed on a GitHub Actions
runner** — no runner is registered — and there is deliberately **no GPU CI
badge**. No live-runner claim is made anywhere. All GPU test evidence in this
repository comes from the validation host itself (§7/§15). This remains a
known gap (§17, item 2).

## 11. vLLM-Omni comparison

Headline (controlled A/B on gfx1151, same checkpoint/workload/GPU, A-B-A
block design over one continuous 1108 s window, 72 timed generations, zero
failures — [`docs/vllm-omni-rocm.md`](vllm-omni-rocm.md); raw rows
`evidence/vllm-vs-qwen-tts-2026-09-21.json`, transcript
`evidence/vllm-vs-qwen-tts-2026-09-21.txt`):

| | qwen-tts (loader path) | vLLM-Omni 0.28.0+rocm723 |
|---|---|---|
| Median RTF (pooled) | 1.303 (48 runs) | **1.182** (24 runs) — **~9% faster generation** |
| Model load | **4.8–4.9 s** | 69.1 s — **14× slower load** |
| Whole-stack measured-phase peak (sysfs vram+gtt) | **7.0 / 7.1 GiB** | 26.3 GiB — **≈3.8× memory (3.78:1 vs the A1 block)** |

Practical reading (measured, not speculative): vLLM-Omni wins steady-state
sequential rendering (1.4–15% per prompt, largest on medium/long); qwen-tts
wins one-shot time-to-audio (the engine is still loading when qwen-tts has
delivered audio) and memory-constrained hosts (a 16–32 GiB machine likely
does not fit the vLLM-Omni side at all). Scope guards: timing/memory only —
**no quality claim**, no serving/batching/streaming/concurrency claims, and
the fairness notes (token-cap, sampling defaults) are in the doc. The
capability-matrix consequence is in §5.

## 12. Quality benchmark status

**Foundation scope, first results** (Task 14; fix round + re-verification):

- **Adopted metrics:** CER and WER (jiwer 4.0.0, char/word normalization
  pinned; ASR via openai-whisper `tiny`, greedy, language pinned —
  `scripts/quality_eval.py`) and **speaker similarity** (resemblyzer 0.1.4
  VoiceEncoder cosine). First results in §7. Test coverage: 28 CPU tests
  (`tests/test_quality_eval.py`) incl. hand-computed CER/WER pins and
  manifest-schema validation.
- **STOI — not adopted for this manifest.** pystoi 0.4.1 was
  **ACCEPTed at the license/feasibility gates (MIT; synthetic-pair smoke
  pass) but is scope-limited**: STOI needs a clean-reference/degraded-signal
  pair and the fixed manifest is text-conditioned (no clean reference
  waveform exists), so it is not computed for these cases; it ships in the
  `[quality]` extras for future signal-pair (tokenizer-roundtrip) work.
  Exact status: Task 14 report §Step 14.3 + the JSON's `omitted_metrics.stoi`.
- **pesq — REJECTED at the license gate.** The PyPI wrapper is MIT-labelled
  but bundles the ITU-T P.862 reference C implementation whose own IPR
  notice names owners BT→Psytechnics and KPN→OPTICOM and forbids
  modification/commercial use/distribution — incompatible with OSS
  distribution. Never installed; the **verbatim IPR notice is archived** in
  `evidence/quality-benchmark-2026-09-21.txt` (that verbatim-capture
  requirement was the round-1 verifier's one blocking finding, addressed and
  re-verified — see §16).
- **Fixed 5-case manifest** (`tests/data/quality_benchmark_manifest.json`,
  seed 20260921): zh-short, zh-mid, en-short, en-mid, clone-ref — a fixed,
  seeded, reproducible set; deliberately **smoke-grade breadth**, not a
  quality verdict.
- **Performance/quality separation enforced:** the quality JSON carries no
  RTF/wall/composite numbers, and the benchmark evidence carries no quality
  numbers; the Task 14 self-check pinned this (24/24 criteria).

## 13. Upstream drift protection

In place since Task 15 (verifier PASS; tri-state live-verified):

- **`scripts/check_upstream_drift.py`** — compares the committed baseline
  `docs/upstream-baseline.json` (upstream main SHA `022e286…`, PyPI
  `qwen-tts` version `0.1.1`, repo tree blob hashes incl.
  `finetuning/sft_12hz.py`, README, and the installed package's four
  generate-API signatures) against live upstream state, with a pinned
  **tri-state exit convention**: `0` UNCHANGED · `1` UPSTREAM DRIFT
  DETECTED (revalidation required — the script never auto-marks new upstream
  capability as Radeon-compatible) · `2` NETWORK/COLLECTION ERROR (drift
  could not be determined). Stdlib-only networking; each HTTP GET retries
  exactly once (this host's known github.com:443 flakiness).
- **Weekly Action** `.github/workflows/upstream-drift.yml` — Mondays 03:17
  UTC + manual dispatch, CPU-only runner, fails the job on exit 1 or 2 so a
  human decides.
- Pinned by `tests/test_check_upstream_drift.py`; baseline drift from
  today's live check: none (§3).

## 14. Zero-patch audit

- **Downstream commits contain zero upstream-source patches.** `main`'s
  history adds no modification to any `qwen_tts` source: the dependency is
  the published `qwen-tts==0.1.1` artifact as-is
  ([`pyproject.toml`](../pyproject.toml)); the fine-tuning fix exists only as
  the fork-branch PR (§4.3). The Task 17 final audit re-confirmed: all 31
  non-loader `qwen_tts` references in `src/` are docstrings/comments or
  runtime imports of the installed package (no monkey-patching, no
  vendoring); zero tracked files under `.upstream/`;
  `evidence/final-regression-2026-09-21.txt` (Step 17.2 block).
- **Upstream clones pristine.** `.upstream/Qwen3-TTS` at `022e286…`,
  porcelain-clean; `.upstream/Qwen3-TTS-fix` at `48b8644`, porcelain-clean —
  re-audited by the Task 17 verifier at program close.
- **Enforcement:** the parity test
  [`tests/test_official_demo_parity.py`](../tests/test_official_demo_parity.py)
  builds the stock, unmodified upstream `build_demo()` around a model loaded
  by this repo's loader and drives the demo's own handler closures (incl. a
  real GPU synthesis) — green transcript `evidence/official-parity.txt`.
  Policy is written into [`CONTRIBUTING.md`](../CONTRIBUTING.md) and
  [`NOTICE`](../NOTICE). Package-level corroboration: the built 0.2.0
  wheel/sdist contain zero `qwen_tts/` entries (Task 17 suite 7/7).
- **The distinction, stated plainly:** *zero committed patches* means no
  upstream source is modified in this repository's tree or packages — it
  does **not** mean every official workflow runs unchanged on this stack.
  The live example is fine-tuning: until PR #373 merges, the official
  fine-tuning script needs the documented one-token workaround (§9). The
  zero-patch posture is exactly why the fix went upstream as a PR instead of
  a local patch.

## 15. Test summary

- **Collected: 351** (313 CPU + 38 GPU) on the validation host.
- **CPU: 313/313 green** — final regression
  `evidence/final-regression-2026-09-21.txt` (`pytest -m 'not gpu'`:
  313 passed, 38 deselected), at the `d389786` regression commit.
- **GPU: 38/38 green** — same transcript (`pytest -m gpu`: 38 passed),
  plus a re-run representative slice by the Task 17 verifier.
- **Regression suite 7/7 green** (same transcript): CPU, GPU, benchmark
  smoke, fine-tuning execution smoke (documented workaround path), vLLM A/B
  path re-run, lint (`ruff check .` — the CI gate), package build (0.2.0
  wheel+sdist, zero vendored `qwen_tts`).
- **CI (CPU): green** on `89d0015` — GitHub Actions run
  [35597375251](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/runs/35597375251),
  conclusion `success` (status re-verified live during this task). CI runs
  the CPU matrix (Python 3.10/3.11/3.12; 1 HIP-gated skip without an AMD
  GPU; 15 visible `[quality]`-extras skips in the plain `.[dev]`
  environment). GPU CI does not run in CI (§10).

## 16. Independent verification reports

Index of the program's fresh-verifier reports
(`docs/superpowers/reports/`), each an independent adversarial pass:

| Report | Scope | Verdict |
|---|---|---|
| `rc02-task-1-verification.md` | Pristine double reproduction of #372 | **PASS** |
| `rc02-task-2-verification.md` | Root cause + controlled isolation experiment | **PASS** |
| `rc02-task-3-verification.md` | Minimal fix as a commit on the FIX branch | **PASS** |
| `rc02-task-4-verification.md` | Upstream unit regression tests | **PASS** |
| `rc02-task-5-verification.md` | Round A targeted loader validations ×3 | **PASS** |
| `rc02-task-6-verification.md` | Rounds B+C two independent E2E trainings | **PASS** |
| `rc02-task-7-verification.md` | Rounds D+E default semantics + diff audit | **PASS** (1 minor prose nit) |
| `rc02-task-8-upstream-372-verdict.md` | Full #372 chain gate (challenges A–H, 12/12 PR preconditions) | **PASS** |
| `rc02-task-10-verification.md` | Downstream fine-tuning claim cleanup | **PASS** |
| `rc02-task-11-verification.md` | GPU CI preparation (blocked-state honesty) | **PASS** |
| `rc02-task-12-verification.md` | vLLM-Omni vs qwen-tts controlled A/B | **PASS** (4 minor, non-blocking) |
| `rc02-task-14-verification.md` | Quality benchmark foundation, round 1 | **FAIL** — 4 findings (1 blocking: pesq IPR verbatim capture) |
| `rc02-task-14-reverification.md` | Quality benchmark, post-fix re-review | **ALL 4 ADDRESSED** (PASS-equivalent close-out) |
| `rc02-task-15-verification.md` | Upstream drift watcher (tri-state) | **PASS** |
| `rc02-task-16-verification.md` | Claims-audit wave (public claims vs evidence) | **PASS** |
| `rc02-task-17-verification.md` | Final full regression + zero-patch audit | **PASS** |

No `rc02-task-9` report exists by design: Task 9 (PR submission) was
user-gated and recorded in `evidence/upstream-372-root-cause.md`'s appendix
+ the program journal; Task 13 was a folded marker absorbed into Task 12.
The final release verifier (Task 18 Step 18.2) is dispatched after this
document; its verdict will be
`docs/superpowers/reports/rc02-task-19-final-verdict.md`.

## 17. Remaining known gaps

Honest-open list at closure; each is disclosed wherever it is relevant in
the public docs:

1. **Second Radeon architecture** — deferred; no second-architecture
   hardware available at execution time (2026-09-21). (§6, §8.)
2. **GPU CI execution** — workflow + runbook prepared and validated, but
   BLOCKED ON RUNNER INFRASTRUCTURE; never executed on a runner; no badge.
   (§10.)
3. **True streaming** — upstream `qwen-tts` 0.1.1 exposes no incremental
   API; measured (single chunk at return, TTFB == wall). Stays 🚫 until
   upstream ships one; the five re-measurements are pre-specified in the
   README roadmap. (§5.)
4. **Fine-tuning zero-patch closure** — needs PR #373 to merge and a
   revalidation against the merged release; until then the published
   package requires the documented temporary workaround. (§4, §9.)
5. **Quality-metrics breadth** — fixed 5-case manifest, whisper-tiny ASR;
   first results are smoke-grade, explicitly not a quality verdict; STOI
   scope-limited, pesq rejected on IPR grounds. (§12.)
6. **`ruff format --check` not enforced** — exits 1 (47 files would
   reformat); the repo's CI gate is `ruff check .` alone, which passes.
   Recorded informationally in the final regression transcript so nobody is
   surprised.
7. **`dist/` stale 0.1.0 artifacts** — Aug-29-built 0.1.0 wheel+sdist sat
   alongside the 0.2.0 pair; removed in this task's commit (untracked,
   gitignored — plain delete) so the pre-tag state is unambiguous: `dist/`
   now holds only the 2026-09-21 `qwen3_tts_rocm-0.2.0` pair (Task 17
   suite-7 build).
8. **Editable-install metadata lag** — the repo `.venv`'s editable install
   reports `qwen3-tts-rocm 0.1.0` while `pyproject.toml` is `0.2.0`
   (code is live via the editable path; only dist-info is stale; the built
   artifacts correctly carry 0.2.0). Harmless to all suites; a
   `pip install -e .` refresh trues it up. (Task 17 report, notes.)
9. **Claims-audit internal-record tally nits** — two §F.2 tallies in
   `evidence/claims-audit-2026-09-21.md` (stated 162 hits vs 159 sum;
   stated 4 files vs 5) were left uncorrected per the dated-record
   convention and are disclosed in `rc02-task-16-verification.md`; all
   live-surface counts reproduce exactly. Internal records only; no public
   claim affected. (Final-verifier finding 1.)

## 18. v0.2.0 release URL

**https://github.com/AIwork4me/Qwen3-TTS-ROCm/releases/tag/v0.2.0** — tag `v0.2.0` on commit `bec162c` (release housekeeping: §17 item 9 + the claims-audit evidence-index row; CPU suite re-verified green there: 313 passed / 38 deselected), created 2026-09-22 after the final independent release verifier returned PASS and the user gate confirmed. The tag and release were created only after
the final independent release verifier (Task 18 Step 18.2) returned PASS.
