# Radeon Reference Closure v0.2 — Design

- **Date:** 2026-09-21
- **Status:** Approved in brainstorming (all 9 design sections); pending spec review
- **Scope:** Tasks 0–9 and 12–20 of the mission brief (10–11 deferred, see below), executed to a gated v0.2.0 release
- **Repo:** `AIwork4me/Qwen3-TTS-ROCm` (local: `/home/amd/Desktop/Qwen3-TTS-ROCm`, branch `main`)
- **Upstream:** `QwenLM/Qwen3-TTS` (pinned clone under gitignored `.upstream/`, currently `022e286`)
- **Source brief:** pasted mission "Qwen3-TTS-ROCm v0.2 → Radeon Reference Closure" (20 tasks, mandatory order)

## North Star

> **Official Qwen3-TTS on AMD Radeon — capability by capability, benchmark by benchmark, with zero upstream patches.**

## Ground truth established at design time (verified 2026-09-21, not from prior notes)

| Fact | Value |
|---|---|
| Project HEAD | `bd8d613` (clean, synced with `origin/main`) |
| Tags | `v0.1.0` only — **0.2.0 is unreleased** despite `pyproject.toml` already reading `0.2.0` |
| Issue #372 | **Already filed** 2026-09-21 by this project (`evidence/upstream-issue-2026-09-21.txt`), quotes the AS-IS failure verbatim, offers both fix options |
| Fine-tuning on ROCm | One E2E smoke exists (`91dfb36`): official workflow, 12 steps, save, reload, sane synthesis — via a disclosed **single-line `sdpa` override** in the gitignored `.upstream` clone, restored pristine afterwards |
| vLLM-Omni | Offline feasibility VERDICT FEASIBLE on gfx1151 (`1b9c197`, isolated `.work-vllm` venv exists) |
| Streaming | Probed 2026-09-21: `qwen-tts` 0.1.1 Python API is blocking-only; row stays 🚫 |
| Benchmarks | Cross-day replication done 2026-09-21 (`181667c`); 1.7B base deviation traced to heat-soak |
| CI | CPU-only GitHub Actions green (run 35526426415); 305/305 tests on validation host |
| GPU inventory | **One machine only**: Radeon 8060S `gfx1151` (unified memory) |
| gh CLI | Authenticated as `AIwork4me`, scopes incl. `repo`/`workflow` |
| Existing program machinery | `docs/superpowers/{specs,plans,reports}` convention proven by the P0 program |

## Resolved decisions (from brainstorming 2026-09-21)

1. **Second architecture (brief Tasks 10–11): DEFERRED to v0.3.** No second Radeon GPU is available (only the gfx1151 host). v0.2.0 ships single-architecture with honest disclosure; hardware matrix records the gap; `radeon-reference-closure-v0.2.md` states "deferred — no second-architecture hardware available at execution time (2026-09-21)".
2. **v0.2.0 release scope: ship on gfx1151 evidence + honest disclosure.** Known Gaps section lists the second architecture (and anything else unproven). The brief's Definition of Done requires claims to match evidence, not multi-architecture coverage.
3. **GPU CI (brief Task 12): prepare-only.** Produce the complete self-hosted-runner workflow YAML + runbook + local validation; mark CI status **BLOCKED ON RUNNER INFRASTRUCTURE**; install no runner; add no green GPU CI badge; make no "GPU CI enabled" claim.
4. **Upstream PR submission (brief Task 8): user-gated.** After the Task-7 verifier PASSes, the orchestrator presents the full diff + PR body + validation summary to the user; `gh pr create` runs only after explicit user confirmation. (Fork target: `AIwork4me/Qwen3-TTS`.)
5. **Execution approach: A — strict serial gating.** Five phases mirroring the brief's mandatory order; every major task gets a fresh independent verifier subagent; no next task while a gate is FAIL/PARTIAL/UNKNOWN/BLOCKED. CPU-only side tasks do NOT jump the queue.

## Process architecture

### Non-negotiable execution rule (from the brief, operationalized)

Every major task: IMPLEMENT/INVESTIGATE → RUN REAL TESTS → ARCHIVE RAW EVIDENCE → LAUNCH FRESH INDEPENDENT SUBAGENT (falsification mandate) → PASS? yes: next task; no: fix current task and re-verify with a NEW verifier. The implementing agent never self-certifies. Verifier instruction (verbatim from brief):

> You are an independent release verifier. Assume the implementing agent may be wrong. Review source, git diff, runtime output, raw evidence, environment metadata and documentation claims. Try to falsify the claimed result. Do not accept "looks correct." Return PASS only when every acceptance criterion is backed by executable or runtime evidence.

Every verifier report carries the 11 fields (verdict, commit/working-tree SHA, acceptance checklist, files reviewed, exact commands, exit codes, runtime evidence, regression tests, claims audit, problems found, justification) and is stored under `docs/superpowers/reports/`.

### Phase structure

| Phase | Brief tasks | Gate |
|---|---|---|
| P1 认知 | 0 ground truth → 1 double repro → 2 root cause | per-task verifier PASS |
| P2 上游修复链 | 3 fix design → 4 implement → 5 regression tests → 6 Rounds A–E → 7 verifier → 8 PR (user-gated) → 9 downstream claims | verifier PASS (Task 7) + user confirmation (Task 8) |
| P3 部署证据 | 12-prep GPU CI (BLOCKED state) → 13 vLLM A/B → 14 A/B verifier | verifier PASS |
| P4 评测基础 | 15 quality benchmark → 16 drift watcher | per-task verifier PASS |
| P5 收口 | 17 claims audit → 18 full regression → 19 final release verifier → 20 tag + GitHub Release | final verifier PASS |

### GPU serialization constraint

One GPU-consuming task at a time (fine-tune E2E, vLLM A/B, benchmarks, regression queue). Before/after each GPU task: record `rocm-smi` state into the evidence file. Benchmark-class comparisons record continuous GPU uptime and use fixed warmup policy (heat-soak lesson from `181667c`).

### Working copies

- `.upstream/Qwen3-TTS` — pristine, pinned SHA recorded by Task 0; `git status --porcelain` must stay empty.
- FIX worktree/branch `fix/finetuning-attn-implementation` — the only modifiable upstream copy, created after root-cause confirmation.
- `.work-finetune/`, `.work-vllm/` — gitignored scratch (already present; recreated fresh per run where the brief demands clean output directories).

## Workstream designs

### P1 — Ground truth, double reproduction, root cause (Tasks 0–2)

- **Task 0:** record both HEADs, installed `qwen-tts`, pyproject version, latest release, test counts, capability matrix state, gfx1151 evidence index, #372 state, and the exact current content of upstream `finetuning/sft_12hz.py`; re-verify the #372 root cause still exists. Archive `evidence/reference-closure-ground-truth-2026-09-21.md`. No modifications.
- **Task 1:** reproduce the AS-IS failure **twice**, two fresh processes, fresh output directories, official prepare + official script, no local edits. Archive `evidence/upstream-372-pristine-failure-run{1,2}.txt` with GPU/gfx/ROCm/torch/Python/qwen-tts/SHA/command/traceback/exit code. Same failure class required in both.
- **Task 2:** answer the brief's 8 root-cause questions; controlled isolation experiment changing ONLY `flash_attention_2`→`sdpa` in the FIX worktree through prepare→load→train→checkpoint→reload→synthesis. A second blocker STOPS the line for investigation — evidence is never forced to match the hypothesis. Archive `evidence/upstream-372-root-cause.md` (causal chain, exact lines, experiment, rejected alternatives, remaining uncertainties).

### P2 — Upstream fix, validation, PR, downstream claims (Tasks 3–9)

- **Task 3 design decision:** A (`--attn_implementation` CLI arg, default `flash_attention_2`) vs B (auto-fallback on missing flash-attn). Evaluate backward compat, predictability, maintainability, cross-platform behavior, silent-CUDA-change risk, upstream style, extensibility. **Recommended: A** (CUDA behavior unchanged, no hidden platform detection, minimal diff) — switch to B only if upstream-structure evidence strongly argues otherwise. Decision documented.
- **Task 4 implementation constraints (prohibitions from brief):** no AMD-specific branches; no `torch.version.hip` checks unless absolutely required; no global `sdpa` default; no vendoring downstream code; no changes to architecture/optimizer/dataset/checkpoint format; no CUDA-default weakening; no dependency on Qwen3-TTS-ROCm.
- **Task 5 regression tests (unit-level, separate from GPU E2E proof):** default remains `flash_attention_2`; explicit `sdpa` selectable; value reaches the model loader; parser backward compatible. No heavy mocking that proves nothing.
- **Task 6 Rounds:** A — 3 fresh Python processes with `--attn_implementation sdpa` loader validation (model loads, backend accepted, no FlashAttention import error, real GPU residency). B — full E2E #1 from clean output dir (prepare → patched official script → real optimizer steps → checkpoint → official-API reload → synthesis → waveform sanity; record steps/loss/batch/precision/walltime/peak-mem/checkpoint/audio metadata). C — full E2E #2, entirely fresh (no reused checkpoint or trainer state). D — default-semantics audit: omitting the flag still means FA2; state "Default configuration semantics preserved; CUDA runtime was not independently tested." E — pristine-vs-fix diff audit (`git diff <upstream-main>...HEAD`): only the configurability change + justified tests/docs; no formatting churn, no unrelated cleanup, no downstream branding.
- **Task 7 verifier challenges A–H:** pristine reproduction real; hard-coded backend is the minimal root cause; upstream default preserved; ROCm fine-tuning executes without runtime source edits; two independent E2E runs passed; patch is generic not AMD-specific; diff is minimal; docs claim no more than evidence. PASS required before any PR action.
- **Task 8 PR:** branch `fix/finetuning-attn-implementation`; commit/PR title `fix(finetuning): make attention implementation configurable`; body per brief template (Summary / Root cause / Validation stack + list / Backward compatibility / Related: Fixes #372); upstream-centered, no downstream promotion. **User gate before `gh pr create`.** Record PR URL/number in downstream evidence. Issue called "resolved" only after merge.
- **Task 9 downstream claims:** while PR is OPEN-unmerged, fine-tuning wording is fixed to: "ROCm E2E execution proven. Upstream portability fix submitted as Qwen3-TTS PR #XXX; current published `qwen-tts==0.1.1` still requires the documented temporary workaround." "Zero-patch fine-tuning fully closed" is forbidden until merge + revalidation of merged pristine upstream (fetch → verify merge SHA → run E2E with zero source edits → archive → then green).
- PR merge does not block v0.2.0; do not wait indefinitely on maintainers.

### P3 — GPU CI preparation and vLLM A/B (Tasks 12–14)

- **Task 12-prep deliverables:** `.github/workflows/gpu-nightly.yml` (self-hosted runner label e.g. `[self-hosted, radeon-gfx1151]`; nightly short GPU suite: upstream parity, tokenizer, 1.7B + 0.6B CustomVoice, VoiceDesign, 1.7B + 0.6B Base clone, reusable prompt, Voice Studio backend workflow; weekly/manual for multilingual full matrix, fine-tune smoke, full benchmark replication); `docs/development/gpu-ci-runbook.md` (install, labels, secrets, maintenance); local YAML + script validation (actionlint or equivalent). Status everywhere: **BLOCKED ON RUNNER INFRASTRUCTURE**. No badge, no runner install, no live claims.
- **Task 13 vLLM A/B protocol:** same GPU, same 1.7B model, same prompt set, same language, same precision where applicable, same warmup policy, same token budget, same text lengths; `.work-vllm` venv vs main `.venv` qwen-tts; A/B alternating order with recorded continuous GPU uptime. Metrics: install/runtime stack, model load time, generated duration, wall time, RTF, peak memory, waveform sanity; batch/concurrency only if genuinely supported. Minimum workloads: CustomVoice 1.7B + one additional reliably-supported path. Outputs: `evidence/vllm-vs-qwen-tts-<date>.json` + `docs/vllm-omni-rocm.md` answering: what works / what doesn't / faster-slower / memory / when to use qwen-tts / when vLLM-Omni / what remains unproven. **No quality claims without a quality metric.**
- **Task 14 verifier audit list:** workload fairness, warmup, token caps, model identity, environment isolation, memory measurement method, RTF calculation, no cherry-picked runs, documentation wording.

### P4 — Quality benchmark foundation and drift watcher (Tasks 15–16)

- **Task 15:** investigation-first. Every candidate metric passes four gates: reproducible on this repo/stack; suitable licensing; feasible on ROCm or CPU; not adopted merely because the upstream paper reports it. Candidates: Voice Clone → speaker-embedding similarity + ASR intelligibility (CER/WER via an open ASR model verified on ROCm); Multilingual → ASR CER/WER + optional language-ID correctness; tokenizer roundtrip → STOI/PESQ where licensing allows (PESQ's ITU patent history is a known trap — if unsuitable, document and skip, never hide). Small fixed benchmark set (fixed prompts, fixed reference wavs, fixed seeds) committed to the repo. Performance and quality benchmarks stay strictly separate; no merged composite score. Rejected metrics archived with reasons.
- **Task 16:** `scripts/check_upstream_drift.py` + committed baseline state file (upstream main SHA, `qwen-tts` PyPI version, checkpoint list, public generate API surface, `finetuning/` script hashes, vLLM instructions fingerprint). Output `UNCHANGED` / `UPSTREAM DRIFT DETECTED` with exact differences; exit 0/1. Never auto-marks new upstream capability Radeon-compatible — drift triggers a "revalidation required" notice. Optional weekly CPU-only scheduled Action. Graceful network-failure reporting.

### P5 — Audit, regression, final verification, release (Tasks 17–20)

- **Task 17:** repo-wide search of the vocabulary list (`validated`, `supported`, `all models`, `all Radeon`, `zero patch(es)`, `fine-tuning`, `vLLM`, `streaming`, `305`, `0.2.0`, `gfx1151`, `gfx1100`); every hit judged against evidence. Enforce the distinction: `zero committed downstream upstream-source patches` ≠ `every official upstream workflow runs unchanged`. Version agreement across README.md, README_CN.md, CHANGELOG.md, docs/, evidence/README.md, pyproject.toml, `__version__`, tests.
- **Task 18 full regression (gfx1151):** CPU suite, GPU suite, official parity, 0.6B, 1.7B, tokenizer, Voice Studio, multilingual, benchmark smoke, fine-tuning execution smoke, vLLM benchmark path; plus lint, package build, clean `git status --porcelain` after committed evidence/docs, zero-vendoring/zero-monkey-patch audit.
- **Task 19 final verifier (fresh subagent):** answers "Is Qwen3-TTS-ROCm v0.2 ready to be described as a Radeon reference without misleading users?" — inspecting #372 work + PR status, capability matrix, gfx1151 evidence, second-GPU gap disclosure, CI status, vLLM comparison, quality benchmark scope, zero-patch wording, release notes, version consistency, final test results. PASS required; no tag on FAIL.
- **Task 20 release:** tag `v0.2.0` on the exact verified commit; `gh release create` with title "Qwen3-TTS-ROCm v0.2.0 — Radeon Reference Closure", North Star narrative, factual summary list (per brief; streaming stays not-claimed). Three prohibitions: no "fixed upstream" unless merged; no "GPU CI live" unless a GPU Actions run succeeded; no second-GPU parity claims. Final deliverable `docs/radeon-reference-closure-v0.2.md` with the brief's 16-section structure.

## Evidence & artifacts map (brief-mandated names)

- `evidence/reference-closure-ground-truth-2026-09-21.md`
- `evidence/upstream-372-pristine-failure-run1.txt`, `...-run2.txt`
- `evidence/upstream-372-root-cause.md`
- Rounds A–E outputs (loader validation ×3, E2E #1/#2 transcripts + metrics, default-semantics audit, diff audit)
- `evidence/vllm-vs-qwen-tts-<date>.json`, `docs/vllm-omni-rocm.md`
- Quality-benchmark evidence + rejection log; drift-watcher baseline + sample runs
- `docs/superpowers/reports/` — one verifier report per gated task (11 fields)
- `docs/radeon-reference-closure-v0.2.md` — final deliverable

## Commit discipline

Logical commits per brief structure (`chore(evidence): …`, `test(upstream): …`, `docs(upstream): …`, `ci: …`, `feat(benchmark): …`, `feat(eval): …`, `chore(upstream): …`, `docs: …`). Evidence-producing stages are never squashed before verification. Upstream fix commits live on the FIX branch only. Downstream `main` receives one commit per verified task. Push cadence: at least after each phase gate (repo is already public and CPU-CI-green; no reason to hoard).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Single-GPU contention (fine-tune / vLLM / benchmarks / regression mutually exclusive) | Strict serialization; `rocm-smi` snapshots around every GPU task; long runs in contiguous windows |
| `github.com` historically unreachable from this host | gh currently authed; on failure fall back to the disclosed clone-bootstrap procedure in `docs/finetuning-rocm.md` and record it |
| Upstream PR response latency | Non-blocking for v0.2.0 (Task 9 wording covers OPEN state); mid-execution merge triggers the 5-step revalidation flow |
| Quality-metric licensing traps (PESQ et al.) | Four-gate investigation; rejections documented, not hidden |
| Benchmark heat-soak drift | Alternating A/B order, recorded continuous uptime, fixed warmup policy |
| Upstream moves `sft_12hz.py` during execution | Task 0 pins SHA; drift check re-run before release; relevant chain re-executed if changed |
| Scope creep | YAGNI: no MOS project, no hardware purchase, no runner install in this program |

## Definition of Done

Every public claim follows: CLAIM → REPRODUCIBLE TEST → REAL RADEON EXECUTION → RAW EVIDENCE → INDEPENDENT VERIFICATION → DOCUMENTATION. For upstream #372 specifically, the PR is submitted only after all 12 brief conditions hold (double pristine repro, causal chain, isolated minimal fix, repeated targeted SDPA loads, two independent E2E passes, checkpoint save/reload, post-finetune synthesis, unchanged FA2 default semantics, clean diff, fresh-verifier PASS) — and the submission itself waits for the user's explicit go. The program is done when Task 20's release exists on the exact verified commit and `docs/radeon-reference-closure-v0.2.md` records every section truthfully, including deferred second-architecture work.

## Out of scope / deferred to v0.3

- Second Radeon architecture validation (brief Tasks 10–11) — blocked on hardware availability.
- Actual GPU CI execution (runner install, first green GPU Actions run) — blocked on runner infrastructure decision.
- Streaming — stays 🚫 until upstream ships a genuine incremental-delivery API.
- Full quality-matrix coverage beyond the minimal foundation (Task 15 scope is deliberately small).
