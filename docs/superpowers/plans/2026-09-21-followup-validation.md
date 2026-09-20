# Follow-up Validation Program — Implementation Plan (A–E)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Sequential gated steps: implementer → independent verifier subagent → PASS → one commit → next step. Single push after all steps PASS.

**Goal:** Close the last unexecuted claim (first real CI on the P0 push) and walk the README roadmap ladder one rung at a time — upstream flash-attn issue, vLLM-Omni offline feasibility verdict, true-streaming measurements, cross-day benchmark replication.

**Spec/context:** `docs/superpowers/specs/2026-09-20-p0-capability-parity-design.md` (P0 program, completed 5/5 PASS, pushed at `8815238`); README "Roadmap (not yet validated)" section.

## Global Constraints (unchanged from P0 program, binding)

- Zero upstream patches; official APIs only; keyword-first; `max_new_tokens=512` guardrail.
- Four-state claim vocabulary; evidence-first (every claim → archived machine-generated artifact under `evidence/`); transcripts never hand-edited.
- Honest boundaries: a step that cannot produce evidence reports `BLOCKED: <reason>` — never a green claim. A feasibility step's deliverable may legitimately be a documented NEGATIVE verdict.
- One logical commit per step after its verifier PASS; implementers never commit; push exactly once at the end.

## Steps

### Step A — CI closure
Verify the first real CI run on push head `8815238` (run 35526426415: test 3.10/3.11/3.12 + build all success; job log "251 passed, 1 skipped, 38 deselected"). Record: `evidence/ci-2026-09-21-8815238.txt` (machine-captured gh outputs), one-line confirmed-run note in `docs/p0-parity-report.md` test summary + README CI sentences (EN+CN). Verifier independently re-queries GitHub and reconciles counts.

### Step B — Upstream flash-attn issue
Draft and file (via `gh`, account AIwork4me — pre-authorized by the user's directive this run) a quality issue on `QwenLM/Qwen3-TTS`: `finetuning/sft_12hz.py` hard-codes `attn_implementation="flash_attention_2"`, which makes official fine-tuning fail out-of-the-box on ROCm stacks that ship no flash-attn; include environment, verbatim failure, pinned SHA, minimal repro (the smoke run), suggested fix (parameterize attn / fall back to sdpa). Record the issue URL in `docs/finetuning-rocm.md` (recommended action → filed) + CHANGELOG. Verifier confirms the filed issue's content is accurate to our evidence (no overclaim) and the URL resolves.

### Step C — vLLM-Omni offline feasibility (verdict-only)
Using the pinned local upstream clone (`.upstream/Qwen3-TTS`, still present) + upstream README/examples: determine whether upstream's documented vLLM-Omni offline path can run on this stack (torch 2.12.0+rocm7.14.0, gfx1151). Attempt install of a compatible vLLM build; if installable, run the smallest documented offline example (CustomVoice) and capture timings/waveform sanity. Deliverable EITHER way: `evidence/vllm-omni-feasibility-2026-09-21.{json,txt}` with a verdict FEASIBLE(+evidence) or BLOCKED(+exact blocker: wheel/dependency/runtime error verbatim). README roadmap row stays non-promissory; at most 🟡 with evidence per P3. Never "supported" language. Verifier re-checks the verdict against the artifacts and re-runs any local commands that are cheap.

### Step D — True streaming measurements (five metrics)
Investigate the installed qwen-tts 0.1.1 streaming surface (e.g. the `non_streaming_mode` kwarg implies an incremental path). If a genuine incremental-audio path exists on the official API: measure on gfx1151 — time to first audio, chunk cadence, total RTF, buffer-underrun behavior (real-time consumer simulation), long-text behavior; archive `evidence/streaming-2026-09-21.{json,txt}`; update README rows strictly per evidence (upstream's "97 ms" never reused as a Radeon number). If no usable incremental path exists locally: document exactly that with the API surface inspected — the 🚫 row stands with the finding recorded. Verifier re-runs the measurement script and reconciles numbers.

### Step E — Cross-day benchmark replication
Re-run both benchmark sets (5 aliases total) dated 2026-09-21 (`.venv/bin/python scripts/benchmark.py --aliases ... --json-out evidence/benchmark-2026-09-21.json` + the 0.6B set), teeing transcripts. Compare against the 2026-09-20 runs within documented iGPU variance; record deltas in `docs/benchmarks.md` (cross-day replication note); evidence/README.md index. No conclusion beyond variance observations. Verifier re-runs a subset and reconciles.

## Push

After E's verifier PASS: single `git push origin main`, then final summary with rulings list.
