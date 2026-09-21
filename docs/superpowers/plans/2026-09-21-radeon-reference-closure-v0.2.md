# Radeon Reference Closure v0.2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved gated program that closes the v0.2 gaps: upstream #372 fixed via a verified minimal PR (user-gated submission), GPU CI prepared (BLOCKED state), vLLM-vs-qwen-tts measured A/B, a quality-benchmark foundation, an upstream drift watcher, claims audit, full regression, and a truthful `v0.2.0` release.

**Architecture:** Strict serial gating per the spec. Per gated task: the orchestrator dispatches a fresh implementation subagent carrying that task's plan section verbatim; the implementer never commits; a fresh independent verification subagent re-runs tests and audits evidence; the orchestrator commits only after PASS. Task N+1 starts only after Task N's verifier returns PASS. GPU work is serialized (one GPU task at a time).

**Tech Stack:** Python 3.12 venv (`.venv/`), `qwen-tts==0.1.1` (never modified), `torch 2.12.0+rocm7.14.0`, pytest with `gpu`/`requires_download` markers, AMD gfx1151 (Radeon 8060S, unified memory), upstream clone `.upstream/Qwen3-TTS` @ `022e286` + FIX worktree `.upstream/Qwen3-TTS-fix`, isolated `.work-vllm/venv` (vllm 0.28.0+rocm723 / vllm-omni 0.28.0), `gh` CLI authenticated as `AIwork4me`.

**Spec:** `docs/superpowers/specs/2026-09-21-radeon-reference-closure-v0.2-design.md` (commit `953e01e`). The plan argues from the spec; executors read both.

## Global Constraints

- **Brief task numbering** (Tasks 0–20) is preserved in this plan's headers as `(brief N)`; verifier reports and commits reference brief numbers.
- **Zero upstream patches downstream.** The downstream repo never modifies `qwen_tts` package source and never commits upstream code. Upstream modifications live ONLY on the FIX worktree branch `fix/finetuning-attn-implementation` under gitignored `.upstream/Qwen3-TTS-fix/`.
- **Pristine invariant:** `.upstream/Qwen3-TTS` stays byte-identical to its pinned SHA; `git -C .upstream/Qwen3-TTS status --porcelain` must output nothing at every check.
- **Upstream-fix prohibitions (brief Task 4):** no AMD-specific branches; no `torch.version.hip` checks; no changed global default (default stays `flash_attention_2`); no vendoring; no changes to model architecture, optimizer, dataset handling, or checkpoint format; no dependency on Qwen3-TTS-ROCm.
- **Evidence policy:** every GPU run produces `evidence/<name>-<date>.{json,txt}` with date, git HEAD at run time, upstream SHA, qwen-tts/python/torch/ROCm/GPU/gfx, exact commands, exit codes, `rocm-smi` snapshots before/after, full output. Transcripts are never hand-edited. `evidence/README.md` index updated for every new file.
- **GPU serialization:** one GPU-consuming task at a time. Before/after every GPU run, capture `rocm-smi --showproductname --showuse --showmeminfo vram` into the transcript.
- **Claim vocabulary:** ✅ Radeon E2E validated / 🟡 partial-load-only / ⬜ not validated / 🚫 not exposed upstream or intentionally unclaimed. "Supported" ≠ "validated". Fine-tuning wording while PR is OPEN-unmerged is fixed to: "ROCm E2E execution proven. Upstream portability fix submitted as Qwen3-TTS PR #XXX; current published `qwen-tts==0.1.1` still requires the documented temporary workaround." No "fixed upstream" until merged; no "GPU CI live" until a real GPU Actions run succeeds; no second-architecture parity claims.
- **CPU suite command:** `.venv/bin/python -m pytest -m 'not gpu' -q` must stay green after every task. GPU suite: `.venv/bin/python -m pytest -m gpu -q`.
- **Honest boundaries:** a step that cannot produce its evidence stops at `BLOCKED: <reason>` — never converted into a green claim.
- **Push cadence:** push `origin/main` after each phase's final gate (P1 → Task 2 commit, P2 → Task 10 commit, P3 → Task 13 commit, P4 → Task 15 commit, P5 → Task 19). The FIX branch is pushed only as part of Task 9's PR flow.
- **User gates (hard stops, no proceeding without explicit user confirmation):** (1) Task 9 `gh pr create`; (2) Task 19 `gh release create` — the release also requires final-verifier PASS.

## Verification Protocol (applies to every gated task; referenced by every gate step)

Dispatch a fresh general-purpose subagent with this contract verbatim, plus the task's acceptance criteria and file list:

> You are an independent release verifier. Assume the implementing agent may be wrong. Review source, git diff, runtime output, raw evidence, environment metadata and documentation claims. Try to falsify the claimed result. Do not accept "looks correct." Return PASS only when every acceptance criterion is backed by executable or runtime evidence.

The verifier must produce a report at `docs/superpowers/reports/` with exactly these fields: **Verdict** (PASS/FAIL); **Exact commit / working-tree SHA reviewed**; **Acceptance criteria checklist**; **Files reviewed**; **Exact commands executed**; **Exit codes**; **Runtime evidence inspected**; **Regression tests**; **Claims audit**; **Problems found**; **Why PASS or FAIL is justified**. The verifier re-runs the CPU suite itself; for GPU tasks it re-runs at least one representative GPU test itself. A PASS without test/evidence details is invalid. On FAIL: orchestrator dispatches a fix round carrying the findings, then a NEW verifier. Repeat until PASS.

## File Structure (new files created by this plan)

```
evidence/reference-closure-ground-truth-2026-09-21.md   (Task 0)
evidence/upstream-372-pristine-failure-run1.txt          (Task 1)
evidence/upstream-372-pristine-failure-run2.txt          (Task 1)
evidence/upstream-372-root-cause.md                      (Task 2)
.upstream/Qwen3-TTS-fix/                                 (Task 3, gitignored worktree, FIX branch)
  finetuning/sft_12hz.py                                 (modified: --attn_implementation)
  tests/test_sft_attn_implementation.py                  (Task 4, new)
evidence/upstream-372-round-a-loads{1,2,3}.txt           (Task 5)
evidence/upstream-372-e2e-run{1,2}.{json,txt}            (Task 6)
evidence/upstream-372-default-semantics.txt              (Task 7)
evidence/upstream-372-diff-audit.txt                     (Task 7)
docs/superpowers/reports/rc02-task-*.md                  (per-gate verifier reports)
.work-upstream/pr-body.md                                (Task 9, gitignored)
.github/workflows/gpu-nightly.yml                        (Task 11)
docs/development/gpu-ci-runbook.md                       (Task 11)
evidence/gpu-ci-prep-validation.txt                      (Task 11)
.work-vllm/ab_{qwen,vllm}.py, ab_prompts.json            (Task 12, gitignored)
evidence/vllm-vs-qwen-tts-<date>.json                    (Task 12)
docs/vllm-omni-rocm.md                                   (Task 12)
evidence/quality-benchmark-<date>.{json,txt}             (Task 14)
tests/data/quality_benchmark_manifest.json               (Task 14)
scripts/quality_eval.py                                  (Task 14)
tests/test_quality_eval.py                               (Task 14)
scripts/check_upstream_drift.py                          (Task 15)
docs/upstream-baseline.json                              (Task 15)
tests/test_check_upstream_drift.py                       (Task 15)
evidence/claims-audit-2026-09-21.md                      (Task 16)
evidence/final-regression-<date>.txt                     (Task 17)
docs/radeon-reference-closure-v0.2.md                    (Task 18)
```

Modified: `docs/finetuning-rocm.md` + `README.md` + `README_CN.md` + `CHANGELOG.md` + `evidence/README.md` (Tasks 10, 12, 14, 15, 16 as their deliverables land), `pyproject.toml` (Task 14 optional extras only if investigation passes).

---

### Task 0: Ground truth 2026-09-21 (brief 0 — chore commit, no verifier gate per P0 precedent)

**Files:**
- Create: `evidence/reference-closure-ground-truth-2026-09-21.md`

**Interfaces:**
- Produces: both HEAD SHAs, versions, matrix/evidence inventory, #372 state, and the verbatim current content of upstream `finetuning/sft_12hz.py` — cited by every later task brief.

- [ ] **Step 0.1: Dispatch the audit subagent** (read-only). It must record: downstream `git rev-parse HEAD` + `git status --porcelain`; upstream HEAD via `git -C .upstream/Qwen3-TTS rev-parse HEAD` AND `git ls-remote https://github.com/QwenLM/Qwen3-TTS HEAD` (flag if they differ — upstream may have moved; if moved, `git -C .upstream/Qwen3-TTS fetch origin && git checkout <new-SHA>` and note it); `.venv/bin/pip show qwen-tts | head -2`; `grep '^version' pyproject.toml`; `gh release list --limit 3`; `gh issue view 372 --repo QwenLM/Qwen3-TTS --json state,title,comments` (record state verbatim); CPU+GPU test counts (`.venv/bin/python -m pytest -m 'not gpu' -q --co -q | tail -1` and same for `-m gpu`); capability-matrix rows as currently in README; `evidence/` inventory; and `cat .upstream/Qwen3-TTS/finetuning/sft_12hz.py` in full with line numbers. It must then verify the #372 root cause still exists: `grep -n 'flash_attention_2' .upstream/Qwen3-TTS/finetuning/sft_12hz.py` shows the hard-coded value at the `from_pretrained` call.
- [ ] **Step 0.2: Orchestrator reviews** — every fact above present; any gap → one follow-up to the same subagent.
- [ ] **Step 0.3: Commit**

```bash
git add evidence/reference-closure-ground-truth-2026-09-21.md evidence/README.md
git commit -m "chore(evidence): reference-closure ground truth 2026-09-21 (both SHAs, #372 state, verbatim sft_12hz.py)"
```

---

### Task 1: Pristine double reproduction (brief 1)

**Files:**
- Create: `evidence/upstream-372-pristine-failure-run1.txt`, `evidence/upstream-372-pristine-failure-run2.txt`

**Interfaces:**
- Consumes: Task 0 SHAs; the validated fine-tune invocation recorded in `evidence/finetune-smoke-2026-09-20.txt` (extract the exact dataset-prep + `sft_12hz.py` commands used there).
- Produces: two independent AS-IS failure transcripts proving the failure class from byte-pristine upstream.

- [ ] **Step 1.1: Verify pristine state** — `git -C .upstream/Qwen3-TTS status --porcelain` (must be empty), `git -C .upstream/Qwen3-TTS rev-parse HEAD` (record), `grep -n flash_attention_2 .upstream/Qwen3-TTS/finetuning/sft_12hz.py` (record line).
- [ ] **Step 1.2: Reproduce run 1** — fresh output dirs `.work-finetune/repro-r1/`; re-run the dataset prep exactly as recorded in the 2026-09-20 smoke transcript (fresh copies, no reuse), then from `.upstream/Qwen3-TTS`:

```bash
cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS && \
rocm-smi --showproductname --showuse --showmeminfo vram && \
/home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python finetuning/sft_12hz.py \
  <exact args from the 2026-09-20 smoke, unchanged> 2>&1 | tee /home/amd/Desktop/Qwen3-TTS-ROCm/.work-finetune/repro-r1.log; echo "EXIT=$?" | tee -a .../repro-r1.log
```

Expected: `ImportError: FlashAttention2 has been toggled on, ... flash_attn seems to be not installed` (verbatim class match with the 2026-09-20 capture). Non-zero exit.
- [ ] **Step 1.3: Reproduce run 2** — new shell/process, fresh dirs `.work-finetune/repro-r2/`, identical procedure. Same failure class required.
- [ ] **Step 1.4: Archive both transcripts** to the two evidence files with headers (GPU, gfx, ROCm, torch, Python, qwen-tts, upstream SHA, exact command, exit code) + `rocm-smi` snapshots. Re-verify pristine (`status --porcelain` empty).
- [ ] **Step 1.5: Verifier gate** — Verification Protocol; criteria: pristine confirmed before/after; two fresh processes/dirs; same failure class both times; transcripts complete and unedited; no source modification occurred.
- [ ] **Step 1.6: Commit**

```bash
git add evidence/upstream-372-pristine-failure-run1.txt evidence/upstream-372-pristine-failure-run2.txt evidence/README.md
git commit -m "chore(evidence): reproduce Qwen fine-tuning attention failure twice from pristine upstream"
```

---

### Task 2: Root cause + controlled isolation (brief 2)

**Files:**
- Create: `evidence/upstream-372-root-cause.md`
- Create (gitignored scratch): `.upstream/Qwen3-TTS-fix` worktree (set up here, first modified in Task 3)

**Interfaces:**
- Consumes: Task 1 transcripts; upstream sources.
- Produces: the causal-chain document Tasks 3/8/9 cite; the FIX worktree used from Task 3 on.

- [ ] **Step 2.1: Create the FIX worktree (no edits yet)**

```bash
cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS && \
git worktree add ../Qwen3-TTS-fix -b fix/finetuning-attn-implementation && \
git -C ../Qwen3-TTS-fix status --porcelain   # must be empty
```

- [ ] **Step 2.2: Answer the 8 brief questions from source** — (1) exact line of the FA2 selection; (2) configurability elsewhere (grep upstream for other `attn_implementation` uses — the inference path/env); (3) passed directly to `Qwen3TTSModel.from_pretrained`? (yes/no + line); (4) does official inference allow SDPA/alternates (check installed `qwen_tts` wrapper + our `loader.py` SDPA HIP default); (5) does model execution itself require FA2 or is it a loader configuration choice (evidence: our validated SDPA inference + the 2026-09-20 sdpa-override training smoke); (6) does replacing ONLY the attention implementation allow training to proceed (→ Step 2.3 experiment); (7) any second hidden FA2 dependency later in the workflow (grep `flash` across `finetuning/` and imported modules; also `sft_12hz.py` for eager/sdpa-specific code paths); (8) does `sdpa` preserve the expected Qwen3-TTS model path (config/`_attn_implementation` check during the experiment; no architecture or training-logic change).
- [ ] **Step 2.3: Controlled isolation experiment in the FIX worktree** — change ONLY the one token: in `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py`, `attn_implementation="flash_attention_2"` → `attn_implementation="sdpa"` (record the one-line diff). Run the full chain: prepare (fresh dir `.work-finetune/iso/`) → load → train (same small scale as the 2026-09-20 smoke) → checkpoint save → reload through official API → synthesis → waveform sanity (`assert_wav_sane`-equivalent: finite, non-silent, expected sr/duration). Capture everything. **If a second blocker appears: STOP, investigate it, and record it — never force-fit the hypothesis.**
- [ ] **Step 2.4: Write `evidence/upstream-372-root-cause.md`** — causal chain, exact source lines, the controlled experiment (command + output + result), alternative hypotheses considered and rejected, remaining uncertainties. Restore the FIX worktree pristine afterwards (`git -C .upstream/Qwen3-TTS-fix checkout -- .`) — Task 3 re-does the change properly.
- [ ] **Step 2.5: Verifier gate** — criteria: all 8 questions answered with source citations; isolation diff was exactly one token; full chain completed (or second blocker honestly documented); worktree restored pristine.
- [ ] **Step 2.6: Commit**

```bash
git add evidence/upstream-372-root-cause.md evidence/README.md
git commit -m "docs(upstream): record Qwen3-TTS #372 root cause and controlled isolation experiment"
```

---

### Task 3: Design decision + minimal fix on FIX branch (brief 3+4)

**Files:**
- Modify: `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py`

**Interfaces:**
- Consumes: Task 2 causal chain; Task 0 verbatim `sft_12hz.py`.
- Produces: patched `sft_12hz.py` exposing `--attn_implementation` (default `flash_attention_2`) and a factored `build_arg_parser()` used by Task 4's tests.

- [ ] **Step 3.1: Document the design decision** (goes into the eventual PR body + Task 4 test docstring): Option A (`--attn_implementation` CLI, default `flash_attention_2`) chosen over Option B (auto-fallback) because A preserves CUDA behavior exactly, has no hidden platform detection, minimal diff, explicit user control; B rejected for silent-behavior-change risk. Decision recorded; revisit only if upstream structure strongly argues otherwise (it does not, per Task 2).
- [ ] **Step 3.2: Apply the patch** to `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py`, matching upstream's argument style (no help-strings, same ordering conventions):

```python
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
```

and the load call becomes:

```python
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
```

If the parser is inline in `train()` (verify against Task 0's verbatim capture), factor minimally so tests can build args without executing training:

```python
def build_arg_parser():
    parser = argparse.ArgumentParser()
    # ... all existing parser.add_argument lines moved here unchanged ...
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser


def train():
    args = build_arg_parser().parse_args()
    # ... unchanged ...
```

No other lines change. Verify: `git -C .upstream/Qwen3-TTS-fix diff` shows only this.
- [ ] **Step 3.3: Smoke-parse check** — `cd .upstream/Qwen3-TTS-fix && /home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python -c "import sys; sys.path.insert(0,'finetuning'); import sft_12hz; a=sft_12hz.build_arg_parser().parse_args(['--attn_implementation','sdpa']); print(a.attn_implementation)"` → prints `sdpa`.
- [ ] **Step 3.4: Verifier gate** — criteria: diff contains ONLY the configurability change (+ parser factoring if needed); default unchanged; upstream style respected; no prohibited change (Global Constraints list) present.
- [ ] **Step 3.5: Commit (on the FIX branch)**

```bash
cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix && \
git add finetuning/sft_12hz.py && \
git commit -m "fix(finetuning): make attention implementation configurable"
```

(Downstream `main` gets no commit for this task — the change lives upstream-side only; the verifier report lands downstream in Task 4's commit.)

---

### Task 4: Upstream unit regression tests (brief 5)

**Files:**
- Create: `.upstream/Qwen3-TTS-fix/tests/test_sft_attn_implementation.py`

**Interfaces:**
- Consumes: `sft_12hz.build_arg_parser()`, `sft_12hz.train()`, `sft_12hz.Qwen3TTSModel`, `sft_12hz.Accelerator` (Task 3).
- Produces: stdlib-unittest proof of the 4 brief criteria, runnable without GPU training.

- [ ] **Step 4.1: Write the test** (stdlib `unittest` only — upstream carries no pytest infra):

```python
"""Regression tests: --attn_implementation in finetuning/sft_12hz.py.

Proves: (1) default stays flash_attention_2; (2) sdpa selectable;
(3) the selected value reaches Qwen3TTSModel.from_pretrained;
(4) parser stays backward compatible (unknown-flag-free default parse).
GPU-free: from_pretrained is intercepted before any weight load.
"""
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))
import sft_12hz  # noqa: E402


class _StopAfterLoad(Exception):
    pass


def _parse(argv):
    with mock.patch.object(sys, "argv", ["sft_12hz.py", *argv]):
        return sft_12hz.build_arg_parser().parse_args()


class AttnImplementationTests(unittest.TestCase):
    def test_default_is_flash_attention_2(self):
        self.assertEqual(_parse([]).attn_implementation, "flash_attention_2")

    def test_explicit_sdpa_selectable(self):
        self.assertEqual(
            _parse(["--attn_implementation", "sdpa"]).attn_implementation, "sdpa"
        )

    def test_value_reaches_model_loader(self):
        captured = {}

        def fake_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            raise _StopAfterLoad()

        with mock.patch.object(sft_12hz, "Accelerator"), \
                mock.patch.object(sft_12hz, "Qwen3TTSModel") as q:
            q.from_pretrained.side_effect = fake_from_pretrained
            with mock.patch.object(sys, "argv",
                                   ["sft_12hz.py", "--attn_implementation", "sdpa"]):
                with self.assertRaises(_StopAfterLoad):
                    sft_12hz.train()
        self.assertEqual(captured["attn_implementation"], "sdpa")

    def test_default_value_reaches_model_loader(self):
        captured = {}

        def fake_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            raise _StopAfterLoad()

        with mock.patch.object(sft_12hz, "Accelerator"), \
                mock.patch.object(sft_12hz, "Qwen3TTSModel") as q:
            q.from_pretrained.side_effect = fake_from_pretrained
            with mock.patch.object(sys, "argv", ["sft_12hz.py"]):
                with self.assertRaises(_StopAfterLoad):
                    sft_12hz.train()
        self.assertEqual(captured["attn_implementation"], "flash_attention_2")

    def test_backward_compatible_parse(self):
        # The pre-patch argument set parses identically with no new flags.
        args = _parse(["--num_epochs", "1", "--speaker_name", "speaker_test"])
        self.assertEqual(args.num_epochs, 1)
        self.assertEqual(args.speaker_name, "speaker_test")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4.2: Run** — `cd .upstream/Qwen3-TTS-fix && /home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python -m unittest tests.test_sft_attn_implementation -v` → all PASS. (If upstream's actual pre-patch flags differ, adjust ONLY `test_backward_compatible_parse`'s flag list to match Task 0's verbatim capture.)
- [ ] **Step 4.3: Verifier gate** — criteria: 4 brief criteria each mapped to a passing test; mocking limited to `Accelerator` + `from_pretrained` interception (no heavy mocking that would prove nothing); tests pass in a fresh process.
- [ ] **Step 4.4: Commit (FIX branch) + downstream report commit**

```bash
cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix && \
git add tests/test_sft_attn_implementation.py && \
git commit -m "test(finetuning): cover default and explicit attention implementation"
cd /home/amd/Desktop/Qwen3-TTS-ROCm && \
git add docs/superpowers/reports/ && \
git commit -m "test(upstream): validate configurable attention backend on ROCm (unit regression proof)"
```

---

### Task 5: Round A — targeted loader validation ×3 fresh processes (brief 6A)

**Files:**
- Create: `evidence/upstream-372-round-a-loads1.txt`, `...-loads2.txt`, `...-loads3.txt`
- Create (gitignored): `.work-finetune/round_a_load_check.py`

**Interfaces:**
- Consumes: FIX worktree patched module; `models/` checkpoint used by the 2026-09-20 smoke (exact path from its transcript).
- Produces: three independent load proofs with real GPU residency.

- [ ] **Step 5.1: Write the driver** `.work-finetune/round_a_load_check.py`:

```python
"""Round A: load the model exactly as patched sft_12hz.py would, via its own symbols."""
import sys

import torch

sys.path.insert(0, "/home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix/finetuning")
import sft_12hz  # noqa: E402

MODEL = sys.argv[1]
args = sft_12hz.build_arg_parser().parse_args(["--attn_implementation", "sdpa"])
print(f"parsed attn_implementation={args.attn_implementation}")
qwen3tts = sft_12hz.Qwen3TTSModel.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation
)
impl = getattr(qwen3tts.model.config, "_attn_implementation", "<absent>")
print(f"config._attn_implementation={impl}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"mem_alloc_GiB={torch.cuda.max_memory_allocated() / 2**30:.2f}")
assert impl == "sdpa", f"expected sdpa, got {impl}"
print("ROUND_A_LOAD_OK")
```

- [ ] **Step 5.2: Run it 3× in fresh processes**, each wrapped with `rocm-smi` snapshots, output teed to its evidence file: criteria per run: `parsed attn_implementation=sdpa`, `config._attn_implementation=sdpa`, real device name printed, exit 0.
- [ ] **Step 5.3: Verifier gate** — criteria: 3 distinct process invocations visible in transcripts; all three PASS criteria; no FlashAttention import error anywhere.
- [ ] **Step 5.4: Commit**

```bash
git add evidence/upstream-372-round-a-loads*.txt evidence/README.md docs/superpowers/reports/
git commit -m "test(upstream): Round A — patched loader accepts sdpa on ROCm in 3 fresh processes"
```

---

### Task 6: Rounds B+C — two independent full fine-tuning E2E runs (brief 6B+6C)

**Files:**
- Create: `evidence/upstream-372-e2e-run1.{json,txt}`, `evidence/upstream-372-e2e-run2.{json,txt}`

**Interfaces:**
- Consumes: FIX branch patched script; dataset prep from the 2026-09-20 flow; `models/` checkpoint; `.work-finetune/reload_check.py` pattern (official-API reload + synthesis check from the prior smoke — adapt paths only).
- Produces: two fully independent E2E proofs (the core PR validation evidence).

- [ ] **Step 6.1: Run E2E #1 from clean dirs** — fresh `.work-finetune/e2e-r1/`; dataset prep exactly as official flow; then from the FIX worktree:

```bash
cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix && \
rocm-smi --showproductname --showuse --showmeminfo vram && \
/home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python finetuning/sft_12hz.py \
  --attn_implementation sdpa <same scale/args as the 2026-09-20 smoke> \
  2>&1 | tee /home/amd/Desktop/Qwen3-TTS-ROCm/.work-finetune/e2e-r1.log; echo "EXIT=$?"
```

Then checkpoint reload through the official `qwen_tts` API + synthesis + waveform sanity (reuse the `.work-finetune/reload_check.py` pattern against the new checkpoint). Record: optimizer step count, loss values (start/end), batch size, precision, wall time, peak memory (`torch.cuda.max_memory_reset()` at start; read at end from the training process — the prior smoke's launcher pattern exists precisely for this; reuse `.work-finetune/run_sft_launcher.py` adapted to the FIX worktree if direct invocation cannot expose peak-mem), checkpoint path, generated audio metadata (sr, duration, finite, non-silent).
- [ ] **Step 6.2: Run E2E #2 fully fresh** — new process, new `.work-finetune/e2e-r2/`, dataset re-prepared, NO reused checkpoint or trainer state. Same procedure and records.
- [ ] **Step 6.3: Archive** both runs' `{json,txt}` evidence with full headers per Global Constraints.
- [ ] **Step 6.4: Verifier gate** — criteria: two runs from genuinely clean state (dirs recreated, dataset re-prepared, processes independent); both completed train→save→reload→synthesis with sane waveforms; metrics recorded for both; the only upstream difference vs pristine is the FIX branch diff.
- [ ] **Step 6.5: Commit**

```bash
git add evidence/upstream-372-e2e-run1.* evidence/upstream-372-e2e-run2.* evidence/README.md docs/superpowers/reports/
git commit -m "test(upstream): Rounds B+C — two independent ROCm fine-tuning E2E runs pass with sdpa"
```

---

### Task 7: Rounds D+E — default semantics + diff audit (brief 6D+6E)

**Files:**
- Create: `evidence/upstream-372-default-semantics.txt`, `evidence/upstream-372-diff-audit.txt`

**Interfaces:**
- Consumes: FIX branch vs upstream pinned main; Task 4 tests.
- Produces: default-preservation proof and the audited minimal diff (PR-ready).

- [ ] **Step 7.1: Round D** — with NO `--attn_implementation` flag: run the Task 4 unittest suite (proves default parse = `flash_attention_2`) AND attempt a real default-flag load on this host: `.work-finetune/round_a_load_check.py` variant parsing `[]` — expected to FAIL here with the FlashAttention ImportError (flash_attn absent). Record both. Statement recorded verbatim: "Default configuration semantics preserved; CUDA runtime was not independently tested." Never claim CUDA validation.
- [ ] **Step 7.2: Round E** — `cd .upstream/Qwen3-TTS-fix && git diff <pinned-upstream-main-SHA>...HEAD > /home/amd/Desktop/Qwen3-TTS-ROCm/.work-finetune/fix-branch.diff`; audit EVERY changed line against: only configurability + tests; no unrelated cleanup; no formatting churn; no downstream branding; no prohibited change. Write the audit with per-hunk verdicts into `evidence/upstream-372-diff-audit.txt` (append the diff itself).
- [ ] **Step 7.3: Verifier gate** — criteria: default audit shows FA2 preserved (unit) + expected ROCm failure unchanged (runtime); diff audit covers every hunk; diff is minimal and generic.
- [ ] **Step 7.4: Commit**

```bash
git add evidence/upstream-372-default-semantics.txt evidence/upstream-372-diff-audit.txt evidence/README.md docs/superpowers/reports/
git commit -m "test(upstream): Rounds D+E — default flash_attention_2 semantics preserved; minimal diff audited"
```

---

### Task 8: Independent #372 chain verifier (brief 7)

**Files:**
- Create: `docs/superpowers/reports/rc02-task-8-upstream-372-verdict.md` (written by the verifier subagent)

**Interfaces:**
- Consumes: everything — issue #372 text (`gh issue view 372 --repo QwenLM/Qwen3-TTS`), pristine SHA, PR-branch diff, both failure runs, root-cause report, all Round A–E evidence, test outputs, checkpoint/reload evidence, waveform evidence.
- Produces: the PASS that unblocks PR preparation.

- [ ] **Step 8.1: Dispatch the fresh verifier** with the Verification Protocol contract PLUS the brief's challenge list, requiring an explicit verdict per challenge:
  - A. Was the original problem really reproduced from pristine upstream (twice)?
  - B. Is the hard-coded attention backend really the minimal root cause?
  - C. Does the fix preserve upstream default behavior?
  - D. Does ROCm fine-tuning execute without editing upstream source at runtime?
  - E. Did two independent E2E training runs pass?
  - F. Is the patch generic rather than AMD-specific?
  - G. Is the PR diff minimal?
  - H. Are documentation claims no stronger than evidence?
- [ ] **Step 8.2: On PASS** — commit the report; on FAIL — fix round + NEW verifier (per protocol) before anything else.

```bash
git add docs/superpowers/reports/rc02-task-8-upstream-372-verdict.md
git commit -m "docs(upstream): #372 chain independent verifier PASS (challenges A-H)"
```

---

### Task 9: PR preparation + USER GATE + submission (brief 8) — HARD USER STOP

**Files:**
- Create (gitignored): `.work-upstream/pr-body.md`

**Interfaces:**
- Consumes: Task 8 PASS; FIX branch; validation summaries.
- Produces: upstream PR URL + number recorded in downstream evidence.

- [ ] **Step 9.1: Assemble the PR body** into `.work-upstream/pr-body.md` exactly per the spec template (Summary / Root cause / Validation with the exact host stack — GPU, gfx, ROCm, torch, Python, Qwen3-TTS SHA — and validation list: pristine failure reproduction ×2, explicit sdpa load, two independent E2E fine-tuning smoke runs, checkpoint save, checkpoint reload, synthesis after reload, regression tests; Backward compatibility: default remains `flash_attention_2`; Related: `Fixes #372`). Upstream-centered; no Qwen3-TTS-ROCm promotion.
- [ ] **Step 9.2: ★ USER GATE ★** — present to the user: the full `git diff` of the FIX branch, the PR body, and the Task 8 verifier summary. Wait for explicit confirmation. **Do not run Step 9.3 without it.**
- [ ] **Step 9.3: Submit (only after user confirmation)**

```bash
gh repo fork QwenLM/Qwen3-TTS --clone=false && \
cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix && \
git remote add fork https://github.com/AIwork4me/Qwen3-TTS.git 2>/dev/null || true && \
git push fork fix/finetuning-attn-implementation && \
gh pr create --repo QwenLM/Qwen3-TTS \
  --base main --head AIwork4me:fix/finetuning-attn-implementation \
  --title "fix(finetuning): make attention implementation configurable" \
  --body-file /home/amd/Desktop/Qwen3-TTS-ROCm/.work-upstream/pr-body.md
```

- [ ] **Step 9.4: Record** PR URL + number into `evidence/upstream-372-root-cause.md` (appendix) and `CHANGELOG.md`. Commit. Issue is called "resolved" only after actual merge.

```bash
git add evidence/upstream-372-root-cause.md CHANGELOG.md
git commit -m "chore(upstream): submit Qwen3-TTS PR for configurable attention implementation (#372)"
```

---

### Task 10: Downstream fine-tuning claim cleanup (brief 9)

**Files:**
- Modify: `docs/finetuning-rocm.md`, `README.md`, `README_CN.md`, `CHANGELOG.md`, `evidence/README.md` (as needed)

**Interfaces:**
- Consumes: Task 9 PR number; the Global Constraints fixed wording.
- Produces: truthful downstream fine-tuning state everywhere.

- [ ] **Step 10.1: Update wording** — `docs/finetuning-rocm.md`: replace the "workaround is the only path" framing with the fixed wording (PR #XXX submitted; published `qwen-tts==0.1.1` still requires the documented temporary workaround until merge); README matrices: fine-tuning row stays at its honest state with a pointer to the PR. If the PR was MERGED during execution instead: fetch merged upstream, verify merge SHA, run the E2E workflow once with ZERO source edits from pristine merged upstream, archive, and only then turn the row fully green.
- [ ] **Step 10.2: Verifier gate** — criteria: no claim stronger than "submitted as PR #XXX"; fixed wording used verbatim where required; CN/EN consistent.
- [ ] **Step 10.3: Commit**

```bash
git add docs/finetuning-rocm.md README.md README_CN.md CHANGELOG.md evidence/README.md docs/superpowers/reports/
git commit -m "docs: align fine-tuning claims with upstream PR status (#372)"
git push origin main
```

---

### Task 11: GPU CI preparation — BLOCKED state (brief 12)

**Files:**
- Create: `.github/workflows/gpu-nightly.yml`, `docs/development/gpu-ci-runbook.md`, `evidence/gpu-ci-prep-validation.txt`

**Interfaces:**
- Consumes: existing CPU CI conventions (`.github/workflows/` — read and match style/trigger conventions).
- Produces: a ready-to-activate workflow + runbook; publicly marked BLOCKED ON RUNNER INFRASTRUCTURE.

- [ ] **Step 11.1: Write `.github/workflows/gpu-nightly.yml`** (match the existing CPU workflow's checkout/venv/pytest invocation style; adjust job names to the suites below):

```yaml
name: gpu-nightly
on:
  schedule:
    - cron: "0 2 * * *"          # nightly 02:00 runner-local time
  workflow_dispatch:
    inputs:
      suite:
        description: "gpu-short | full-weekly"
        required: false
        default: "gpu-short"

jobs:
  gpu-short:
    if: ${{ github.event.inputs.suite != 'full-weekly' }}
    runs-on: [self-hosted, radeon-gfx1151]
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - name: Environment diagnostics
        run: |
          rocm-smi --showproductname --showuse --showmeminfo vram
          .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
      - name: Upstream parity + tokenizer (GPU)
        run: .venv/bin/python -m pytest -m gpu -q -k "parity or tokenizer"
      - name: Core generation suites
        run: .venv/bin/python -m pytest -m gpu -q -k "custom_voice or voice_design or voice_clone or voice_workflow or validate_languages"
      - name: Demo backend workflow
        run: .venv/bin/python -m pytest -m gpu -q -k "demo or studio"

  full-weekly:
    if: ${{ github.event.inputs.suite == 'full-weekly' }}
    runs-on: [self-hosted, radeon-gfx1151]
    timeout-minutes: 720
    steps:
      - uses: actions/checkout@v4
      - name: Full GPU regression
        run: .venv/bin/python -m pytest -m gpu -q
      - name: Fine-tuning execution smoke
        run: bash scripts/verify_gpu.sh   # extended per runbook §fine-tune if needed
      - name: Benchmark replication
        run: .venv/bin/python scripts/benchmark.py --json evidence/benchmark-nightly.json
```

(Refine `-k` filters against the actual test node names discovered at execution: `.venv/bin/python -m pytest -m gpu --co -q`. The YAML above is the required shape: self-hosted label, nightly short suite = parity/tokenizer/both CustomVoice/VoiceDesign/both Base clone/reusable prompt/Voice Studio backend; weekly/manual = full GPU matrix + fine-tune smoke + benchmark.)
- [ ] **Step 11.2: Write `docs/development/gpu-ci-runbook.md`** — sections: prerequisites (the validated stack, `models/` present, `.venv` built by `scripts/install.sh`); runner install (`gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/runners/registration-token` flow, `./config.sh --labels radeon-gfx1151 --unattended`, `./svc.sh install/start`); label security note (self-hosted on private desktop — repo must stay public-trusted; never enable for forks via `Settings → Actions → Runner group`); maintenance (venv refresh, models refresh, disk); how to trigger `workflow_dispatch`; current status header: **BLOCKED ON RUNNER INFRASTRUCTURE — workflow validated, never executed on a runner; no GPU CI badge until a real run is green**.
- [ ] **Step 11.3: Local validation** — YAML parse (`python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/gpu-nightly.yml'))"` or actionlint if available); collect the actual `-m gpu --co -q` node list proving every `-k` filter selects tests; archive all into `evidence/gpu-ci-prep-validation.txt`.
- [ ] **Step 11.4: Update status wording** — README CI section + `docs/p0-parity-report.md` cross-ref if needed: GPU CI = "workflow + runbook prepared; BLOCKED ON RUNNER INFRASTRUCTURE". No badge.
- [ ] **Step 11.5: Verifier gate** — criteria: YAML valid; every `-k` filter maps to real collected tests; runbook complete and honest; BLOCKED wording everywhere applicable; no badge, no live claims.
- [ ] **Step 11.6: Commit**

```bash
git add .github/workflows/gpu-nightly.yml docs/development/gpu-ci-runbook.md evidence/gpu-ci-prep-validation.txt README.md evidence/README.md docs/superpowers/reports/
git commit -m "ci: add Radeon GPU nightly regression workflow (BLOCKED ON RUNNER INFRASTRUCTURE)"
```

---

### Task 12: vLLM-Omni vs qwen-tts controlled A/B (brief 13)

**Files:**
- Create (gitignored): `.work-vllm/ab_prompts.json`, `.work-vllm/ab_qwen.py`, `.work-vllm/ab_vllm.py`
- Create: `evidence/vllm-vs-qwen-tts-<date>.json` (use execution date), `docs/vllm-omni-rocm.md`

**Interfaces:**
- Consumes: `.work-vllm/venv` (vllm-omni stack, `HF_HOME=.work-vllm/hf-home` with the models symlink), main `.venv` (qwen-tts), 1.7B CustomVoice checkpoint alias, the `end2end.py` API usage pattern.
- Produces: the A/B evidence JSON + the deployment-guidance doc Tasks 13/18/19 cite.

- [ ] **Step 12.1: Fix the workload** — `.work-vllm/ab_prompts.json`: 8 prompts (4 zh / 4 en, 2 short ~20 chars / 2 medium ~60 chars / 2 long ~120 chars per language), plus fixed speaker/instruction params copied from the validated CustomVoice invocation; `max_new_tokens=512` both sides.
- [ ] **Step 12.2: Write both drivers** sharing that manifest. `ab_qwen.py` (run with main `.venv`): warmup 1 generation, then per-prompt 3 timed runs via `loader.load("custom-voice-1.7b")`, recording wall time, audio duration, RTF, `torch.cuda.max_memory_allocated`. `ab_vllm.py` (run with `.work-vllm/venv`, `HF_HOME` set): same warmup/timing structure using the same LLM omni invocation pattern as `end2end.py --query-type CustomVoice` (byte-pattern reuse of its API calls; adapt to iterate the shared manifest). Both drivers emit per-prompt JSON rows.
- [ ] **Step 12.3: Execute interleaved** — order qwen, vllm, vllm, qwen (ABBA to cancel drift), one side fully unloaded between sides; `rocm-smi` snapshots around each side; record continuous GPU uptime in the evidence (heat-soak control). Batch/concurrency measurements ONLY if vllm-omni genuinely supports them for this path — otherwise record "not supported in tested path".
- [ ] **Step 12.4: Aggregate** into `evidence/vllm-vs-qwen-tts-<date>.json`: environment blocks (both stacks), per-prompt rows, medians, load times, peak memory, waveform sanity per output (finite, non-silent, sr/duration), explicit unsupported-parity list.
- [ ] **Step 12.5: Write `docs/vllm-omni-rocm.md`** answering, each keyed to evidence rows: What works? What does not? What is faster/slower? What consumes more/less memory? When should a Radeon developer use qwen-tts? When vLLM-Omni? Which conclusions are still unproven? **No quality claims without a quality metric** (point to Task 14's future metrics as explicitly out of scope here).
- [ ] **Step 12.6: Verifier gate** (this is brief Task 14's audit) — criteria: same-workload fairness (identical prompts/params/limits both sides); warmup policy equal; model identity identical; environment isolation real (separate venvs, no cross-imports); memory measurement method stated; RTF = generation_wall / audio_duration consistently; all runs included (no cherry-picking — row count matches manifest × 3); doc wording matches data.
- [ ] **Step 12.7: Commit**

```bash
git add evidence/vllm-vs-qwen-tts-*.json docs/vllm-omni-rocm.md evidence/README.md docs/superpowers/reports/
git commit -m "feat(benchmark): qwen-tts vs vLLM-Omni controlled A/B on gfx1151"
git push origin main
```

---

### Task 13: (folded into Task 12 Step 12.6 — brief 14's independent verifier runs as that gate; no separate implementation work)

---

### Task 14: Quality benchmark foundation (brief 15)

**Files:**
- Create: `scripts/quality_eval.py`, `tests/data/quality_benchmark_manifest.json`, `tests/test_quality_eval.py`, `evidence/quality-benchmark-<date>.{json,txt}`

**Interfaces:**
- Consumes: `loader.load` aliases; `testing.assert_wav_sane`.
- Produces: a manifest-driven quality harness (CER/WER + speaker similarity; STOI where licensing allows) + the licensing decision log; optional extras group `quality` in `pyproject.toml`.

- [ ] **Step 14.1: Investigation gates (decide before installing anything into `.venv`)** — for each candidate record license + ROCm/CPU feasibility + smoke result into the evidence file: `jiwer` (WER/CER math, permissive — expected ACCEPT), `resemblyzer` (speaker embeddings, MIT — probe install + one embedding on CPU), `openai-whisper` (MIT, torch-based — probe `tiny` model transcribe on GPU; if ROCm issues, fall back to CPU and record), `pystoi` (STOI; check license at execution — permissive expected), `pesq` (PESQ; ITU license history — expected REJECT, document and skip). Rejections are recorded, never hidden. Only ACCEPTed candidates are added to `pyproject.toml` `[project.optional-dependencies] quality = [...]` and installed.
- [ ] **Step 14.2: Write the fixed benchmark manifest** `tests/data/quality_benchmark_manifest.json`:

```json
{
  "seed": 20260921,
  "cases": [
    {"id": "zh-short",  "text": "今天天气很好。", "lang": "zh"},
    {"id": "zh-mid",    "text": "这个周末我们打算去山上徒步，顺便拍一些照片。", "lang": "zh"},
    {"id": "en-short",  "text": "The weather is nice today.", "lang": "en"},
    {"id": "en-mid",    "text": "We are planning to go hiking this weekend and take some photos.", "lang": "en"},
    {"id": "clone-ref", "role": "reference-clone", "ref": "tests/data/clone_ref.wav", "text": "欢迎使用语音克隆功能。", "lang": "zh"}
  ]
}
```

(Use the existing bundled clone reference asset from `tests/test_voice_clone_workflow.py` as `ref`; if a different path, adjust to the real one.)
- [ ] **Step 14.3: Write `scripts/quality_eval.py`** — structure (implement fully; degrade gracefully per Step 14.1 verdicts):

```python
#!/usr/bin/env python
"""Quality benchmark: ASR intelligibility (CER/WER via jiwer+whisper) and
speaker similarity (resemblyzer) on the fixed manifest. Separate from the
performance benchmark; never merged into one score."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from qwen3_tts_rocm import loader, testing


def cer(ref: str, hyp: str) -> float: ...   # jiwer on char level (zh)
def wer(ref: str, hyp: str) -> float: ...   # jiwer on word level (en)
def speaker_sim(a: Path, b: Path) -> float: ...  # resemblyzer cosine
def asr_transcribe(wav: Path, lang: str) -> str: ...  # whisper tiny

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="tests/data/quality_benchmark_manifest.json")
    ap.add_argument("--out", required=True)  # JSON path
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    model = loader.load("custom-voice-1.7b")
    rows = []
    for case in manifest["cases"]:
        wav, sr = model.generate_custom_voice(text=case["text"], ...)  # manifest params
        testing.assert_wav_sane(wav, sr_expected=sr)
        row = {"id": case["id"], "cer_or_wer": ..., "transcript": asr_transcribe(...)}
        if case.get("role") == "reference-clone":
            row["speaker_sim_ref_vs_output"] = speaker_sim(...)
        rows.append(row)
    Path(args.out).write_text(json.dumps({"seed": manifest["seed"], "rows": rows}, indent=2))
```

Real implementations of the four helpers are written by the implementer following the ACCEPTed candidates' documented APIs (jiwer `wer`, whisper `transcribe`, resemblyzer `VoiceEncoder.embed_utterance` cosine) — no stubs in the delivered script; any metric whose candidate was REJECTED is omitted and noted in the JSON as `"omitted": "<reason>"`.
- [ ] **Step 14.4: CPU unit tests** `tests/test_quality_eval.py` — `cer`/`wer` correctness on hand-computed examples (e.g. `cer("你好世界","你号世界")==0.25`), `speaker_sim` on identical vs different synthetic tones (monotone sines at different frequencies → high vs low sim; no GPU, no model download, synthesizing wavs with `numpy` + `soundfile` in tmp_path), manifest schema validation. `pytestmark` plain CPU.
- [ ] **Step 14.5: GPU evidence run** — `scripts/quality_eval.py --out evidence/quality-benchmark-<date>.json` with full transcript capture per Global Constraints.
- [ ] **Step 14.6: Verifier gate** — criteria: licensing log complete with per-candidate verdicts; only ACCEPTed deps in `pyproject.toml`; unit tests green in CPU suite; evidence rows match manifest; performance vs quality separation respected; rejected metrics documented not hidden.
- [ ] **Step 14.7: Commit**

```bash
git add scripts/quality_eval.py tests/data/quality_benchmark_manifest.json tests/test_quality_eval.py pyproject.toml evidence/quality-benchmark-*.{json,txt} evidence/README.md docs/superpowers/reports/
git commit -m "feat(eval): automated TTS quality benchmark foundation (CER/WER + speaker similarity)"
```

---

### Task 15: Upstream drift watcher (brief 16)

**Files:**
- Create: `scripts/check_upstream_drift.py`, `docs/upstream-baseline.json`, `tests/test_check_upstream_drift.py`

**Interfaces:**
- Consumes: upstream GitHub/PyPI endpoints (stdlib `urllib.request` only — no new deps).
- Produces: `check(baseline_path) -> dict` and CLI printing `UNCHANGED` / `UPSTREAM DRIFT DETECTED` + diffs, exit 0/1; `--update-baseline` writes the current state.

- [ ] **Step 15.1: Write `scripts/check_upstream_drift.py`**:

```python
#!/usr/bin/env python
"""Detect upstream Qwen3-TTS drift vs docs/upstream-baseline.json.

Checks: upstream main SHA, qwen-tts PyPI version, repo file tree
(checkpoints/finetuning), finetuning/sft_12hz.py blob hash, README.md
(vLLM instructions) hash, and the installed package's public generate API
signatures. Never marks new upstream capability as Radeon-compatible --
drift output only demands revalidation."""
from __future__ import annotations
import argparse, inspect, json, sys, urllib.request
from pathlib import Path

UPSTREAM = "QwenLM/Qwen3-TTS"
PYPI = "https://pypi.org/pypi/qwen-tts/json"

def _get(url: str):  # returns bytes; raises with clear message on network failure
    req = urllib.request.Request(url, headers={"User-Agent": "qwen3-tts-rocm-drift-check"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

def _gh_api(path: str):
    return json.loads(_get(f"https://api.github.com/repos/{UPSTREAM}/{path}"))

def upstream_sha() -> str:
    return _gh_api("commits/main")["sha"]

def pypi_version() -> str:
    return json.loads(_get(PYPI))["info"]["version"]

def repo_file_hashes() -> dict[str, str]:
    tree = _gh_api("git/trees/main?recursive=1")["tree"]
    return {e["path"]: e.get("sha", "") for e in tree if e["type"] == "blob"}

def generate_api_surface() -> dict[str, str]:
    from qwen_tts import Qwen3TTSModel
    out = {}
    for name in dir(Qwen3TTSModel):
        if name.startswith("generate") or name in {"create_voice_clone_prompt"}:
            try:
                out[name] = str(inspect.signature(getattr(Qwen3TTSModel, name)))
            except (TypeError, ValueError):
                out[name] = "<no-signature>"
    return out

def collect() -> dict:
    return {
        "upstream_sha": upstream_sha(),
        "pypi_version": pypi_version(),
        "file_hashes": repo_file_hashes(),
        "api_surface": generate_api_surface(),
    }

def diff_state(baseline: dict, current: dict) -> dict:
    d = {}
    for key in ("upstream_sha", "pypi_version"):
        if baseline.get(key) != current[key]:
            d[key] = {"baseline": baseline.get(key), "current": current[key]}
    for key in ("file_hashes", "api_surface"):
        old, new = baseline.get(key, {}), current[key]
        if set(old) != set(new) or any(old[k] != new[k] for k in old):
            d[key] = {
                "added": sorted(set(new) - set(old)),
                "removed": sorted(set(old) - set(new)),
                "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
            }
    return d

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="docs/upstream-baseline.json")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()
    current = collect()
    if args.update_baseline:
        Path(args.baseline).write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print("BASELINE UPDATED")
        return 0
    baseline = json.loads(Path(args.baseline).read_text())
    d = diff_state(baseline, current)
    if not d:
        print("UNCHANGED")
        return 0
    print("UPSTREAM DRIFT DETECTED — revalidation required (new upstream capability is NOT auto-marked Radeon-compatible):")
    print(json.dumps(d, indent=2))
    return 1

if __name__ == "__main__":
    sys.exit(main())
```

(The delivered file is exactly the code above — stdlib only, `Path` imported at top.)
- [ ] **Step 15.2: Create the baseline** — `.venv/bin/python scripts/check_upstream_drift.py --update-baseline` then re-run to confirm `UNCHANGED` exit 0 (network required; on failure record `BLOCKED: network` per honest-boundary rule and retry later in the task).
- [ ] **Step 15.3: CPU unit tests** `tests/test_check_upstream_drift.py` — `diff_state` on synthetic dicts: identical → `{}`; changed SHA/version → both reported; file_hashes add/remove/change → exact sets; api_surface signature change → `changed`. Mock nothing network-level (tests only cover `diff_state`); no `pytestmark` markers.
- [ ] **Step 15.4: Optional scheduled Action** — add `upstream-drift` job to `.github/workflows/` (weekly cron, CPU-only, runs the script, fails the job on drift). Skip if it complicates the existing CPU workflow; decision recorded in the task report.
- [ ] **Step 15.5: Verifier gate** — criteria: script stdlib-only; `UNCHANGED`/`DRIFT` outputs exact; exit codes 0/1; baseline committed and reproducible; tests green; drift never auto-greens anything (wording in output).
- [ ] **Step 15.6: Commit**

```bash
git add scripts/check_upstream_drift.py docs/upstream-baseline.json tests/test_check_upstream_drift.py .github/workflows/ evidence/README.md docs/superpowers/reports/
git commit -m "chore(upstream): add upstream drift detection (SHA/PyPI/tree/API surface)"
git push origin main
```

---

### Task 16: Claims audit wave (brief 17)

**Files:**
- Create: `evidence/claims-audit-2026-09-21.md`
- Modify: every file the audit corrects (README.md, README_CN.md, CHANGELOG.md, docs/, evidence/README.md, pyproject.toml, `src/qwen3_tts_rocm/__init__.py`, tests as needed)

**Interfaces:**
- Consumes: the full evidence index; Task 9 PR state.
- Produces: audit table (term → hit → verdict → action) + the corrected files.

- [ ] **Step 16.1: Grep the vocabulary** — `validated|supported|all models|all Radeon|zero patch|zero patches|fine-tuning|finetuning|vLLM|streaming|305|0.2\.0|gfx1151|gfx1100` across README.md, README_CN.md, CHANGELOG.md, docs/**, evidence/README.md, src/**, tests/**, pyproject.toml. Every hit judged against evidence; enforce `zero committed downstream upstream-source patches` ≠ `every official workflow runs unchanged`; fine-tuning wording reflects PR status exactly; "305" updated to the true current count (recount: `.venv/bin/python -m pytest --co -q | tail -1`); version agreement README/README_CN/CHANGELOG/docs/pyproject/`__version__`/tests.
- [ ] **Step 16.2: Fix findings** — one docs commit; genuine code/test inconsistencies fixed in the same commit with the fix noted in the audit file.
- [ ] **Step 16.3: Verifier gate** — criteria: audit table covers every grep hit; each fix evidence-backed; version agreement verified programmatically (grep all version strings and compare); no claim stronger than evidence remains.
- [ ] **Step 16.4: Commit**

```bash
git add -A && git commit -m "docs: claims audit wave — every public claim evidence-accurate for v0.2.0"
```

---

### Task 17: Full regression (brief 18)

**Files:**
- Create: `evidence/final-regression-<date>.txt`

**Interfaces:**
- Consumes: the entire validated program.
- Produces: the pre-release regression record.

- [ ] **Step 17.1: Run everything on gfx1151, serialized**: `.venv/bin/python -m pytest -m 'not gpu' -q` (CPU suite) → `.venv/bin/python -m pytest -m gpu -q` (full GPU suite: parity, 0.6B, 1.7B, tokenizer, Voice Studio, multilingual, voice workflow) → benchmark smoke (`.venv/bin/python scripts/benchmark.py`, small) → fine-tuning execution smoke (the documented workaround path — still the truthful current state for published 0.1.1) → vLLM benchmark path re-run (one prompt pair from Task 12's manifest to confirm the A/B path still executes) → lint (`ruff check . && ruff format --check .` or the repo's configured linter — match existing CI) → package build (`python -m build` or the repo's existing build command; confirm wheel/sdist contents unchanged upstream-wise).
- [ ] **Step 17.2: Zero-patch audit** — `grep -rn "qwen_tts" src/ | grep -v loader` review: no monkey-patching, no vendored upstream code (`git ls-files | grep -i upstream` empty except docs/evidence references); `.upstream/Qwen3-TTS` pristine check; `git status --porcelain` clean after committing evidence.
- [ ] **Step 17.3: Archive** everything (commands, exit codes, summaries) into the evidence file.
- [ ] **Step 17.4: Verifier gate** — criteria: every listed suite green or honestly BLOCKED with reason; zero-patch audit clean; working tree clean.
- [ ] **Step 17.5: Commit**

```bash
git add evidence/final-regression-*.txt evidence/README.md docs/superpowers/reports/
git commit -m "test: final full regression on gfx1151 before v0.2.0"
```

---

### Task 18: Closure doc + final independent release verifier (brief 19)

**Files:**
- Create: `docs/radeon-reference-closure-v0.2.md`, `docs/superpowers/reports/rc02-task-19-final-verdict.md`

**Interfaces:**
- Consumes: everything the program produced.
- Produces: the final deliverable doc + the PASS/FAIL that gates the release.

- [ ] **Step 18.1: Write `docs/radeon-reference-closure-v0.2.md`** with the brief's 16 sections: North Star; Final repository SHA; Upstream SHA; Upstream #372 (reproduction/root cause/local fix/repeated validation/verifier verdict/PR URL/PR status); Capability matrix; Hardware matrix (gfx1151 only, second arch explicitly "deferred — no second-architecture hardware available at execution time"); gfx1151 results; second-architecture results (deferred statement); Fine-tuning status (exact PR-state wording); GPU CI status (BLOCKED ON RUNNER INFRASTRUCTURE); vLLM-Omni comparison (summary + link); Quality benchmark status; Upstream drift protection; Zero-patch audit; Test summary; Independent verification reports (index); Remaining known gaps; v0.2.0 release URL (placeholder `pending Task 19` is allowed HERE ONLY, replaced in Task 19).
- [ ] **Step 18.2: Dispatch the FINAL fresh verifier** with the Verification Protocol contract and the single question: "Is Qwen3-TTS-ROCm v0.2 ready to be described as a Radeon reference without misleading users?" — inspecting: #372 work + current PR status (`gh pr view`), capability matrix, gfx1151 evidence, second-GPU gap disclosure, CI status, vLLM comparison, quality benchmark scope, zero-patch wording, release notes draft (Task 19 body), version consistency, final test results. **No v0.2.0 tag if FAIL.**
- [ ] **Step 18.3: Commit**

```bash
git add docs/radeon-reference-closure-v0.2.md docs/superpowers/reports/rc02-task-19-final-verdict.md
git commit -m "docs: radeon reference closure v0.2 final report + independent release verifier verdict"
```

---

### Task 19: Release v0.2.0 (brief 20) — HARD USER STOP before publication

**Files:**
- Modify: `docs/radeon-reference-closure-v0.2.md` (fill release URL), `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 18 PASS.
- Produces: the published release.

- [ ] **Step 19.1: Draft the release notes** — title `Qwen3-TTS-ROCm v0.2.0 — Radeon Reference Closure`; narrative leads with the North Star; factual summary list per spec (1.7B+0.6B validation, CustomVoice, VoiceDesign, Voice Clone, reusable prompts, Design→Clone→Reuse, 10-language matrix, tokenizer roundtrip, fine-tuning execution state, upstream #372 PR status, benchmark replication, GPU CI state, vLLM-Omni deployment comparison, quality benchmark foundation, streaming explicitly not claimed). Three prohibitions enforced: no "fixed upstream" unless merged; no "GPU CI live"; no second-GPU parity claims.
- [ ] **Step 19.2: ★ USER GATE ★** — show the user: final commit SHA, notes draft, verifier verdict. Explicit confirmation required.
- [ ] **Step 19.3: Publish (only after confirmation)**

```bash
git tag v0.2.0 <verified-SHA> && git push origin main --tags && \
gh release create v0.2.0 --title "Qwen3-TTS-ROCm v0.2.0 — Radeon Reference Closure" \
  --notes-file .work-upstream/release-notes-v020.md
```

- [ ] **Step 19.4: Fill the release URL** into the closure doc + CHANGELOG; final commit + push.

```bash
git add docs/radeon-reference-closure-v0.2.md CHANGELOG.md
git commit -m "docs: record v0.2.0 release URL (radeon reference closure complete)"
git push origin main
```

---

## Self-Review (done at plan time)

1. **Spec coverage:** brief 0→Task 0; 1→Task 1; 2→Task 2; 3+4→Task 3; 5→Task 4; 6A→Task 5; 6B/C→Task 6; 6D/E→Task 7; 7→Task 8; 8→Task 9; 9→Task 10; 12→Task 11; 13→Task 12; 14→Task 12 Step 12.6 (plan Task 13 is the folded marker); 15→Task 14; 16→Task 15; 17→Task 16; 18→Task 17; 19→Task 18; 20→Task 19. Brief 10–11 (second GPU): deferred per spec — no task, gap disclosed in Task 18. ✓
2. **Placeholders:** `<exact args from the 2026-09-20 smoke>` / `<same scale/args>` are deliberate executable pointers into an archived transcript (the executor extracts and reuses them verbatim — the scale decision is already recorded evidence, not an open design choice); `<date>` = execution date per evidence convention. Task 14 helper bodies are specified by their ACCEPTed-candidate APIs with the omission rule — the implementer writes real implementations, stubs are forbidden explicitly. ✓
3. **Type consistency:** `build_arg_parser()` introduced Task 3, consumed Tasks 4/5; evidence filenames consistent between File Structure and task steps; `rc02-task-*` report naming consistent. ✓
