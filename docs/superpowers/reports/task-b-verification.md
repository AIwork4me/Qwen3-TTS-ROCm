# Task B verification — gpu-nightly.yml hardening

Verifier: independent (assumed implementer wrong until proven right). Date: 2026-09-21.
Method: execution-first — every check below that can be run was run by me from
fresh extractions/fixtures, not trusted from the implementer's report. Nothing
outside this report file was modified; no workflow was dispatched; no sudo; no
subagents.

## Verdict

**PASS** — all 8 criteria confirmed by execution; 1 minor out-of-scope
documentation staleness noted (runbook §Triggering still cites the old cron).

## Criteria checklist

| # | Criterion | Result |
|---|-----------|--------|
| 1 | YAML valid; jobs/labels/timeouts/-k slices byte-preserved vs parent | PASS |
| 2 | Cron `0 18 * * *` + local-night comment; workflow_dispatch preserved | PASS |
| 3 | `concurrency: {group: gpu-nightly, cancel-in-progress: false}` at workflow level | PASS |
| 4 | actions/checkout fully replaced; sync logic read + `bash -n` + offline stub sims re-run | PASS |
| 5 | PYTHONPATH outranks editable .pth — decoy experiment reproduced independently | PASS |
| 6 | Foreign-cwd (/tmp) pytest collection smoke, nodes > 0 | PASS (6/351) |
| 7 | No relative `.venv/` or bare `models/` in run: steps; MODELS_DIR env; provenance assertion | PASS |
| 8 | a3a8a75 == origin/main (gh api); working tree clean | PASS |

## Commands and exit codes (mine, not the implementer's)

- `yaml.safe_load` on `.github/workflows/gpu-nightly.yml` (repo `.venv` python)
  — OK. Parsed: `cron == '0 18 * * *'`; `workflow_dispatch.inputs.suite`
  preserved (`gpu-short | full-weekly`, default `gpu-short`);
  `concurrency == {'group': 'gpu-nightly', 'cancel-in-progress': False}` at
  workflow level (sibling of `on:`/`jobs:`, not inside a job);
  `defaults.run.shell: bash`; both jobs `runs-on: ['self-hosted',
  'radeon-gfx1151']`; timeouts 120 (gpu-short) / 720 (full-weekly).
- Programmatic parent-vs-current comparison (parent extracted via
  `git show a3a8a75~1:.github/workflows/gpu-nightly.yml`):
  - `-k` filter set identical: `official_demo or tokenizer`, `custom_voice or
    voice_design or voice_clone`, `voice_workflow` (+ bare `-m gpu -q` in
    full-weekly). Job names, `if:` expressions, `runs-on`, `timeout-minutes`
    all equal. Test-step names identical.
  - `git show a3a8a75 --stat`: only `.github/workflows/gpu-nightly.yml`
    (+133/−27). Full diff read line-by-line: changes are exactly cron,
    concurrency/defaults, job `env:`, sync-step substitution, diagnostics
    hardening (absolute paths + provenance), interpreter absolutization,
    header comments. The suite-selection comment block is untouched (context
    lines only in the diff). Only plumbing differs — confirmed.
- `grep -n "uses:" .github/workflows/gpu-nightly.yml` → exit 1 (no matches):
  actions/checkout fully replaced in BOTH jobs (both jobs use the identical
  `Sync repository` run step; the two sync bodies are byte-identical, as are
  the two diagnostics bodies).
- Sync-step reading (from the YAML extraction, not the report):
  `set -euo pipefail`; `if [[ ! -d .git ]] then git init -q; git remote add
  origin <url>` (first-run path); bounded retry `for attempt in $(seq 1 15)`
  with `sleep 60` between attempts (skipped after attempt 15 → 14 min worst
  case, matching the error message's "~14 min"); on success `git clean -qfdx
  && git checkout -q FETCH_HEAD` and `echo "SYNCED_TO $(git rev-parse HEAD)"`
  (SHA recorded); on 15/15 failure prints a runbook-pointing ERROR line and
  `exit 1`. All required elements present.
- `bash -n` on (a) the verbatim sync body extracted from HEAD YAML, (b) the
  implementer's `sync.sh`, (c) `diag.sh`, (d) my provenance driver — all OK.
  Diff (a) vs implementer's `sync2.sh`: byte-identical; `sync.sh` differs from
  `sync2.sh` only in the two documented offline edits (origin URL → local
  path, `sleep 60` → `sleep 0.2`) + one trailing newline. Harness is faithful.
- Offline stub simulation RE-RUN with MY OWN stubs and fixtures (fresh
  /tmp/taskb-verify-* scratch, local clone as origin; stub `git` failed fetch
  with simulated `GnuTLS -110` per a counter file, delegated all else to
  /usr/bin/git):
  - SIM A fresh workspace, fetch fails 2x: attempts 1–2 retried, attempt 3
    fetched; `git init` + `remote add` path exercised; `SYNCED_TO
    a3a8a75971b5d7ae35bbb5a8309935f75b4951b5` printed (== local origin's
    main HEAD); tracked tree populated (workflow file present); EXIT=0;
    exactly 3 fetch invocations.
  - SIM B existing `.git` + untracked `dirty-leftover.txt`: EXIT=0, dirty
    file removed by `git clean -qfdx`, tracked tree intact.
  - SIM C fetch never succeeds: exactly 15/15 fetch invocations, clear ERROR
    line with runbook pointer, EXIT=1 (3 s wall with the shortened sleep).
- Criterion 5 decoy experiment (my own, per spec): created
  `/tmp/decoy-src-verify/qwen3_tts_rocm/__init__.py` with marker
  `TASKB-VERIFIER-DECOY-2026-09-21`; from /tmp,
  `PYTHONPATH=/tmp/decoy-src-verify <repo>/.venv/bin/python -c "import
  qwen3_tts_rocm; print(.__file__)"` →
  `/tmp/decoy-src-verify/qwen3_tts_rocm/__init__.py` and the marker string
  printed (exit 0). Control with `PYTHONPATH=` resolves to
  `/home/amd/Desktop/Qwen3-TTS-ROCm/src/...` (editable). sys.path probe:
  decoy at index 1, editable `src` at index 6 — PYTHONPATH precedes the .pth
  append, decisively. Decoy removed afterwards (verified gone).
- Criterion 6: from /tmp, `PYTHONPATH=<repo>/src <repo>/.venv/bin/python -m
  pytest <repo>/tests --co -q -m gpu -k "official_demo or tokenizer"` →
  `6/351 tests collected (345 deselected) in 4.32s`, exit 0 (nodes > 0,
  matches the header's slice-1 count of 6).
- Provenance assertion (verbatim command from the diagnostics body) both
  ways: workspace containing `/_work/` in path → prints the workspace src
  path, exit 0; `GITHUB_WORKSPACE=/tmp/taskb-decoy` (no `/_work/`) → exit 1
  with the "host editable install won over PYTHONPATH" ERROR. Assertion
  present and identical in BOTH jobs' diagnostics steps.
- Criterion 7 audit (parsed YAML, regex over every `run:` block): zero
  relative `.venv/` references, zero bare `models/` references (all
  venv/models access via `$HOST_REPO` / `$QWEN3_TTS_ROCM_MODELS_DIR`);
  `env: HOST_REPO` + `QWEN3_TTS_ROCM_MODELS_DIR` present in both jobs.
  `scripts/benchmark.py` / `evidence/` are relative but intentionally so
  (fetched-SHA code under cwd=$GITHUB_WORKSPACE — correct).
- Criterion 8: `gh api repos/AIwork4me/Qwen3-TTS-ROCm/commits/main --jq
  '.sha'` → `a3a8a75971b5d7ae35bbb5a8309935f75b4951b5` (attempt 1, no retry
  needed) == local HEAD. `git status --porcelain` → empty (clean).

## Problems found

1. (Minor, out of Task B's file scope) `docs/development/gpu-ci-runbook.md`
   §Triggering still says the schedule is cron `0 2 * * *` and frames
   `0 18 * * *` as a future adjustment "when the runner goes live". The
   workflow is now at `0 18 * * *` and the runner is live, so that paragraph
   is stale. Not a Task B criterion failure (the task's file scope was the
   workflow), but it should be fixed in a follow-up docs pass.
2. No other problems. The implementer's scratch harness was faithful to the
   committed YAML (byte-verified), and every reported experimental result
   reproduced exactly under my own fixtures.

## Justification

Every claim in the implementer's report that is checkable offline was
re-executed independently — YAML structure, parent-diff preservation of all
test plumbing, checkout replacement, sync-step retry/bounds/clean/checkout/
rev-parse/exit semantics (3 stubbed simulations), the PYTHONPATH-vs-editable
decoy experiment, the foreign-cwd collection smoke, the provenance assertion
in both polarities, and origin/main identity — and all reproduced exactly.
