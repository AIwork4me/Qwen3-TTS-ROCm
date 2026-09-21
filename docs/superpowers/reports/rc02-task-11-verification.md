# Task 11 independent verification — GPU CI preparation (BLOCKED state)

Verifier role: adversarial release verifier (assume implementer wrong until
proven right). No subagents dispatched. Nothing modified except this report
file. All commands below were executed by the verifier on 2026-09-21 in
`/home/amd/Desktop/Qwen3-TTS-ROCm` unless noted.

## 1. Verdict

**PASS**

## 2. Exact commit / working-tree SHA reviewed

- Reviewed commit: `5b5168d57ed3d2c14b6471889c1ff378fc4d2d1c` (`ci: add Radeon
  GPU nightly regression workflow (BLOCKED ON RUNNER INFRASTRUCTURE)`,
  2026-09-21 16:23:14 +0800).
- `git rev-parse HEAD` = `5b5168d57ed3d2c14b6471889c1ff378fc4d2d1c`;
  `git rev-parse origin/main` = same. Working tree clean
  (`git status`: "nothing to commit, working tree clean").
- Remote confirmed live: `git ls-remote origin refs/heads/main` →
  `5b5168d57ed3d2c14b6471889c1ff378fc4d2d1c` (exit 0). A repeat attempt
  failed with the environment's documented github.com:443 flake
  (`GnuTLS recv error (-110)`, exit 128) — network flake only, not a
  repo-state discrepancy; the successful query is the evidence of record.
- `git show --stat 5b5168d` file list matches the review package
  (`review-3bb9c0a..5b5168d.diff`) exactly: 6 files, 462 insertions, 1
  deletion.

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | YAML valid; legal cron; `[self-hosted, radeon-gfx1151]`; timeouts; `workflow_dispatch` input | PASS | Verifier's own `yaml.safe_load` + structural assertion script, exit 0 (see §5/§6) |
| 2 | Every `-k` filter maps to real tests; required capabilities genuinely selected or gaps documented | PASS | Independent re-collection: 38 total, 6+22+4=32 disjoint, complement 6; see §7 |
| 3 | Runbook completeness (registration, fork security, maintenance, dispatch, BLOCKED header) | PASS | Full read of `docs/development/gpu-ci-runbook.md`; all five elements present |
| 4 | No live-GPU-CI claim, no badge, no runner installed | PASS | Greps + host inspection; see §7 |
| 5 | No fine-tuning execution claim; correct benchmark flag | PASS | `scripts/verify_gpu.sh` + `scripts/benchmark.py` argparse inspected; see §7 |
| 6 | Evidence transcript with standard header, commands, outputs | PASS | Read + byte-diff of node lists vs verifier's fresh collection |
| 7 | Commit pushed, tree clean, CPU suite green | PASS | `git ls-remote` exit 0; `pytest -m 'not gpu' -q` → 267 passed, 38 deselected, exit 0 |

## 4. Files reviewed

- `/home/amd/Desktop/Qwen3-TTS-ROCm/.github/workflows/gpu-nightly.yml` (new, 108 lines)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/docs/development/gpu-ci-runbook.md` (new, 201 lines)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/evidence/gpu-ci-prep-validation.txt` (new, 140 lines)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-11-brief.md`
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-11-report.md`
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-3bb9c0a..5b5168d.diff`
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.github/workflows/ci.yml` (convention comparison: checkout pin, permissions, pytest style)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/README.md` (GPU CI state paragraph, badges)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/docs/development/README.md`, `evidence/README.md` (index rows)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/scripts/benchmark.py` (argparse), `scripts/verify_gpu.sh` (full read)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/docs/p0-parity-report.md` lines 160–178 (untouched cross-ref)
- `tests/test_demo_backend.py`, `tests/test_demo_ui.py` (marker audit for the "studio" claim)

## 5. Exact commands executed (verifier)

```
git status && git log --oneline -5 && git rev-parse HEAD && git rev-parse origin/main
git show --stat 5b5168d
git diff 3bb9c0a..5b5168d --stat
git ls-remote origin refs/heads/main                      # attempt 1: OK; attempt 2: 443 flake
.venv/bin/python <YAML structural assertion script>       # yaml.safe_load + assertions, heredoc
.venv/bin/python -m pytest -m gpu --co -q
.venv/bin/python -m pytest -m gpu --co -q -k "official_demo or tokenizer"
.venv/bin/python -m pytest -m gpu --co -q -k "custom_voice or voice_design or voice_clone"
.venv/bin/python -m pytest -m gpu --co -q -k "voice_workflow"
.venv/bin/python -m pytest -m gpu --co -q -k "parity or tokenizer"        # draft filter
.venv/bin/python -m pytest -m gpu --co -q -k "demo or studio"             # draft filter
.venv/bin/python -m pytest -m gpu --co -q -k "validate_languages"         # draft filter
.venv/bin/python -m pytest --co -q -k "studio"                            # all markers, no -m
.venv/bin/python -m pytest -m "not gpu and not requires_download" --co -q tests/test_demo_backend.py tests/test_demo_ui.py
.venv/bin/python -m pytest -m 'not gpu' -q                                # CPU regression suite
.venv/bin/python scripts/benchmark.py --help
grep -n 'json' scripts/benchmark.py ; cat scripts/verify_gpu.sh
grep -rniE 'badge|shields\.io|GPU CI (enabled|passing|green|live|running)|...' README.md docs/development/gpu-ci-runbook.md .github/workflows/gpu-nightly.md docs/development/README.md evidence/README.md
grep -rn 'BLOCKED ON RUNNER INFRASTRUCTURE' README.md docs/ .github/ evidence/gpu-ci-prep-validation.txt
grep -rn 'gpu-nightly' --include='*.md' --include='*.yml' .               # stray-reference scan
grep -n 'pytestmark|pytest.mark' tests/test_demo_backend.py tests/test_demo_ui.py
ls -la ~/actions-runner ; systemctl list-units 'actions.runner*' --no-pager ; ps aux | grep -i '[r]unner'
diff <transcript node lists> <verifier fresh collection lists>            # /tmp/t_*.txt vs /tmp/v_*.txt
timeout 30 rocm-smi --showproductname --showuse --showmeminfo vram        # workflow diagnostics command
```

## 6. Exit codes

| Command | Exit |
|---|---|
| YAML parse + structural assertions | 0 |
| `pytest -m gpu --co -q` | 0 (38/305 collected, 267 deselected) |
| `--co -q -k "official_demo or tokenizer"` | 0 (6/305) |
| `--co -q -k "custom_voice or voice_design or voice_clone"` | 0 (22/305) |
| `--co -q -k "voice_workflow"` | 0 (4/305) |
| `--co -q -k "parity or tokenizer"` (draft) | 0 (8/305 — over-match confirmed) |
| `--co -q -k "demo or studio"` (draft) | 0 (3/305 — `studio` matches 0 GPU nodes confirmed) |
| `--co -q -k "validate_languages"` (draft) | 5, "no tests collected (305 deselected)" — 0 nodes confirmed |
| `pytest -m 'not gpu' -q` (full CPU suite) | 0 — **267 passed, 38 deselected in 16.03s** (exactly the expected counts) |
| `git ls-remote origin refs/heads/main` | 0 (attempt 1; attempt 2 flaked with 128) |
| `rocm-smi --showproductname --showuse --showmeminfo vram` | 0 (gfx1151, Radeon 32 GB VRAM) |

(Pytest exit 5 for an intentionally-zero-match draft filter is the expected
"no tests collected" code and supports, rather than contradicts, the
implementer's documented reason for dropping that keyword.)

## 7. Runtime evidence inspected

**Criterion 1 — YAML.** Verifier's own `yaml.safe_load` succeeded and
assertions passed: trigger keys exactly `{schedule, workflow_dispatch}` (no
`pull_request`/`push`); single cron `0 2 * * *` with 5 fields and legal
minute/hour values; `workflow_dispatch.inputs.suite` = description
"gpu-short | full-weekly", required false, default "gpu-short";
`permissions: contents: read`; both jobs `runs-on ==
['self-hosted', 'radeon-gfx1151']`; timeouts 120 (gpu-short) / 720
(full-weekly); `if` conditions as specified; checkout pinned to
`3d3c42e5aac5ba805825da76410c181273ba90b1`, byte-identical to the pin in
`ci.yml` (so the "matches ci.yml convention" claim is real).

**Criterion 2 — filters.** Fresh independent collection reproduced the
implementer's numbers exactly: 38 `-m gpu` nodes; the three shipped filters
select 6, 22, and 4 nodes; pairwise overlaps s1∩s2 = s1∩s3 = s2∩s3 = 0
(disjoint); union 32; complement (38−32 = 6) =
`test_load_smoke_each_tts_model[base|base-0.6b|custom-voice|custom-voice-0.6b|voice-design]`
+ `test_testing_utils.py::test_gpu_fixture_smoke`, which matches the
documented full-weekly-only set. Capability coverage inside the nightly 32:
upstream parity (3 nodes incl. `test_real_synthesis_through_upstream_callback`),
tokenizer (2 codec + `test_tokenizer_load`), CustomVoice 1.7B (4) and 0.6B
(4), VoiceDesign (4), Base clone 1.7B (5) and 0.6B (5) — each clone suite
including `test_create_prompt_reuse` (reusable prompt), Voice Studio
workflow (4: design→preview→save→reuse). Gaps are documented, not silent:
`studio` selects 0 GPU nodes (verifier confirmed: `test_demo_backend.py` /
`test_demo_ui.py` have zero `mark.gpu` occurrences and all 100 of their
nodes collect under ci.yml's `-m "not gpu and not requires_download"`, so
they are covered by the CPU workflow, exactly as the runbook states);
`validate_languages` selects 0 nodes (it is a script). The brief's draft
filters were also re-derived: `parity or tokenizer` → 8 (over-match into the
clone step), `demo or studio` → 3 (subset of step 1), `validate_languages`
→ 0 — all matching the documented refinement rationale. The one
brief-mandated wording nit — "Voice Studio backend path" — is satisfied on
the GPU side by `test_voice_workflow.py` plus step 1's end-to-end demo
handler execution, with the CPU-side backend tests explicitly assigned to
ci.yml in runbook §Suite selection, workflow header comments, the
implementer report, and transcript block [5].

**Criterion 3 — runbook.** `docs/development/gpu-ci-runbook.md` contains:
status header "BLOCKED ON RUNNER INFRASTRUCTURE — workflow validated
locally, never executed on a runner. No GPU CI badge until a real run is
green" (lines 3–13); runner registration with the
`gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/runners/registration-token`
token flow, `./config.sh --url … --labels radeon-gfx1151 --unattended`,
`sudo ./svc.sh install` / `start` / `status` (lines 109–134); fork/label
security section (repo-scope only, runner-group public-repo access OFF, no
PR trigger, contents:read, lines 136–156); maintenance (venv/models
refresh, disk, timeout policy, lines 171–190); dispatch instructions (web
UI + `gh workflow run`, UTC cron correction incl. the UTC+8 note, 60-day
schedule auto-disable, lines 158–169); plus a go-live checklist gating any
badge on real green runs.

**Criterion 4 — no badge / no live claim / no runner.** README badges are
only the pre-existing CPU `ci.yml` badge and Release/License/Python/ROCm/
Hardware shields badges — no `gpu-nightly` badge anywhere. Every "GPU CI"
mention in README.md / runbook / development index is BLOCKED or negative
wording ("prepared but BLOCKED ON RUNNER INFRASTRUCTURE", "never executed
on GitHub Actions", "deliberately no GPU CI badge"). BLOCKED wording
confirmed present in all four applicable places (workflow header comment,
runbook header, README GPU-CI-state paragraph, evidence transcript header)
plus the development index row. Host inspection: no `~/actions-runner`
directory, zero `actions.runner*` systemd units, no runner processes. The
diagnostics command itself is real on this host (`rocm-smi … vram` exit 0,
GFX Version gfx1151).

**Criterion 5 — honesty of steps.** `scripts/verify_gpu.sh` inspected in
full: it asserts a ROCm 7.14 torch build, device availability, gfx1151
arch, a 512×512 bf16 matmul, an SDPA call, and a torchaudio import — a
stack sanity check, not fine-tuning. The workflow step is named
"GPU stack sanity (fine-tune prerequisite)" with a comment stating the real
fine-tuning smoke is a manual procedure NOT run by the workflow — honest.
`scripts/benchmark.py` argparse (line 423) defines `--json-out`
(`--json` does not exist); `--help` confirms; the workflow invokes
`--json-out evidence/benchmark-nightly.json`. Correct flag used.

**Criterion 6 — transcript.** `evidence/gpu-ci-prep-validation.txt` carries
a standard header (timestamp 2026-09-21T08:21:10Z, host amd-HP-ZBook-Ultra,
git HEAD `3bb9c0a…`, purpose, BLOCKED banner), `cmd:` lines, verbatim
pytest outputs with node IDs, `PYTEST_EXIT` codes, the draft-filter counts
(8/3/0), the `--json-out` help excerpt, and a verdict block. The four node
lists in the transcript (full 38, 6, 22, 4) are **byte-identical** to the
verifier's fresh collections (diff of sorted lists: IDENTICAL ×4).

**Criterion 7 — commit/tree/suite.** HEAD = origin/main tracking ref =
remote main = `5b5168d57ed3d2c14b6471889c1ff378fc4d2d1c`; tree clean.
CPU suite run by the verifier: `267 passed, 38 deselected in 16.03s`,
exit 0 — exactly the expected 267/38.

## 8. Regression tests

- `.venv/bin/python -m pytest -m 'not gpu' -q` → **267 passed, 38
  deselected**, exit 0 (expected counts matched exactly; no failures, no
  skips on this GPU-equipped host, consistent with the HIP-gated test
  running locally rather than being skipped).
- GPU collection regression: 38/305 `-m gpu` nodes collected, exit 0,
  matching the pre-change inventory in the transcript and README (305
  total = 267 + 38). (GPU execution itself is out of scope for this
  prepare-only task; collection is the executable evidence the brief
  demands, and it was re-derived from scratch.)

## 9. Claims audit (implementer report vs verifier evidence)

| Implementer claim | Verifier finding |
|---|---|
| YAML valid, required shape | Confirmed by independent parse + assertions (exit 0) |
| 38 GPU nodes; 6/22/4 = 32 disjoint; 6-node complement | Confirmed exactly; overlaps all zero; complement identical |
| Draft filters 8/3/0 and why | Confirmed exactly (incl. pytest exit 5 for the 0-match filter) |
| `studio` matches 0 GPU nodes; demo backend/ui CPU-only and covered by ci.yml | Confirmed (0 `mark.gpu` in those files; 100 nodes collect under ci.yml's marker expression) |
| `--json-out` is the real flag | Confirmed (argparse line 423; `--json` nonexistent; `--help` output matches transcript) |
| verify_gpu.sh is stack sanity, not fine-tune; step name honest | Confirmed by full script read |
| Checkout pin matches ci.yml | Confirmed byte-identical pin `3d3c42e5…` |
| Pushed: remote main = 5b5168d | Confirmed via successful `git ls-remote` (one later attempt flaked on the known 443 issue; not a discrepancy) |
| Tree clean | Confirmed |
| No runner installed, nothing executed on GitHub Actions | Confirmed by host inspection; and no badge/live-claim wording found anywhere applicable |
| p0-parity-report.md not touched, cross-ref still accurate | Checked lines 160–178: CPU-only first-verified-run record under its as-of framing; no live-GPU-CI claim, no contradiction with BLOCKED state |
| Transcript regenerated clean before commit | Consistent with the archived file (no duplicated `--co` artifacts; node lists byte-identical to fresh collection) |

## 10. Problems found

None blocking. Three non-blocking observations:

1. Transcript block [4] (union/disjointness cross-check) archives its
   output but not the generating code verbatim. Mitigated: the verifier
   re-derived union/disjointness/complement independently with identical
   results.
2. The transcript header records HEAD `3bb9c0a` (validation necessarily ran
   before the commit `5b5168d` that archives the file). Inherent to
   self-archived evidence; the file is committed in `5b5168d` per the diff.
3. `git ls-remote` succeeded once and flaked once (documented github.com:443
   environment flake). The successful query establishes the push; the flake
   is a network condition, not a repository-state issue.

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence the verifier produced
itself rather than trusting the implementer: the YAML was re-parsed with
structural assertions passing (exit 0); all three shipped `-k` filters plus
the three draft filters were re-collected from scratch with counts and node
lists identical to the archived transcript (byte-identical diffs), with
disjointness and the 6-node complement independently computed; the required
nightly capabilities (parity, tokenizer, both CustomVoice models,
VoiceDesign, both Base clones, reusable prompt, Voice Studio workflow) are
each present in the collected selections, and the two keyword gaps (`studio`,
`validate_languages`) are documented in runbook, workflow comments, report,
and transcript rather than silently dropped; the runbook contains all five
required elements including the BLOCKED header; no GPU badge or live-GPU-CI
claim exists anywhere applicable (BLOCKED wording present in all four
places) and no runner is installed on the host; `verify_gpu.sh` and the
benchmark CLI were source-verified for honest step naming and the correct
`--json-out` flag; the commit is the clean working-tree HEAD and confirmed
on the remote; and the CPU suite was re-run by the verifier with exactly
the expected result (267 passed, 38 deselected, exit 0). No falsification
attempt succeeded.
