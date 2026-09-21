# GPU CI runbook — self-hosted Radeon 8060S (`gfx1151`) regression runner

> **STATUS: BLOCKED ON RUNNER INFRASTRUCTURE — workflow validated locally,
> never executed on a runner. No GPU CI badge until a real run is green.**
>
> As of 2026-09-21 the workflow file
> [`.github/workflows/gpu-nightly.yml`](../../.github/workflows/gpu-nightly.yml)
> exists and is validated (YAML parse + every `-k` filter proven against the
> collected GPU test nodes — transcript:
> [`evidence/gpu-ci-prep-validation.txt`](../../evidence/gpu-ci-prep-validation.txt)),
> but **no self-hosted runner is registered** and the workflow has **never
> run on GitHub Actions**. This runbook is the prepare-only record of how to
> close that gap. Nothing on the public README claims live GPU CI.

Audience: the maintainer of the private validation desktop (AMD Ryzen AI
Max+ PRO 395 / Radeon 8060S, `gfx1151`, ROCm 7.14.0, Ubuntu kernel
6.17.0-1032-oem). Internal development document — see
[development index](README.md).

## What the workflow is

`gpu-nightly.yml` defines two mutually exclusive jobs on the
`[self-hosted, radeon-gfx1151]` label:

| Job | When | Timeout | Content |
|---|---|---|---|
| `gpu-short` | nightly cron `0 2 * * *` (02:00 **UTC** — GitHub schedules in UTC, not runner-local time; scheduled runs can also be delayed by load, so treat dispatch as the reliable path) or `workflow_dispatch` with `suite=gpu-short`/default | 120 min | environment diagnostics, then 32 of the 38 GPU test nodes in three disjoint `-k` slices (below) |
| `full-weekly` | `workflow_dispatch` with `suite=full-weekly` only (no cron yet — flip the schedule on once a weekly cadence is actually wanted) | 720 min | all 38 GPU nodes + `scripts/verify_gpu.sh` stack sanity + `scripts/benchmark.py` RTF replication |

Unlike the CPU workflow (`ci.yml`, ubuntu-latest, hermetic per-job install),
the runner **installs nothing per job**: it reuses the persistent host
`.venv` and `models/` tree. Both jobs fail fast in the diagnostics step if
`.venv/bin/python` or `models/` is missing.

## Prerequisites (on the desktop, before any runner step)

These are exactly the validated stack — the same environment every
`evidence/` transcript was produced on:

1. **ROCm userspace + GPU access**: `rocm-smi` on `PATH`, user in the
   `render`/`video` groups, `/dev/kfd` + `/dev/dri` accessible (see README
   Quick Start / `docker/README.md` for the group setup).
2. **The pinned wheel stack in `.venv`**: built by
   `bash scripts/install.sh` from a clean checkout of the branch under test
   (ROCm 7.14 torch wheels, Python 3.12, editable install of
   `qwen3-tts-rocm`). `scripts/verify_gpu.sh` must exit 0.
3. **Model weights**: `bash scripts/download_models.sh` so `models/` holds
   all six repositories (≈ 17 GB; sizes/provenance:
   `evidence/models-dl.txt`). `models/` is gitignored — the runner uses the
   host copy; the workflow's checkout does not fetch weights.
4. **The full GPU suite green locally**, once, on the exact commit being
   baselined: `.venv/bin/python -m pytest -m gpu -q` → 38 passed.

## Suite selection — filter → node mapping

Derived 2026-09-21 from `.venv/bin/python -m pytest -m gpu --co -q`
(38 nodes; full lists in the validation transcript). `-k` matches substrings
of the whole node ID, file path included — that is why `official_demo`
selects by file and `voice_clone` selects both clone files at once.

| Nightly capability | Workflow step / `-k` expression (all with `-m gpu`) | Nodes | Selected tests |
|---|---|---|---|
| Upstream parity (official demo builds, executes via Gradio closures, one real synthesis) + tokenizer | step 1: `official_demo or tokenizer` | 6 | `test_official_demo_parity.py::test_upstream_blocks_constructed`, `::test_official_closure_executes_via_gradio_fns`, `::test_real_synthesis_through_upstream_callback`; `test_tokenizer_codec.py::test_metadata_getters_populated`, `::test_encode_decode_roundtrip_sane_and_duration_bound`; `test_loader_all_models.py::test_tokenizer_load` |
| 1.7B + 0.6B CustomVoice | step 2: `custom_voice or voice_design or voice_clone` | 8 | `test_generate_custom_voice.py::{test_single, test_batch, test_instruct_changes_output, test_sampling_kwarg_passthrough_effect}` + `test_generate_custom_voice_06b.py::{test_single, test_batch, test_instruct_behavior, test_kwargs_passthrough}` |
| VoiceDesign | step 2 (same expression) | 4 | `test_voice_design.py::{test_single_sane, test_batch_of_two_sane, test_two_instructions_both_sane_and_distinct, test_unsupported_language_raises_valueerror}` |
| 1.7B + 0.6B Base clone, incl. reusable prompt | step 2 (same expression) | 10 | `test_voice_clone_workflow.py` + `test_voice_clone_06b.py`, each `::{test_clone_with_ref_text, test_clone_x_vector_only, test_create_prompt_reuse, test_save_load_roundtrip_parity, test_batch_clone}` |
| Voice Studio workflow + demo backend (GPU side) | step 3: `voice_workflow` | 4 | `test_voice_workflow.py::{test_design_voice_creates_prompt_and_preview, test_reuse_across_two_sentences, test_save_load_roundtrip_then_reuse, test_official_schema_preserved}` |

The three workflow steps (disjoint slices, 32 nodes total):

1. `-k "official_demo or tokenizer"` → **6** nodes.
2. `-k "custom_voice or voice_design or voice_clone"` → **22** nodes
   (8 CustomVoice + 4 VoiceDesign + 10 clone).
3. `-k "voice_workflow"` → **4** nodes.

Deliberately **not selected** by the nightly short suite (documented, never
silently claimed):

- The remaining **6 GPU nodes** — `test_loader_all_models.py::`
  `test_load_smoke_each_tts_model[...]` (5 params) and
  `test_testing_utils.py::test_gpu_fixture_smoke` — run only in
  `full-weekly`. Every short-suite step already loads the model it
  exercises, so the loader smokes add runtime, not coverage, to the nightly.
- **`scripts/validate_languages.py`** has no pytest node at all; a `-k
  "validate_languages"` filter selects 0 tests. It stays a manual/scripted
  matrix run (evidence: `multilingual-matrix.*`).
- **The keyword `studio`** selects 0 GPU-marked nodes: the Voice Studio /
  demo-backend unit tests (`test_voice_studio_*` in
  `tests/test_demo_backend.py`, plus `test_demo_ui.py`) are CPU-only with
  mocked models and are already covered by `ci.yml`. On GPU, the Voice
  Studio *workflow* is covered by step 3, and the demo *backend* execution
  is covered by step 1 (`test_official_demo_parity` drives the official
  demo's handler closures end to end, including one real synthesis).
- **Fine-tuning execution smoke**: `full-weekly` runs
  `scripts/verify_gpu.sh` (stack-level prerequisite: bf16 matmul + SDPA on
  gfx1151). The real fine-tuning run (upstream `prepare_data.py` +
  `sft_12hz.py` inside the gitignored `.upstream/` clone —
  `evidence/finetune-smoke-2026-09-20.txt`) is a manual procedure and is
  **not** executed by this workflow.

Filter refinements vs. the task-11 brief draft (why the shipped expressions
differ): the draft's `-k "parity or tokenizer"` also matched the two clone
`test_save_load_roundtrip_parity` nodes (8, not 6, and duplicating step 2);
`-k "demo or studio"` matched only the 3 parity nodes already in step 1
(`studio` matched nothing); `validate_languages` matched nothing; and the
benchmark flag is `--json-out` (the script's real flag), not `--json`. All
counts recorded in the validation transcript.

## Runner installation (the blocking step)

Never done yet — this is the exact procedure when it happens. Work in a
directory outside the repo tree, e.g. `~/actions-runner/`:

```bash
# 1. Registration token (short-lived, org/repo scope):
TOKEN=$(gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/runners/registration-token \
        --jq .token)

# 2. Download + configure the runner (linux x64):
cd ~/actions-runner
./config.sh --url https://github.com/AIwork4me/Qwen3-TTS-ROCm \
            --token "$TOKEN" \
            --labels radeon-gfx1151 \
            --unattended

# 3. Install as a service and start it:
sudo ./svc.sh install
sudo ./svc.sh start
./svc.sh status   # or: systemctl status actions.runner.*
```

Then confirm the runner appears as *Idle* with the `radeon-gfx1151` label
under repo **Settings → Actions → Runners**, and do one manual
`workflow_dispatch` run (below) as the go-live test.

## Label + fork security (binding constraints)

This is a self-hosted runner on a private desktop with direct access to the
GPU, the `models/` tree and the home directory. Treat it as
infrastructure, not as an ephemeral CI container:

- **Restrict the runner to this repository only** — register it at repo
  scope (the `config.sh --url` above), never at org scope.
- **Never enable it for public forks.** In repo **Settings → Actions →
  Runner groups**, keep the runner in the *Default* group with access
  limited to this repository, and leave **"Allow public repositories to
  use this runner group"** OFF. Fork PRs must never land on this runner —
  this is also why `gpu-nightly.yml` has **no `pull_request` trigger**
  (only `schedule` + `workflow_dispatch`; `workflow_dispatch` is
  repo-write-access only).
- Keep `permissions: contents: read` in the workflow; never widen it for
  convenience.
- The runner executes whatever is on the checked-out ref. If the repo ever
  accepts outside contributions routinely, revisit whether a self-hosted
  runner is acceptable at all (GitHub's guidance: self-hosted runners are
  for private repos / trusted contributors).

## Triggering

- **Manual (preferred, reliable)**:
  - Web: repo **Actions → gpu-nightly → Run workflow →** pick branch,
    `suite` = `gpu-short` (default) or `full-weekly`.
  - CLI: `gh workflow run gpu-nightly.yml --ref main -f suite=gpu-short`
    (then `gh run watch` or `gh run view --workflow gpu-nightly.yml`).
- **Nightly schedule**: cron `0 2 * * *` is evaluated by GitHub in **UTC**
  and may start late under load. If a specific local time matters, adjust
  the cron (host is UTC+8 → 02:00 local ≈ `0 18 * * *` UTC) when the
  runner goes live. Scheduled workflows are also auto-disabled after 60
  days of repo inactivity — re-enable if that trips.

## Maintenance

- **venv refresh**: after dependency/ROCm changes, rebuild with
  `bash scripts/install.sh` (or `pip install -e ".[dev]"` inside `.venv`)
  and re-run `scripts/verify_gpu.sh` + the short suite once manually. The
  workflow reuses the host `.venv` verbatim — it will happily run green
  tests against a stale stack, so refresh deliberately.
- **models refresh**: `bash scripts/download_models.sh <alias>` when
  upstream publishes new checkpoints; keep `models/` on disk with ≥ 30 GB
  free (the fine-tuning smoke alone needs headroom).
- **Disk**: runner work dirs (`~/actions-runner/_work/`) accumulate
  checkouts; prune periodically. `benchmark-nightly.json` (written by
  `full-weekly` into the checkout's `evidence/`) is a job artifact —
  retrieve it from the run page, then commit it under a dated name if it
  should become evidence.
- **Timeouts**: 120 min (short) / 720 min (full) cover the observed suite
  (38 nodes ≈ 5–10 min warm) plus one cold model-load margin; if the
  nightly starts timing out, fix the cause — do not raise the cap silently.
- **After any first green run**: update this header, the README CI
  paragraph, and only then consider adding a GPU CI badge.

## Go-live checklist (in order)

1. Prerequisites section green on the desktop.
2. Runner installed, registered, labeled `radeon-gfx1151`, Idle.
3. Runner-group fork policy verified OFF for public repos.
4. One manual `workflow_dispatch` run of `gpu-short` → green.
5. One manual `workflow_dispatch` run of `full-weekly` → green (schedule a
   convenient start time; it is long).
6. Update this file's STATUS header, README's CI paragraph, and
   `evidence/` (archive both run transcripts). Badge only after 4–5.
