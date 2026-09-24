# GPU CI runbook — self-hosted Radeon 8060S (`gfx1151`) regression runner

> **STATUS: LIVE — first green run 35857806038 on 2026-09-23 (gpu-short,
> 32 nodes, commit a3a8a75). Nightly gpu-short at 02:00 local (+08:00) =
> 18:00 UTC; runs when the validation host is powered/online at the window
> (a missed window can be re-dispatched manually).**
>
> The self-hosted runner (amd-HP-ZBook-Ultra, labels
> `[self-hosted, Linux, X64, radeon-gfx1151]`) is registered at repo scope
> and runs as a **user-level systemd service** with linger enabled — see
> [Runner installation](#runner-installation-done-2026-09-23--user-level-systemd--linger).
> The workflow [`.github/workflows/gpu-nightly.yml`](../../.github/workflows/gpu-nightly.yml)
> was hardened for real execution in commit `a3a8a75` (retrying `git fetch`
> sync replacing `actions/checkout` — this host's github.com link is
> TLS-flaky; absolute host `.venv`/`models/` paths; a provenance assertion
> that `PYTHONPATH=$GITHUB_WORKSPACE/src` wins over the host editable
> install). First real run green, every step:
> [`evidence/gpu-ci-first-green-2026-09-23.txt`](../../evidence/gpu-ci-first-green-2026-09-23.txt).
> One prior scheduled run (`35831353081`) failed at checkout **before**
> the hardening (relative-path + TLS defects) — history, not current state.

Audience: the maintainer of the private validation desktop (AMD Ryzen AI
Max+ PRO 395 / Radeon 8060S, `gfx1151`, ROCm 7.14.0, Ubuntu kernel
6.17.0-1032-oem). Internal development document — see
[development index](README.md).

## What the workflow is

`gpu-nightly.yml` defines two mutually exclusive jobs on the
`[self-hosted, radeon-gfx1151]` label:

| Job | When | Timeout | Content |
|---|---|---|---|
| `gpu-short` | nightly cron `0 18 * * *` (18:00 **UTC** = 02:00 **local** +08:00 — GitHub schedules in UTC, not runner-local time; scheduled runs can also be delayed by load, so treat dispatch as the reliable path) or `workflow_dispatch` with `suite=gpu-short`/default | 120 min | environment diagnostics, then 32 of the 38 GPU test nodes in three disjoint `-k` slices (below) |
| `full-weekly` | `workflow_dispatch` with `suite=full-weekly` only (no cron yet — flip the schedule on once a weekly cadence is actually wanted) | 720 min | all 38 GPU nodes + `scripts/verify_gpu.sh` stack sanity + `scripts/benchmark.py` RTF replication |

Unlike the CPU workflow (`ci.yml`, ubuntu-latest, hermetic per-job install),
the runner **installs nothing per job**: it reuses the persistent host
`.venv` and `models/` tree via absolute `HOST_REPO` paths. Both jobs fail
fast in the diagnostics step if `.venv/bin/python` or `models/` is missing.

As hardened in commit `a3a8a75` (after the first scheduled run
`35831353081` failed at checkout):

- **No `actions/checkout`** — the sync step is a plain
  `git fetch --depth=1 origin main` wrapped in a bounded retry loop
  (15 attempts, 60 s apart; this host's github.com link is intermittently
  TLS-flaky — GnuTLS -110 / connect timeouts — and checkout's 3 attempts
  were not enough). No third-party action runs at all.
- **Host assets by absolute path**: `HOST_REPO=/home/amd/Desktop/Qwen3-TTS-ROCm`
  (`.venv`) and `QWEN3_TTS_ROCM_MODELS_DIR` → the host `models/` tree —
  the checked-out workspace contains neither (both gitignored).
- **Code-under-test provenance**: every pytest/benchmark step sets
  `PYTHONPATH=$GITHUB_WORKSPACE/src`, which precedes the host venv's
  editable-install `.pth`; the diagnostics step asserts
  `qwen3_tts_rocm.__file__` resolves inside `/_work/` before any test
  runs — the run tests the pushed commit, not the host tree.
- **Concurrency**: workflow-level group `gpu-nightly` with
  `cancel-in-progress: false` — one GPU, so a new run queues behind an
  in-flight one instead of cancelling it.

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

## Runner installation (done 2026-09-23 — user-level systemd + linger)

The runner (name **amd-HP-ZBook-Ultra**, labels `[self-hosted, Linux, X64,
radeon-gfx1151]`) lives in `~/actions-runner/` and runs as a **user-level
systemd service** under the desktop's normal user `amd` — **not root**, no
`sudo` anywhere in the actual install (GPU access works through the user's
`render`/`video` group membership):

- Unit `~/.config/systemd/user/github-runner.service`:
  `ExecStart=/home/amd/actions-runner/run.sh`, `WorkingDirectory=/home/amd/actions-runner`,
  `Type=simple`, `Restart=on-failure`, `RestartSec=10`,
  `After=network-online.target`, `WantedBy=default.target`.
- `systemctl --user enable --now github-runner.service` — enabled + active.
- **Linger ON** (`loginctl enable-linger amd` → `Linger=yes`): the user
  manager starts at boot without a login session, so the runner survives
  reboots and runs the nightly window unattended.

The exact registration procedure as performed (for a clean reinstall):

```bash
# 1. Registration token (short-lived, repo scope). NOTE: this endpoint
#    REQUIRES an explicit POST --method:
TOKEN=$(gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/runners/registration-token \
        --method POST --jq .token)

# 2. Download + configure the runner (linux x64). NOTE: the runner
#    release tarball ships NO .sha256 sidecar file — if you want an
#    integrity check, take the digest from the release-asset API's
#    `digest` field (e.g. `gh api repos/actions/runner/releases/tags/<tag>
#    --jq '.assets[].digest'`) rather than looking for a sidecar.
cd ~/actions-runner
./config.sh --url https://github.com/AIwork4me/Qwen3-TTS-ROCm \
            --token "$TOKEN" \
            --labels radeon-gfx1151 \
            --unattended

# 3. Run as a USER-LEVEL systemd service (what was actually done):
mkdir -p ~/.config/systemd/user
#   write ~/.config/systemd/user/github-runner.service as described above,
#   then:
systemctl --user daemon-reload
systemctl --user enable --now github-runner.service
loginctl enable-linger "$USER"   # boot-persistent without a login session
```

Alternative (the upstream default, **not** what was done): the documented
`sudo ./svc.sh install && sudo ./svc.sh start` installs a system-level
service running as root. On this single-user desktop the user-level unit
is preferred: the runner needs no root, and systemd user units + linger
give the same boot persistence with a smaller privilege surface.

Then confirm the runner appears as *Idle* with the `radeon-gfx1151` label
under repo **Settings → Actions → Runners**, and do one manual
`workflow_dispatch` run (below) as the go-live test.

## Security posture (binding constraints, as actually deployed)

This is a self-hosted runner on a private desktop with direct access to
the GPU, the `models/` tree and the home directory. Treat it as
infrastructure, not as an ephemeral CI container. The reality of this
deployment — a **personal (non-org) public repository**:

- **Repo-scope registration** — the runner is registered against
  `AIwork4me/Qwen3-TTS-ROCm` only (the `config.sh --url` above), never at
  org scope. Org **runner-group** endpoints do not apply here at all:
  the account is personal, so there is no runner-group management surface
  to configure (the earlier draft's runner-group instructions were
  org-account boilerplate, not something this repo can set).
- **Containment = the workflow's trigger surface, which is deliberately
  minimal**: `gpu-nightly.yml` has only `schedule` + `workflow_dispatch`
  triggers — **no `pull_request`, no `push`** — and `workflow_dispatch`
  requires **write access** to the repository. Fork/outside contributors
  therefore cannot cause any code to execute on this runner; fork PRs
  never land on it.
- Keep `permissions: contents: read` in the workflow; never widen it for
  convenience. No third-party actions run at all (the sync step is plain
  git against the public repo URL, no token needed).
- **Residual risk, recorded honestly**: the repo's allowed-actions setting
  remains "all" (the default). The trigger-surface restriction above —
  not an action allowlist — is what actually contains this runner. If the
  repo ever accepts outside contributions routinely, revisit whether a
  self-hosted runner is acceptable at all (GitHub's guidance: self-hosted
  runners are for private repos / trusted contributors).

## Triggering

- **Manual (preferred, reliable)**:
  - Web: repo **Actions → gpu-nightly → Run workflow →** pick branch,
    `suite` = `gpu-short` (default) or `full-weekly`.
  - CLI: `gh workflow run gpu-nightly.yml --ref main -f suite=gpu-short`
    (then `gh run watch` or `gh run view --workflow gpu-nightly.yml`).
- **Nightly schedule**: cron `0 18 * * *` — 18:00 **UTC** = 02:00
  **local** (+08:00). GitHub evaluates cron in UTC and may start runs
  late under load. The desktop is a validation host, not a managed
  server: the nightly happens only if the machine is powered/online at
  the window — a missed window is not an incident and leaves no signal;
  re-dispatch manually (above) if a night matters. Scheduled workflows
  are also auto-disabled after 60 days of repo inactivity — re-enable if
  that trips.

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
  checkouts; prune periodically.
- **Never write tracked files from workflow steps**: the sync step's
  `git clean -qfdx` removes *untracked* files only — it does **not**
  revert modifications to *tracked* files. A step that modified a tracked
  file would leave the workspace permanently dirty, and later runs would
  test stale content instead of the fetched SHA (with `checkout` failing
  or silently keeping the modification). Any future step must leave the
  checkout untouched except for untracked scratch paths (which the next
  run's clean removes).
- **`benchmark-nightly.json` persists as a run artifact**: `full-weekly`
  writes it into the runner workspace (`evidence/benchmark-nightly.json`
  under `_work/…`) and, since 2026-09-24, an `if: always()` step uploads
  it via GitHub's first-party `actions/upload-artifact@v4` as the run
  artifact `benchmark-nightly-json` (365-day retention) — the workspace
  copy is still deleted by the next run's `git clean -qfdx`, but the run
  page keeps the JSON downloadable. The only action in the workflow is
  this GitHub-first-party upload, which runs after all test steps and
  touches no code. The JSON is deliberately NOT committed back into the
  repository; a nightly that should become evidence is downloaded from the
  run page and committed under a dated name by a human decision.
- **Timeouts**: 120 min (short) / 720 min (full) cover the observed suite
  (38 nodes ≈ 5–10 min warm) plus one cold model-load margin — the first
  green run took ~18 min wall including ~10 min of sync retries through a
  TLS-flaky window; if the nightly starts timing out, fix the cause — do
  not raise the cap silently.
- **First green run happened 2026-09-23** (run `35857806038`): this
  header, both READMEs' CI paragraphs and the GPU CI badge were flipped
  in the same docs-only change set; transcript
  `evidence/gpu-ci-first-green-2026-09-23.txt`.

## Go-live checklist (state as of 2026-09-23)

1. ✅ Prerequisites section green on the desktop.
2. ✅ Runner installed, registered, labeled `radeon-gfx1151`, Idle —
   user-level systemd service + linger (see above).
3. ✅ Trigger-surface containment verified: `schedule` +
   `workflow_dispatch` only, no `pull_request`; repo-scope registration
   on a personal account (no org runner-group surface applies).
4. ✅ One manual `workflow_dispatch` run of `gpu-short` → **green**
   (run `35857806038`, 2026-09-23, commit `a3a8a75`).
5. ☐ One manual `workflow_dispatch` run of `full-weekly` → green —
   **still pending**; schedule a convenient start time (it is long).
6. ✅ STATUS header + both READMEs' CI paragraphs flipped, evidence
   archived (`evidence/gpu-ci-first-green-2026-09-23.txt`), GPU CI badge
   added — the badge was added at step 4 with `full-weekly` (step 5)
   still unrun: it reflects the workflow's last-run status, so the first
   `full-weekly` result will show on it directly.
