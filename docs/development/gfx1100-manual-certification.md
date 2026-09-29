# gfx1100 Manual GPU Certification Flow (v0.3 Task 13)

Status decision recorded 2026-09-29: **the v0.3 gfx1100 execution host is
NOT suitable as a self-hosted GitHub Actions runner**, because it is an
ephemeral containerized lab session (AMD one-click lab runtime with an
egress-restricted proxy) rather than a persistently online machine; it
cannot "remain online reliably" beyond the session. Per the v0.3 Task 13
brief, this document defines the **manual certification flow instead**, and
NO scheduled workflow is added for gfx1100 (nothing may sit permanently
queued for hardware that is not attached).

The historical `radeon-gfx1151` self-hosted runner and
`.github/workflows/gpu-nightly.yml` are untouched — their evidence remains
historical and is not renamed or rewritten.

## When to use this flow

Any time a gfx1100-class GPU (validated reference: Radeon Pro W7900D, 48 GB)
is available on a machine where the pinned stack prerequisites below hold.
The flow reproduces the exact v0.3 primary-gate evidence shape
(`evidence/gfx1100-v0.3/`, candidate `01273a8`) so results are comparable.

## Prerequisites (checked by step 0)

- Ubuntu 24.04-class host, Python 3.10–3.12, `bash`, `git`;
- an AMD GPU whose *visible compute agent* is exactly `gfx1100`
  (`rocm_agent_enumerator` prints only `gfx1100`);
- working `amdgpu` kernel driver with `/dev/kfd` + `/dev/dri` access
  (`render`/`video` groups);
- network access to `repo.amd.com` (wheel index) and PyPI;
- ~30 GB disk (venv + models), models via `scripts/download_models.sh`.

## Certification steps (verbatim commands)

0. **Provenance & environment freeze** — run from a CLEAN checkout of the
   commit being certified (fresh clone or `git archive` extraction; the
   tree must equal the commit's tree):

   ```bash
   git rev-parse HEAD 'HEAD^{tree}'    # record BOTH
   git status --porcelain              # MUST be empty before install
   grep PRETTY /etc/os-release; uname -r; python3 --version
   rocminfo | grep -E '^  Name: +gfx'   # the visible agent(s)
   rocm-smi --showproductname           # board identity
   ```

1. **Normal installation — ZERO source edits** (this is the point of the
   certification; if any file must be edited, STOP — that is a regression
   of the v0.3 contract):

   ```bash
   bash scripts/install.sh              # auto resolves gfx1100 fail-closed
   # equivalently: bash scripts/install.sh --gfx-target gfx1100
   ```

   Success shape: `==> [gfx] target: gfx1100 | wheel extras: device-gfx1100`
   printed BEFORE any download, and the final `scripts/verify_gpu.sh` gate
   prints `SPIKE-GPU-OK`.

2. **Stack sanity + classification**:

   ```bash
   bash scripts/verify_gpu.sh                     # auto: exact-arch contract
   QWEN3_TTS_ROCM_GFX_TARGET=gfx1100 bash scripts/verify_gpu.sh
   .venv/bin/qwen3-tts-rocm-check                 # 'validated architecture: gfx1100'
   .venv/bin/pip list | grep -E '^(torch|amd-torch|rocm-sdk-device)'
   # expected: torch 2.12.0+rocm7.14.0, amd-torch-device-gfx1100,
   #           rocm-sdk-device-gfx1100 — NO device-gfx1151 package
   ```

3. **Test suites** (models downloaded first):

   ```bash
   bash scripts/download_models.sh
   .venv/bin/python -m pip install -e ".[quality]"   # full CPU scope
   .venv/bin/python -m pytest -m "not gpu" -q        # v0.3: 358 passed expected
   .venv/bin/python -m pytest -m gpu -q              # 40 passed expected
   .venv/bin/ruff check .
   ```

4. **Benchmarks** (v1 + v2, JSONs into the evidence dir):

   ```bash
   .venv/bin/python scripts/benchmark.py    --json-out <ev>/benchmark-v1-<tag>.json
   .venv/bin/python scripts/benchmark_v2.py --json-out <ev>/benchmark-v2-<tag>.json
   ```

5. **Record final integrity** — after everything:

   ```bash
   git diff | wc -l               # MUST be 0 (no source edits)
   git status --porcelain | wc -l # only gitignored artifacts (.venv/, models/)
   sha256sum <ev>/* > <ev>/SHA256SUMS
   ```

6. **Report**: open an issue from the hardware-validation template with the
   exact commands, totals, RTF medians, and the SHA256SUMS manifest; label
   the evidence namespace `evidence/gfx1100-v<version>-*`.

## Honest-reporting rules (same as the rest of the repo)

- Record failures verbatim; never soften or trim transcripts (append-only).
- A zero-edit failure at ANY step is a contract regression, not a flaky
  test — report it as such.
- Only `gfx1100` results may be labeled gfx1100; do not generalize to
  other gfx11 parts; do not attach gfx1151 claims to gfx1100 runs.
- The `historically validated architecture: gfx1151` line printed by
  `qwen3-tts-rocm-check` reflects pre-v0.3 evidence and must not be
  presented as a fresh run.

## Future promotion path (not enabled now)

If a persistent gfx1100 machine becomes available later, the manual flow
above converts 1:1 into a self-hosted workflow: register the runner with a
new label (suggested: `radeon-gfx1100`; do NOT reuse `radeon-gfx1151`),
mirror `gpu-nightly.yml`'s hardening (no fork-PR triggers,
`contents: read`, retrying fetch, PYTHONPATH provenance assertion,
serialized concurrency, absolute venv/models paths) and gate it behind
`workflow_dispatch` first; only add a schedule once the runner has proven
reliability. Until then, no gfx1100 workflow file exists in this
repository on purpose.
