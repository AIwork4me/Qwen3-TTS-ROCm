# Multi-Architecture Support Contract — v0.3 Design (Task 1)

Status: **design of record** for the v0.3 multi-arch refactor (Tasks 2–5, 13).
Scope: make `gfx1100` a zero-source-edit first-class Radeon reference path
while preserving `gfx1151` exactly as historically validated.

## 1. Goals and non-goals

Goals:

1. `gfx1100` → `device-gfx1100` wheel extras; `gfx1151` → `device-gfx1151`
   wheel extras; nothing else changes about the pinned stack
   (`torch==2.12.0+rocm7.14.0`, `torchvision==0.27.0+rocm7.14.0`,
   `torchaudio==2.11.0+rocm7.14.0`, index
   `https://repo.amd.com/rocm/whl-multi-arch/`).
2. A fresh gfx1100 checkout installs and verifies with **zero repository
   source edits**.
3. Unknown/ambiguous architecture requests **fail closed**. No silent
   guessing, no `gfx11*` family shortcut, no `HSA_OVERRIDE_GFX_VERSION`
   spoofing (the code never sets it and actively warns when a user has).
4. gfx1151 CLI semantics preserved as far as possible (one-command UX must
   keep working for both validated architectures).

Non-goals: supporting any architecture beyond the two evidence-backed ones;
auto-installing ROCm itself; multi-arch co-existence in one venv.

## 2. Evidence classes (must stay distinct everywhere)

| Target | Wheel extras | Validation state | Meaning |
|---|---|---|---|
| `gfx1100` | `device-gfx1100` | **current** | Freshly executable/validated in v0.3 on the Radeon Pro W7900D host available to this program (issue #1 pre-program evidence + v0.3 Tasks 8ff fresh execution) |
| `gfx1151` | `device-gfx1151` | **historical** | Validated on the Radeon 8060S host before v0.3 (v0.2/0.2.1 closure). The hardware is NOT available in this program; v0.3 only preserves and regression-tests the code path (CPU/static). No fresh gfx1151 hardware execution is claimed anywhere. |
| anything else | — | unknown | Never receives a "validated" verdict; selection fails. |

## 3. Architecture detection investigation (performed on the real host)

Before designing the CLI we measured what pre-PyTorch detection can actually
rely on. Host: the W7900D machine of this program (Ubuntu 24.04.4,
kernel 6.8.0-79-generic, system ROCm 7.2.1 userspace at `/opt/rocm`).

| Source | Observed behavior | Verdict for detection |
|---|---|---|
| `rocm_agent_enumerator` | prints one line per *visible* compute agent: `gfx1100` (rc 0). Requires ROCm userspace on PATH; enumerates only agents the process can access (device cgroup aware). | Good primary probe — but **not universally present** on hosts that never installed ROCm userspace (the pinned pip stack is self-contained per issue #1, so a bare host may legitimately have none). |
| `rocminfo` | full HSA topology: `Name: gfx1100`, `Marketing Name: AMD Radeon Graphics`; also lists CPU nodes and ISA strings (`amdgcn-amd-amdhsa--gfx1100`, `gfx11-generic`). | Same availability constraint; heavier output, parseable but second choice. |
| sysfs `/sys/class/drm/card*/device` | vendor `0x1002`, device `0x744b` (W7900D), unique ids. On this chassis sysfs shows the *host-wide* card set (8× `0x744b`) while the device cgroup exposes exactly one `/dev/dri/card1` + `renderD128` to this session; KFD topology nodes of the masked GPUs return EPERM. | A device-id→gfx mapping table would be *our own guess data* (maintenance hazard, wrong on masked multi-GPU hosts). **Rejected** as a detection source. |
| `/sys/class/kfd/kfd/topology/nodes/*` | accessible GPU nodes expose the arch via `properties` → `gfx_target_version 110000` (the `name` file itself reads `ip discovery` on GPU nodes — measured); cgroup-masked nodes return EPERM on read. | Same availability constraint as rocminfo (needs loaded amdgpu+kfd); version-number encoding (not a `gfxNNNN` string) makes it a cross-check only, not a primary probe. |
| Docker `build` layer | no `/dev/kfd`, no GPU at all (by design; see `QWEN3_TTS_ROCM_SKIP_VERIFY`). | Detection impossible by construction → explicit target required for image builds (Task 5 build-arg). |

Additional measured reality: this very host physically carries multiple
W7900D devices but the session sees exactly one compute agent. Multi-GPU
ambiguity is therefore not hypothetical: **auto-detection must dedupe arches
across all visible agents and require exactly one distinct gfx value.**

### Decision

Detection is **deterministic where it works and absent where it cannot
work** — i.e., *not robust enough to be the only path*. Per the brief:

- `--gfx-target gfx1100` and `--gfx-target gfx1151` are the **explicit,
  always-available** selection;
- `--gfx-target auto` (the default) is a **fail-closed convenience**:
  - probe `rocm_agent_enumerator`, fall back to `rocminfo`;
  - collect all gfx agent names; dedupe;
  - exactly one distinct value that is a known target → resolve to it;
  - zero GPUs visible, tool absent, unknown value, or >1 distinct values →
    hard error telling the user to pass an explicit `--gfx-target`;
  - the resolved target is printed **prominently before any download
    starts**.

Auto is admissible only because it *cannot guess*: it either resolves a
unique visible arch or refuses. An explicit flag always overrides. No
substring/prefix matching exists anywhere (`gfx1103` is NOT `gfx110*`);
matching is exact-string against the contract table.

## 4. CLI surface

### 4.1 `scripts/install.sh` (Task 2)

```
bash scripts/install.sh [--gfx-target {gfx1100,gfx1151,auto}] [--with-models]
```

- Default: `auto` (fail-closed detection as above; preserves the historical
  one-command UX on both validated hosts).
- Invalid value → immediate `exit 2` listing the supported values, before
  the venv is created and long before any wheel download.
- Resolution happens **before** the pinned-wheel step; the banner prints
  `==> [gfx] target: gfx1100 (wheel extras: device-gfx1100)` (or the
  detection-failure error).
- The constructed pip command is identical to v0.2.1's for gfx1151
  (byte-identical modulo the extras token) and to issue #1's validated
  gfx1100 command:
  `.venv/bin/python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx<NNNN>]==2.12.0+rocm7.14.0" "torchvision[device-gfx<NNNN>]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0" --no-input`
- No CUDA/PyPI fallback path exists or will be added; if repo.amd.com is
  unreachable the script fails like today.
- `QWEN3_TTS_ROCM_SKIP_VERIFY` and `--with-models` semantics unchanged.
- Idempotence unchanged (pip itself is the idempotence mechanism; the
  extras string is part of the "already satisfied" check).

### 4.2 `scripts/verify_gpu.sh` (Task 3)

```
QWEN3_TTS_ROCM_GFX_TARGET={gfx1100,gfx1151,auto}   # default auto
```

Runs inside the venv, so the **actual** arch comes from torch's
`gcnArchName` (ground truth, not a probe). Semantics:

- HIP build present; HIP `7.14.x` (pinned-stack invariant, arch-independent);
- at least one GPU visible; actual arch read via `gcnArchName`;
- `auto`: actual arch must be **exactly** one of the known targets
  (`gfx1100`/`gfx1151`) — anything else fails with "ROCm-visible but not a
  validated architecture";
- explicit target: actual arch must equal the request — mismatch fails
  (an explicit gfx1151 request on gfx1100 hardware is a **failure**, and
  vice versa; no reinterpretation);
- bf16 matmul finite, SDPA finite, torchaudio import, success marker
  `SPIKE-GPU-OK` (unchanged).

### 4.3 `qwen3-tts-rocm-check` (Task 3)

Output gains one explicit classification line per visible GPU arch:

- `validated architecture: gfx1100` — current-class target;
- `historically validated architecture: gfx1151` — historical-class target
  (with the not-rerun-in-v0.3 scope note);
- `ROCm-visible but not validated architecture: gfxXXXX` — everything else.

Exit code semantics unchanged (0 iff no errors).

### 4.4 Docker (Task 5)

```
docker build -f docker/Dockerfile \
  --build-arg QWEN3_TTS_ROCM_GFX_TARGET=gfx1100 \
  -t qwen3-tts-rocm:gfx1100 .
```

- Build-arg mandatory-by-default: inside a build layer there is no GPU, so
  `auto` would be a lie; the Dockerfile passes the arg through to
  `install.sh --gfx-target` explicitly. `auto` remains accepted only when
  the builder knowingly opts in (documented as unsupported for images).
- OCI labels updated to the multi-arch truth (no gfx1151-only wording).

## 5. Single source of truth and testable helpers

- `install.sh` runs before any venv exists, so it keeps a **literal bash
  `case` table** (`gfx1100) EXTRA=device-gfx1100;; gfx1151) EXTRA=device-gfx1151;; *) fail`).
- `src/qwen3_tts_rocm/gfx.py` becomes the canonical Python contract
  (mapping, validation-state classification, wheel-command construction as
  pure functions) used by `verify_gpu.sh` (via the venv python) and
  `cli_check`/`env`.
- A CPU test parses `install.sh` and asserts the bash table equals the
  Python table (drift between the two is a test failure), mirroring how
  `tests/test_install_sh.py` already pins the python-version guard.

## 6. Failure-mode matrix

| Situation | Behavior |
|---|---|
| `--gfx-target gfx1200` (unknown) | immediate exit 2, supported values listed, nothing installed |
| `--gfx-target auto`, no ROCm tools on PATH | exit 1 with "pass --gfx-target explicitly" (before downloads) |
| `--gfx-target auto`, zero visible GPU agents | same as above |
| `--gfx-target auto`, two distinct visible arches | same as above (ambiguity refused) |
| `--gfx-target auto`, N agents all gfx1100 | proceed with gfx1100, banner printed |
| explicit target ≠ detected arch in verify | verify exit 1 (mismatch is failure) |
| arch visible but not in contract (e.g. gfx1101) | verify/auto fail: "not a validated architecture" |
| repo.amd.com unreachable | pip fails as today (no fallback index) |
| user set `HSA_OVERRIDE_GFX_VERSION` | refused-by-warning (env.py advisory generalized; we never set it) |

## 7. Testing strategy

CPU/static (Task 4; run everywhere, including gfx1151-less environments):

- mapping: gfx1100→device-gfx1100, gfx1151→device-gfx1151;
- invalid target rejection (install.sh argument parsing via real script
  invocation with a stubbed pip);
- no-silent-fallback: install command text contains the repo.amd.com index
  and the exact pinned versions for BOTH targets (parsed from the script);
- `bash -n` shell syntax; `ruff check .`;
- verify_gpu.sh control flow with a fake python (simulated arches:
  gfx1100/gfx1151/gfx1101/missing) — **simulation proves code logic only**;
  any gfx1151 simulation is labeled CODE LOGIC, never hardware evidence;
- architecture mismatch (requested gfx1151 + detected gfx1100) fails;
- validation-state classification lines from `gfx.py`;
- install-command generation equality between bash table and Python table.

Real execution (this program's hardware = gfx1100 only): Tasks 8ff. The
gfx1151 branch is additionally covered by the Task 7 regression audit
(static branch → expected wheel target + validation configuration).

## 8. Explicitly rejected alternatives

- `gfx11*`/family prefix matching — would silently claim unvalidated arches.
- Defaulting unknown hosts to either target — a guess.
- Setting `HSA_OVERRIDE_GFX_VERSION` — spoofing; breaks code-object truth.
- A device-id→arch lookup table — unmaintained guess data; wrong under
  device-cgroup masking (measured on this host).
- Auto-detection as the only path — impossible inside Docker build layers
  and absent on ROCm-less bare hosts (both measured/documented above).

## 9. Compatibility statement

- v0.2.1 gfx1151 command → v0.3 gfx1151 command: byte-identical pip
  arguments (same versions, index, extras).
- No-arg `install.sh` on a single-arch visible host: same one-command UX.
- `verify_gpu.sh` success marker, HIP-version gate, bf16/SDPA/torchaudio
  checks: unchanged.
- gpu-nightly.yml (historical gfx1151 runner): untouched in Tasks 2–5;
  gfx1100 CI gets its own additive workflow/label in Task 13.
