# Task 9 status — gfx1100 Docker GPU E2E (v0.3)

**Headline: a real OCI-container build/run is BLOCKED in this program's
environment; the image-content stack was nevertheless built and GPU-validated
by other means, honestly labeled below. This is NOT "gfx1100 Docker
validated" — the in-container run remains pending until a runtime-capable
host exists.**

## What is blocked and why (evidence: `preflight.txt`)

- `dockerd` runs (29.1.3, vfs, no bridge/iptables) but every image-layer or
  container operation needs namespace syscalls: `docker build` (legacy
  builder) fails with `unshare: operation not permitted`; `docker import`
  of an official Ubuntu cloud rootfs fails the same way at layer unpack;
  `unshare -m`/`unshare -U` both fail (seccomp filter mode 2 in this
  session container; `max_user_namespaces` is high but the profile denies
  the syscalls).
- `registry-1.docker.io` is Forbidden through the environment egress proxy
  (no image pulls).
- Fallback attempts that also failed, disclosed for completeness:
  - `proot` (ptrace-based, namespace-free) executes the rootfs but hangs on
    network I/O (apt/curl) — `10-apt-layer-proot.txt` shows the hung run.
  - mknod is denied (no CAP_MKNOD), tar extraction of device nodes fails.

## What WAS executed (real gfx1100 hardware, real image stack)

All logs are in this directory; every command is recorded verbatim.

1. **Image content build** (the Dockerfile's steps, outside a container
   runtime):
   - Root filesystem: the official Ubuntu 24.04 (noble) server cloud
     rootfs (`cloud-images.ubuntu.com`, 24.04.5 at fetch time — the
     rolling `ubuntu:24.04` tag equivalent).
   - apt layer: the Dockerfile's exact package list
     (`ca-certificates curl ffmpeg git libatomic1 libsndfile1 python3
     python3.12-venv`) with its full recursive dependency closure —
     303 `.deb`s downloaded on the host from the Ubuntu archive and
     unpacked into the rootfs with `dpkg-deb -x` (because apt inside
     proot hung on network I/O; the resulting file content is the same
     package set). ffmpeg/git/python3.12-venv(ensurepip) verified present.
   - repo layer: the tracked build context per `.dockerignore` semantics
     (87 files; `scripts/install.sh` hash-verified equal to candidate
     HEAD `2c106cc`).
   - install layer: `QWEN3_TTS_ROCM_SKIP_VERIFY=1 bash scripts/install.sh
     --gfx-target gfx1100` — verbatim the Dockerfile RUN — creating
     `/workspace/.venv` inside the rootfs with the pinned
     `device-gfx1100` wheel stack (`11-install-layer.txt`).
2. **GPU E2E probe** (`12-e2e-imagestack.txt` +
   `docker-gpu-e2e-gfx1100-imagestack.json` +
   `docker-gpu-e2e-gen-gfx1100.wav`): the repository's own
   `scripts/docker_gpu_e2e.py`, executed with the image's own venv
   (`/workspace/.venv/bin/python` of the rootfs), on the real W7900D with
   the models tree provided via the documented `QWEN3_TTS_ROCM_MODELS_DIR`
   (analog of the `-v models:/workspace/models` mount). Result:
   **16/16 checks OK, `DOCKER-GPU-E2E-OK`** — ROCm build, HIP 7.14.60850,
   GPU `AMD Radeon Pro W7900D`, arch `gfx1100`, finite bf16 matmul + SDPA,
   torchaudio, `loader.load("custom-voice-0.6b")` (5.85 s), 9 speakers,
   official-API synthesis 4.16 s @ 24 kHz (RMS 0.1348, non-silent, WAV
   written by soundfile).
3. **Representative GPU pytest slice** (`13-gpu-slice-imagestack.txt`):
   `pytest -m gpu tests/test_generate_custom_voice_06b.py` with the image
   venv — **4 passed in 37.31 s** (historical in-container analog: 4
   passed in 38.33 s on gfx1151, 2026-09-24).

## Scope boundaries (what this does and does not prove)

- PROVES: the Dockerfile's build recipe (package set + install.sh
  `--gfx-target gfx1100`) produces, from a stock Ubuntu 24.04 userland, a
  wheel stack whose own interpreter passes the full in-image GPU probe and
  a GPU pytest slice on the gfx1100 host — i.e. the image **content** is
  correct and GPU-capable.
- DOES NOT PROVE: container isolation semantics (namespaces/cgroups),
  `--device /dev/kfd --device /dev/dri` passthrough behavior, or
  anything requiring the OCI runtime — those were impossible in this
  environment (see above). The "Docker GPU E2E (in-container)" capability
  row for gfx1100 therefore remains **pending**, not claimed.

The gfx1151 in-container E2E (2026-09-24) remains the historical reference
for the container-runtime path.

## Erratum (post-verification)

`10-apt-layer-proot.txt` is 0 bytes (the hung proot/apt attempt was killed
before any output flushed); it documents that the attempt produced nothing,
not the hang itself. The hang was observed interactively; the preflight and
the dpkg-deb fallback path remain the substantive evidence. Noted by the
Task 9 verifier; appended without altering any earlier text.
