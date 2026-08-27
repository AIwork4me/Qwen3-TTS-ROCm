# =============================================================================
# NOTE: deliberately NO `# syntax=` directive — it forces a Docker Hub
# round-trip for the frontend image before anything else runs (fine upstream,
# fatal offline). Only classic Dockerfile features are used.
# =============================================================================
# Qwen3-TTS-ROCm — container image (AMD ROCm / gfx1151 focused).
#
# Build (from the repo root — note the -f, the context is the whole repo):
#
#     docker build -f docker/Dockerfile -t qwen3-tts-rocm:dev .
#
# Run with GPU passthrough (see docker/README.md for the full story):
#
#     docker run --rm \
#         --device /dev/kfd --device /dev/dri \
#         --group-add video --group-add render \
#         -v /path/to/models:/workspace/models \
#         -p 8000:8000 qwen3-tts-rocm:dev
#
# Design notes:
# * Base is stock ubuntu:24.04, which ships Python 3.12 natively — no
#   deadsnakes PPA needed.
# * The pip stack (pinned AMD ROCm torch wheels + `-e ".[dev,demo]"`) is
#   installed by calling scripts/install.sh VERBATIM — it stays the single
#   source of truth; there are deliberately NO duplicated pip lines here.
#   The only knob is QWEN3_TTS_ROCM_SKIP_VERIFY=1, which skips install.sh's
#   final verify_gpu.sh step because /dev/kfd does not exist inside a build
#   RUN layer. It changes nothing about WHAT gets installed.
# =============================================================================

FROM ubuntu:24.04

LABEL org.opencontainers.image.title="Qwen3-TTS-ROCm" \
      org.opencontainers.image.description="ROCm (gfx1151) runtime adapter, model manager and enhanced Gradio demo for Qwen3-TTS" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

# Minimal system deps:
#   ca-certificates        TLS for repo.amd.com / pypi.org during install
#   curl                   handy inside the container, kept from install docs
#   git                    git+https-style pulls sometimes requested upstream
#   libatomic1             runtime dep of the AMD ROCm torch wheel (torch
#                          fails to import without it — found by smoke test)
#   libsndfile1            soundfile runtime requirement
#   ffmpeg                 audio format handling expected by qwen-tts tooling
#   python3                metapackage: provides the *unversioned* `python3`
#                          symlink (install.sh calls it); noble ships 3.12
#   python3.12-venv        venv module + ensurepip needed by install.sh's
#                          fallback path (uv is NOT baked into the image)
# sox is intentionally omitted (optional): add `sox \` below if you want CLI
# post-processing of wavs inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libatomic1 \
        libsndfile1 \
        python3 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Repo -> /workspace. `.dockerignore` (repo root) keeps .git/.venv/models/
# evidence out of the context; model weights ALWAYS come in at run time as a
# mount, never baked into the image.
COPY . /workspace

# Single source of truth: the same scripts/install.sh a bare-metal host runs.
# SKIP_VERIFY=1 documented above; cache cleanup stays in the SAME layer so the
# ~GB of wheel downloads do not linger in image size.
RUN QWEN3_TTS_ROCM_SKIP_VERIFY=1 bash scripts/install.sh \
    && rm -rf /root/.cache/pip /root/.cache/uv

# --- Runtime configuration ---------------------------------------------------
# Put the project's console entry points (qwen3-tts-rocm-check/-demo) and
# python/pip of the in-image venv on PATH so plain sessions resolve them:
#   docker run --rm --entrypoint bash qwen3-tts-rocm:dev -c 'qwen3-tts-rocm-check'
# (`scripts/run_demo.sh` resolves the venv explicitly and does not need this,
# but overridden-entrypoint runs do.)
ENV PATH="/workspace/.venv/bin:${PATH}"
# Models are mounted, not baked. Point the resolver at the standard mount;
# override either side freely (`-v host:/elsewhere -e QWEN3_TTS_ROCM_MODELS_DIR=/elsewhere`).
ENV QWEN3_TTS_ROCM_MODELS_DIR=/workspace/models
VOLUME ["/workspace/models"]
# HF_ENDPOINT intentionally has NO default: downloads normally target the
# volume-mounted caches at run time; set it per-run (e.g. hf-mirror.com) if
# your network needs it.

EXPOSE 8000

# Entrypoint = the project's own launcher; any CMD/extra args after the image
# name are forwarded verbatim to `qwen3-tts-rocm-demo` (e.g. --help).
ENTRYPOINT ["bash", "scripts/run_demo.sh"]
