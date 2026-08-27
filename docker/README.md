# Docker for Qwen3-TTS-ROCm (ROCm / gfx1151)

Everything about running the project in a container lives here on purpose —
`README.md` / `README_CN.md` stay short and freeze review state.

## Build

The build context is the **repo root** (`install.sh` needs the whole tree),
so pass `-f` explicitly:

```bash
docker build -f docker/Dockerfile -t qwen3-tts-rocm:dev .
```

Expect roughly 10–25 minutes: inside an image layer, `scripts/install.sh`
downloads the pinned AMD ROCm torch wheel stack (~4 GB) from
`repo.amd.com`, exactly like a bare-metal host does. The pip logic is not
duplicated in the Dockerfile — it *calls* `scripts/install.sh`.

## Run with GPU passthrough

gfx1151 APUs expose themselves through `/dev/kfd` + `/dev/dri`; give the
container access and the right supplementary groups:

```bash
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    -v "$PWD/models:/workspace/models" \
    -p 8000:8000 \
    qwen3-tts-rocm:dev
```

Then open <http://localhost:8000>.

Notes:

* Model weights are **not** baked into the image. The image declares
  `VOLUME /workspace/models` and defaults `QWEN3_TTS_ROCM_MODELS_DIR=/workspace/models`,
  so mount your downloaded weights there (run `bash scripts/download_models.sh`
  on the host first, or let the demo download into the mounted dir).
* `--group-add video --group-add render`: the numbers are resolved against the
  **host's** groups; if your distro uses different GIDs (e.g. `render` is
  often 992 instead of 109), use numeric forms,
  e.g. `--group-add "$(getent group video | cut -d: -f3)"`.
* **security-group caveat:** on some kernels/seccomp profiles, WebGL-ish GPU
  rendering for the Gradio UI can fail inside the container even when synthesis
  works — the AMD OpenGL/VA stack needs extra device nodes or cap adds for UI
  acceleration only; TTS inference itself is unaffected.

## Smoke test without any GPU

The entrypoint forwards args verbatim to the demo CLI:

```bash
docker run --rm qwen3-tts-rocm:dev --help
```

prints the `qwen3-tts-rocm-demo` help and exits 0 — proof that
entrypoint → `scripts/run_demo.sh` → `.venv` entry point all resolve inside
the image (this works without `/dev/kfd`; no model needed).

## CPU-only / quick diagnostics override

```bash
# The image ships its own ROCm-stack self-check (needs passthrough devices):
# NOTE: --entrypoint is exec-form and does NOT search .venv/bin — use the
# absolute in-image path.
docker run --rm --entrypoint /workspace/.venv/bin/qwen3-tts-rocm-check \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    qwen3-tts-rocm:dev

# CPU style run (slower, bf16 unsupported → pass fp32):
docker run --rm --entrypoint bash qwen3-tts-rocm:dev \
    -c 'exec bash scripts/run_demo.sh --device cpu --dtype float32'
```

## Build-time behavior worth knowing

* `scripts/install.sh` runs with `QWEN3_TTS_ROCM_SKIP_VERIFY=1`: the final
  `verify_gpu.sh` gate is skipped because `/dev/kfd` does not exist inside a
  build layer. What gets INSTALLED is byte-for-byte identical to bare metal;
  verify the stack after starting a container with
  `--entrypoint qwen3-tts-rocm-check` (above).
* `HF_ENDPOINT` has deliberately no default; set `-e HF_ENDPOINT=...` per run
  if your network requires a mirror.
* Image size lands around ~10 GB+ (ROCm wheels dominate); the pip caches are
  cleaned within the same RUN layer to avoid double-charging the size.
