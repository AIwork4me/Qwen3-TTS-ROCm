# Task 11 — gfx1100 vLLM-Omni ladder status (v0.3)

Scope honesty first: **the current upstream-documented serving stack
(vllm 0.30.0+rocm723 + vllm-omni 0.30.0) could NOT be installed in this
program environment** — its only wheel source (`wheels.vllm.ai`) is blocked
by the environment's egress proxy (HTTP 403 at the squid layer, both
`/whl/rocm/` and `/rocm/0.30.0/rocm723/` paths probed; no ROCm wheel exists
on PyPI, GitHub release assets, or repo.amd.com; the vllm/vllm-openai-rocm
Docker image is unreachable because registry-1.docker.io is also forbidden).
Every rung below was therefore executed on the **closest real, upstream
stack available here**: the host lab image's AMD ROCm vLLM build
(`vllm 0.16.1.dev0+g89a77b108.d20260317.rocm721`, torch `2.9.1+gitff65f5b`,
HIP `7.2.53211`) paired with **vllm-omni 0.16.0 from PyPI** (the matching
0.16.x minor per upstream's pairing rule) in an isolated gitignored
`.work-vllm/venv` that bridges the system site-packages via a `.pth` file.
The validated `.venv` was untouched. gfx1151 settings/thresholds were NOT
inherited; everything was re-measured on this host.

## 11A — feasibility/import/install: **ACHIEVED (substituted stack)**

- Import/platform: `vllm 0.16.1.dev0+rocm721`, `RocmPlatform`, capability
  `DeviceCapability(major=11, minor=0)`, GPU visible (`gfx1100`) —
  `11a-import.txt`.
- **Compiled-architecture evidence**: vllm's compiled extension
  `_C.abi3.so` embeds code objects for exactly `gfx1100, gfx1101, gfx1150,
  gfx1151, gfx1200, gfx1201, gfx90a, gfx942` — gfx1100 is a first-class
  compiled target of this build.
- Real minimal inference on the GPU: LLM(Qwen2.5-0.5B via ModelScope,
  enforce-eager) generated text at ~14.8 tok/s output —
  `11a-minimal-inference.txt` (two prior attempts failed for harness
  reasons — stdin re-exec and missing `__main__` guard — disclosed in the
  log).

## 11B — offline inference: **ACHIEVED** (all three 1.7B task families)

Upstream example: `vllm-omni v0.16.0`
`examples/offline_inference/qwen3_tts/end2end.py`, fetched byte-exact from
the `v0.16.0` tag. Two recorded deviations in a local copy
(`end2end-localpath.diff`, 20 lines): (1) the three hard-coded HF model ids
replaced by local model paths (`omni_snapshot_download` short-circuits
local paths; the HF-hub online download path stalled behind the proxy and
the offline-cache symlink trick of the gfx1151 era no longer satisfies
`snapshot_download` metadata checks — both attempts logged); (2) the Base
family's remote OSS `ref_audio` URL replaced by a base64 data-URI of the
repo's own self-generated reference clip (the 0.16.0 worker refuses
http(s) ref_audio by design: "must be resolved by the serving layer").

- CustomVoice: exit 0, WAV `wavs/output_0_933ee150….wav` (24 kHz PCM_16).
- VoiceDesign: exit 0, WAV `wavs/output_0_e9f16ac3….wav`.
- Base (inline clone via data-URI): exit 0 after the deviation,
  WAV `wavs/output_0_31cb5fd9….wav`.
- Dependency gaps fixed inside the isolated venv (recorded): `sox`,
  `torchsde`, `imageio`, `resampy`, `pydub` (+ system sox binary).

## 11C — online serving: **ACHIEVED**

Upstream recipe (`vllm-omni serve <model> … --enforce-eager --omni`, port
8091) with the local CustomVoice 1.7B path. `GET /v1/audio/voices` → 200
(9 voices); `POST /v1/audio/speech` non-streaming → 200, valid 24 kHz WAV
(268,844 bytes; 30.8 s wall — enforce-eager is slow; no speed claim).

## 11D — HTTP PCM streaming: **MEASURED, split result**

Using the repository's own client-side timestamp probe
(`scripts/vllm_streaming_probe.py`, unchanged):

- short input (25 chars): 20 chunks, TTFA == total wall (10.50 s) —
  effectively single delivery for short prompts on this stack
  (STREAM-VERDICT: NOT-INCREMENTAL);
- long input (~342 chars): 983 chunks, **TTFA 3.03 s of 137.1 s wall**
  (ttfa/wall 0.022), decoded audio 81.9 s, total RTF 1.674, playback
  simulation 40 underruns — **true incremental streaming IS delivered**
  for longer generations (STREAM-VERDICT: INCREMENTAL).
- Scope: measured on THIS stack (0.16-era) only; the gfx1151 0.30-era
  TTFA figures (0.22–0.25 s) are NOT comparable and NOT inherited.

## 11E — concurrency: **MEASURED** (default config; this host only)

`scripts/vllm_concurrency_probe.py --levels 1,2,4,8 --requests-per-level 8`
against the live CustomVoice server — **32/32 requests OK, zero failures,
zero OOM**:

| c | TTFA p50 | TTFA p95 | E2E p50 | audio/wall | req/min | peak VRAM |
|---|---|---|---|---|---|---|
| 1 | 3.04 s | 8.68 s | 8.45 s | 0.269 | 5.79 | 15.88 GiB |
| 2 | 4.73 s | 21.20 s | 18.57 s | 0.317 | 6.71 | 15.88 GiB |
| 4 | 8.69 s | 16.44 s | 11.42 s | 0.379 | 8.55 | 15.97 GiB |
| 8 | 16.36 s | 36.10 s | 17.69 s | 0.431 | 9.04 | 16.11 GiB |

No stage-1 `max_num_seqs:1` tuning was applied (that was a gfx1151/0.30
deploy-config finding); the packaged default config was measured as-is.

## Not claimed

- No results on the current 0.30.x stack (install-blocked, above).
- No WebSocket rung (upstream-broken historically; not retried here).
- No quality claims; no cross-GPU comparisons; numbers are this-host-only.

## Errata (post-verification, round 1 — appended, nothing above altered)

1. 11D long input length: the JSON ground truth is `input_chars: 300`
   (the "~342 chars" figure above counted the generating expression, not
   the delivered input).
2. 11A compiled-target list: `_C.abi3.so` also embeds gfx950-family code
   objects beyond the 8 listed; gfx1100 remains present as stated. The
   word "exactly" overreached.
3. 11A failed-attempt disclosure: the two harness failures (stdin re-exec;
   missing __main__ guard) are described HERE in the status doc; the
   archived `11a-minimal-inference.txt` contains only the final green run.
4. Supplementary observation (for completeness): the 11B Base-family WAV is
   163.8 s — the upstream example's default `max_new_tokens: [2048]` allows
   very long generations; no duration constraint was imposed.
