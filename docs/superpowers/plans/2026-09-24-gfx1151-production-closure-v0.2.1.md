You are the lead engineer responsible for the next evidence-driven closure of:

https://github.com/AIwork4me/Qwen3-TTS-ROCm

Program name:

# Qwen3-TTS-ROCm v0.2.1 — gfx1151 Production Closure

## Mission

Deepen the existing Radeon 8060S / gfx1151 reference before the project moves to the second-architecture gfx1100 v0.3 program.

The North Star remains:

> Official Qwen3-TTS on AMD Radeon — capability by capability, benchmark by benchmark, with zero upstream patches.

This program must NOT weaken that claim.

The current validated host is:

* AMD Ryzen AI Max+ PRO 395
* Radeon 8060S
* gfx1151
* ROCm 7.14.0
* PyTorch 2.12.0+rocm7.14.0
* Python 3.12
* official qwen-tts 0.1.1 for the primary reference path

The existing v0.2.0 reference closure, evidence/, README, CHANGELOG, CI, tests, and reports are authoritative historical evidence. Do not rewrite historical transcripts.

---

# NON-NEGOTIABLE RULES

## Rule 1 — Evidence before claims

Never change a README status from:

* not validated
* partial
* smoke
* pending
* not proven

to a stronger state until a real execution artifact proves it.

Every public claim must trace to a verbatim transcript and/or machine-readable JSON in `evidence/`.

No fabricated or reconstructed benchmark results.

---

## Rule 2 — Preserve zero-upstream-patch semantics

The official `qwen-tts` reference path must continue using the published official package/API unmodified.

Do NOT vendor, monkey-patch, copy, or silently modify qwen-tts.

Fine-tuning may continue using the already-documented temporary SDPA workaround inside a gitignored upstream clone while PR #373 remains unmerged, but this fact must remain explicit.

vLLM-Omni work must be isolated from the official qwen-tts environment and claims.

---

## Rule 3 — Never mix runtimes

Maintain three clearly separate evidence categories:

A. Official `qwen-tts` Python API
B. Official Qwen fine-tuning workflow
C. `vLLM-Omni` deployment runtime

In particular:

* official qwen-tts Python API currently has NO true incremental audio API;
* if vLLM-Omni streaming works, call it:
  `vLLM-Omni true streaming on gfx1151`
* NEVER rewrite the qwen-tts capability row to imply its Python API streams incrementally.

---

## Rule 4 — Sequential execution with independent verification

There are 10 tasks below.

For EVERY task:

1. inspect current repository truth;
2. write a short implementation plan;
3. execute the task;
4. run all relevant tests;
5. capture verbatim evidence;
6. commit the implementation;
7. launch an independent verification Subagent;
8. the verifier must assume the implementer is wrong;
9. the verifier must reproduce important checks independently;
10. save its report under:
    `docs/superpowers/reports/`
11. only if the verifier returns PASS may you proceed to the next task.

If verifier says FAIL or PASS_WITH_CONCERNS:

* stop;
* fix the problem;
* rerun verification;
* do NOT enter the next task.

Never let an implementation agent verify its own task.

---

# TASK 0 — BASELINE FREEZE

Before changing anything:

Record:

* git HEAD
* git status
* current version
* current README capability matrix
* CPU test count
* GPU test count
* ROCm / torch / Python
* GPU name and gcnArchName
* qwen-tts version
* available model directories
* current upstream Qwen3-TTS issue #372 / PR #373 status
* current vLLM-Omni main SHA

Run:

```bash
git status --porcelain
git rev-parse HEAD

.venv/bin/python -m pytest -m "not gpu" -q
.venv/bin/python -m pytest -m gpu --co -q

rocm-smi --showproductname --showuse --showmeminfo vram

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("HIP", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name)
    print("arch", getattr(p, "gcnArchName", None))
PY
```

Save:

`evidence/gfx1151-production-closure-baseline-YYYY-MM-DD.txt`

Do not change claims.

Verifier must confirm clean provenance and exact baseline.

---

# TASK 1 — FIRST FULL-WEEKLY GPU CI GREEN

Current state:

* gpu-short has already had a real green self-hosted run.
* runbook still marks the first `full-weekly` execution pending.
* full-weekly contains:

  * all 38 GPU-marked pytest nodes
  * verify_gpu.sh
  * benchmark replication

Goal:

Obtain a real green `full-weekly` run on the gfx1151 self-hosted runner.

Requirements:

1. inspect `.github/workflows/gpu-nightly.yml`;
2. confirm workflow provenance guard still works;
3. dispatch:
   `suite=full-weekly`;
4. do not weaken any tests to get green;
5. capture:

   * run id
   * head SHA
   * runner name/labels
   * all job steps
   * 38 GPU test result
   * verify_gpu result
   * benchmark result
6. persist benchmark output as a GitHub Actions artifact or similarly durable CI artifact.

Do NOT automatically commit benchmark-nightly.json back into the repository.

Update:

* GPU CI runbook checklist
* README GPU CI wording if necessary
* evidence index
* CHANGELOG [Unreleased]

Evidence:

`evidence/gpu-ci-full-weekly-first-green-YYYY-MM-DD.txt`

Verifier must independently query GitHub Actions and confirm:

* self-hosted gfx1151 runner;
* code-under-test is fetched SHA;
* all 38 GPU nodes ran;
* benchmark executed;
* no skipped/failed job was hidden.

Commit only after evidence is complete.

---

# TASK 2 — DOCKER REAL-GPU E2E CLOSURE

Current Docker evidence proves image construction, not a complete GPU synthesis path.

Goal:

Prove a fresh Docker image can execute real Qwen3-TTS synthesis on Radeon 8060S / gfx1151.

Procedure:

1. build from current HEAD:

   ```bash
   docker build --no-cache -f docker/Dockerfile \
     -t qwen3-tts-rocm:gfx1151-e2e .
   ```

2. run with:

   * `/dev/kfd`
   * `/dev/dri`
   * correct render/video groups
   * mounted models directory

3. inside the container prove:

   * torch is ROCm
   * HIP 7.14
   * GPU is Radeon 8060S
   * arch is gfx1151
   * bf16 matmul finite
   * SDPA finite
   * torchaudio imports

4. load at least the 0.6B CustomVoice checkpoint using the repository loader.

5. generate real audio.

6. validate:

   * sample rate expected
   * finite waveform
   * non-silent
   * duration sane
   * WAV written successfully

7. run at least one meaningful GPU pytest slice from inside the container if technically practical.

Archive:

* build transcript
* runtime transcript
* machine-readable summary JSON

Files:

`evidence/docker-gpu-e2e-gfx1151-YYYY-MM-DD.txt`
`evidence/docker-gpu-e2e-gfx1151-YYYY-MM-DD.json`

Only then upgrade Docker wording from build-validated to GPU-runtime-validated.

Verifier must reproduce the container GPU diagnostics and inspect the WAV assertions.

---

# TASK 3 — OFFICIAL API BATCH INFERENCE CLOSURE

Upstream officially supports batch inference.

Do not invent a custom batching layer.

Use official qwen-tts APIs exactly as documented.

Validate batching for:

A. 0.6B CustomVoice
B. 1.7B CustomVoice
C. 1.7B VoiceDesign
D. 0.6B Base with reusable clone prompt
E. 1.7B Base with reusable clone prompt

Start with batch sizes:

1, 2, 4, 8

If a size fails due to real memory/runtime limits:

* record the failure verbatim;
* do not hide it;
* stop increasing that model's batch size;
* report the maximum validated batch size.

For every row collect:

* model
* batch size
* prompt lengths
* wall time
* generated audio duration per item
* total generated audio seconds
* aggregate audio seconds / wall second
* RTF where meaningful
* torch peak allocated/reserved
* ROCm/system GPU memory if measurable
* every output finite
* every output non-silent
* valid sample rate

Implement a reusable script such as:

`scripts/benchmark_batch.py`

with machine-readable output.

Evidence:

`evidence/batch-inference-gfx1151-YYYY-MM-DD.txt`
`evidence/batch-inference-gfx1151-YYYY-MM-DD.json`

Add GPU regression tests that prove batch shape/output correctness without turning normal nightly CI into a huge benchmark.

Do not claim batching performance beyond the tested host/configuration.

Verifier independently reruns representative B=1 and the highest stable B for at least two model families.

---

# TASK 4 — BENCHMARK V2: CONTROLLED REPRODUCIBILITY

Current benchmark evidence is useful but much of it is n=2 and explicitly susceptible to thermal/background variance.

Create benchmark methodology v2.

Do NOT delete historical benchmark evidence.

Requirements:

1. fixed prompt manifest;
2. fixed random seeds where official API permits;
3. one explicit warmup;
4. at least n=5 measured runs per benchmark cell;
5. report:

   * median
   * min/max
   * p10/p90 where statistically meaningful
   * mean
   * stdev
6. separate:

   * model load time
   * warmup time
   * generation time
7. record:

   * audio duration
   * RTF
   * torch peak allocated/reserved
   * GPU/system memory observations
8. capture initial and final:

   * rocm-smi
   * temperature/clock information if exposed reliably
9. distinguish cold-start and warm-model measurements.

Cover:

* CustomVoice 0.6B
* CustomVoice 1.7B
* VoiceDesign 1.7B
* Base 0.6B
* Base 1.7B

Do not collapse them to one "Radeon performance" number.

Create:

`docs/benchmarks-v2.md`

and:

`evidence/benchmark-v2-gfx1151-YYYY-MM-DD.json`
`evidence/benchmark-v2-gfx1151-YYYY-MM-DD.txt`

The existing benchmark script may be extended only if backward compatibility is preserved.

Verifier must recompute all reported statistics directly from JSON rather than trusting the Markdown table.

---

# TASK 5 — LONG-TEXT + SOAK STABILITY

Goal:

Move beyond sentence-scale correctness and characterize stability.

Part A — Long-text ladder

Create a fixed manifest with increasing text lengths.

Use at minimum:

* short
* medium
* long
* very long

Use English and Chinese.

Measure:

* input chars/tokens where accessible
* wall time
* output audio duration
* RTF
* peak memory
* output waveform sanity
* whether EOS was natural
* whether max_new_tokens truncated generation
* exceptions / HIP errors / NaNs

Do not claim arbitrary "maximum length" unless a real boundary is experimentally established.

Part B — Soak test

Keep one model resident and repeatedly synthesize for a bounded session.

Target:

* at least 30 minutes;
* extend to 60 minutes if the run remains stable and practical.

Every N requests record:

* successful request count
* failed request count
* wall time
* GPU memory
* process RSS
* generated audio duration
* peak memory
* cumulative failures
* temperature if reliable

Detect memory growth.

Also run a second mode that repeatedly:
load → generate → unload

to test allocator cleanup.

Create:

`scripts/soak_test.py`

with:

* deterministic manifest
* configurable duration/request count
* JSONL telemetry
* nonzero exit on correctness failure

Evidence:

`evidence/long-text-gfx1151-YYYY-MM-DD.*`
`evidence/soak-gfx1151-YYYY-MM-DD.*`

No "memory leak free" claim unless measured baseline behavior supports it.

Verifier independently analyzes telemetry for monotonic memory growth and failure rate.

---

# TASK 6 — 0.6B BASE FINE-TUNING E2E

Important upstream fact:

Qwen3-TTS upstream documents fine-tuning support for BOTH:

* Qwen3-TTS-12Hz-1.7B-Base
* Qwen3-TTS-12Hz-0.6B-Base

The repository currently has strong 1.7B execution evidence but not equivalent 0.6B FT closure.

Goal:

Repeat the existing disciplined fine-tuning validation protocol with 0.6B Base.

Do not claim convergence or quality.

Required chain:

1. use pinned pristine upstream source;
2. confirm upstream PR #373 state live;
3. if still unmerged:

   * preserve current disclosed SDPA workaround only inside gitignored upstream clone;
4. dataset preparation;
5. real training;
6. at least two epochs / enough optimizer steps to prove repeated updates;
7. checkpoint save;
8. checkpoint fingerprint;
9. official API reload;
10. post-finetune synthesis;
11. waveform sanity;
12. peak memory measurement;
13. restore upstream clone pristine.

Prefer two independent training runs if cost is reasonable, mirroring the 1.7B evidence discipline.

Evidence:

`evidence/finetune-06b-gfx1151-run1-YYYY-MM-DD.*`
`evidence/finetune-06b-gfx1151-run2-YYYY-MM-DD.*`

Update fine-tuning docs to distinguish:

* 1.7B execution validation
* 0.6B execution validation
* no convergence claim
* no multi-speaker claim
* PR #373 still required until merged/released

Verifier checks pristine-before/after state and reloads one produced checkpoint independently.

---

# TASK 7 — QUALITY BENCHMARK V2

Current quality benchmark is smoke-grade.

Upgrade breadth without pretending objective metrics equal human perceptual quality.

Goals:

A. content correctness across all 10 official languages
B. speaker similarity for Base cloning
C. reproducible manifests and metric provenance

Create a fixed, versioned manifest.

All 10 official languages:

* Chinese
* English
* Japanese
* Korean
* German
* French
* Russian
* Portuguese
* Spanish
* Italian

For every sample record:

* intended text
* generated WAV
* ASR transcript
* normalization method
* WER or CER as appropriate
* language
* model
* seed
* audio duration

ASR model:

Evaluate whether the current whisper-tiny metric is adequate.

If replacing it with a stronger ASR model:

* document exact model/version;
* first prove it runs correctly on gfx1151;
* do not silently compare scores across different ASR models.

For Base cloning:

* use speaker-embedding cosine similarity;
* include same-speaker positive controls;
* include different-speaker negative controls;
* do NOT invent universal quality thresholds unless justified by evidence.

Do NOT produce a composite "quality score".

Do NOT claim MOS.

Instruction/emotion adherence may be recorded as "not objectively scored" unless a defensible metric exists.

Evidence:

`evidence/quality-v2-gfx1151-YYYY-MM-DD.json`
`evidence/quality-v2-gfx1151-YYYY-MM-DD.txt`

Documentation must explicitly explain:

* what the metrics measure;
* what they do NOT measure.

Verifier recomputes a sample of metrics from raw files.

---

# TASK 8 — CURRENT vLLM-OMNI TRUTH RUN

The existing vLLM-Omni evidence is a dated snapshot.

Current upstream has changed significantly.

Before testing, capture:

* vllm-project/vllm-omni main SHA
* current Qwen3-TTS recipe
* supported-model table
* current ROCm-specific deploy config behavior
* recent relevant Qwen3-TTS commits

Use an isolated environment.

NEVER mutate the official qwen-tts `.venv`.

First reproduce the current known CustomVoice offline path.

Then test current upstream support for:

1. CustomVoice 1.7B
2. VoiceDesign 1.7B
3. Base 1.7B voice clone

Optionally test 0.6B variants only after the 1.7B task families are understood.

For every task:

* use an upstream documented invocation;
* do not modify upstream code to force success;
* capture exact SHA/version;
* record startup/load time;
* output waveform checks;
* memory;
* failures verbatim.

If current main regresses on gfx1151:

* bisect only if useful;
* identify the first responsible upstream change if practical;
* file an upstream issue only after reproducing cleanly.

Evidence:

`evidence/vllm-omni-current-gfx1151-YYYY-MM-DD.txt`
`evidence/vllm-omni-current-gfx1151-YYYY-MM-DD.json`

Update stale roadmap wording based only on results.

Verifier independently checks upstream SHA and reruns one representative request.

---

# TASK 9 — vLLM-OMNI ONLINE SERVING + TRUE STREAMING

Current upstream officially exposes Qwen3-TTS through:

`POST /v1/audio/speech`

and documents:

* CustomVoice
* VoiceDesign
* Base voice clone
* PCM streaming
* WebSocket streaming
* async chunking

Goal:

Determine exactly which of these work on Radeon 8060S / gfx1151.

Strict ladder:

## 9A CustomVoice non-streaming server

Launch exactly from upstream recipe.

Verify:

* server reaches ready state;
* `/v1/audio/speech`;
* `/v1/audio/voices`;
* generated WAV valid.

## 9B VoiceDesign serving

Restart with VoiceDesign checkpoint.

Verify natural-language voice description request.

## 9C Base voice-clone serving

Restart with Base checkpoint.

Test:

* inline ref_audio + ref_text;
* if supported, precomputed voice;
* do not test runtime voice upload before simple Base request is proven.

## 9D PCM streaming

Use:

* `stream=true`
* `stream_format=audio`
* `response_format=pcm`

Measure actual byte arrival timestamps.

Do not infer streaming from server logs alone.

For every streaming request record:

* request start
* first response byte
* first complete audio chunk
* TTFA
* chunk count
* chunk byte sizes
* inter-chunk gap distribution
* total response wall time
* decoded audio duration
* total RTF
* longest inter-chunk gap
* playback-buffer simulation / underruns

## 9E WebSocket streaming

If upstream's documented WebSocket client works on gfx1151, record:

* connect time
* input event timing
* first audio event
* chunk cadence
* input.done → completion timing

## 9F Long-text streaming

Test one long input and characterize cadence/memory.

Create a dedicated script:

`scripts/vllm_streaming_probe.py`

Do NOT reuse the qwen-tts Python streaming result incorrectly.

Public wording must remain:

* official qwen-tts Python API: no true incremental streaming
* vLLM-Omni runtime: [measured result on gfx1151]

Evidence:

`evidence/vllm-online-serving-gfx1151-YYYY-MM-DD.*`
`evidence/vllm-streaming-gfx1151-YYYY-MM-DD.*`

Verifier must inspect actual client-side chunk timestamps.

---

# TASK 10 — vLLM-OMNI CONCURRENCY / THROUGHPUT CHARACTERIZATION

Only execute after Task 9 server correctness is PASS.

Start with CustomVoice.

Concurrency ladder:

c = 1, 2, 4, 8

If the host becomes unstable or OOMs:

* record the failure;
* stop increasing concurrency;
* never hide the boundary.

Use a fixed prompt set.

At each concurrency collect at least enough requests to compute:

* success count
* failure count
* TTFA p50
* TTFA p95
* E2E latency p50
* E2E latency p95
* generated audio seconds
* audio seconds / wall second
* requests / minute
* peak VRAM/GTT
* process RSS
* queueing behavior
* chunk cadence where streaming
* OOM / HIP errors

Investigate upstream's current Qwen3-TTS stage tuning:

* Base uses a larger stage-1 `max_num_seqs`
* CustomVoice / VoiceDesign upstream notes recommend stage-1 `max_num_seqs: 1` for TTFA

On gfx1151 compare only a small defensible matrix, e.g.:

CustomVoice:

* default
* stage 1 max_num_seqs=1

Base:

* default
* one conservative alternative only if evidence suggests it is useful

Do not over-tune hundreds of combinations.

The result should answer:

> What is the highest concurrency that is stable and useful on Radeon 8060S for this stack?

not:

> What is the universal best configuration?

Evidence:

`evidence/vllm-concurrency-gfx1151-YYYY-MM-DD.txt`
`evidence/vllm-concurrency-gfx1151-YYYY-MM-DD.json`

Create:

`docs/vllm-omni-gfx1151-serving.md`

Verifier recomputes p50/p95 and throughput directly from JSON.

---

# FINAL TASK — CLAIMS AUDIT AND v0.2.1 CLOSURE

After Tasks 1–10 all independently PASS:

Run the full project regression.

At minimum:

```bash
.venv/bin/python -m pytest -m "not gpu" -q
.venv/bin/python -m pytest -m gpu -q
ruff check .
```

Also run every new focused suite.

Perform a repository-wide claims audit for:

* validated
* supported
* streaming
* batch
* Docker
* quality
* fine-tuning
* vLLM
* production
* gfx1151
* Radeon

For every public statement classify it as:

* measured/validated
* upstream fact
* roadmap/not validated
* removed because unsupported

Never convert upstream claims into local Radeon claims without local evidence.

Update:

* README.md
* README_CN.md
* CHANGELOG.md
* evidence/README.md
* relevant docs

Preserve dated historical reports verbatim; append corrections instead of rewriting captured evidence.

Create:

`docs/gfx1151-production-closure-v0.2.1.md`

with:

1. baseline SHA
2. final SHA
3. task-by-task result
4. verifier result per task
5. exact tested hardware/software
6. capability matrix before vs after
7. remaining known gaps
8. no-overclaim audit
9. release readiness verdict

Launch one FINAL independent release-verification Subagent.

It must independently verify:

* all test counts;
* all public claims;
* evidence existence;
* version consistency;
* upstream status;
* zero-upstream-patch semantics;
* separation of qwen-tts and vLLM-Omni streaming claims.

Only if final verifier returns PASS:

prepare v0.2.1 release notes.

Do NOT tag or publish the release without explicit user approval.

---

# STOP CONDITIONS

Stop immediately and investigate instead of working around a failure if:

* official qwen-tts needs source modification;
* a benchmark produces NaN/non-finite audio;
* GPU provenance is unclear;
* code under test comes from the wrong checkout;
* an upstream repository SHA cannot be established;
* vLLM-Omni requires an undocumented local patch;
* a verifier cannot reproduce a claimed result;
* evidence and README disagree.

Failures are valuable evidence.

Never convert a real failure into a green result by weakening an assertion.

---

# DEFINITION OF SUCCESS

The program succeeds when the repository can truthfully say, specifically for:

**Radeon 8060S / gfx1151**

that it has:

* real nightly GPU regression;
* real full-weekly GPU regression;
* real Docker GPU E2E;
* official API batch inference characterization;
* controlled benchmark v2;
* long-text and soak stability evidence;
* both 1.7B and 0.6B fine-tuning execution evidence;
* broader multilingual quality evidence;
* current vLLM-Omni compatibility evidence;
* OpenAI-compatible Qwen3-TTS serving evidence if it works;
* true vLLM-Omni streaming measurements if it works;
* concurrency/throughput measurements if it works;
* no unsupported extrapolation beyond gfx1151.

Then stop.

Do not begin gfx1100 adaptation in this program.
