# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-22 — [Radeon Reference Closure](https://github.com/AIwork4me/Qwen3-TTS-ROCm/releases/tag/v0.2.0) (opened 2026-09-20; previously Unreleased; tag `v0.2.0` on `bec162c` after final release-verifier PASS)

### Added

- 0.6B end-to-end capability validation on the Radeon 8060S (P0 capability
  parity, Task 1): new on-GPU test suites for `custom-voice-0.6b`
  (metadata surface, single/batch generation, sampling-kwarg passthrough,
  and the pinned `instruct` boundary — accepted but silently ignored by the
  installed wrapper for 0.6B, per the Task 0 ground-truth audit) and
  `base-0.6b` (ref-text cloning, x-vector-only cloning, reusable
  `create_voice_clone_prompt`, official-demo-format save/load roundtrip
  parity, batch cloning). The suite grows to 250/250 on the validation host
  (216 CPU + 34 real-GPU); CPU-only CI was expected to pass 215 + 1
  HIP-gated skip (host-derived expectation at authoring time — the first
  actually executed CI run, 2026-09-21, is recorded under [Unreleased]).
- `scripts/benchmark.py`: the 0.6B aliases (`custom-voice-0.6b`,
  `base-0.6b`) are benchmarked through their family's official entry
  points; per-alias `load_seconds` (timed `loader.load`) and
  `peak_alloc_gb` (torch peak allocation across the alias's run, reset
  after load) are printed, stored under `meta.per_alias`, and attached to
  every per-cell record; JSON `meta` now also records `git_head` and the
  audited upstream `upstream_qwen3_tts_sha`. Archived 0.6B evidence run:
  `evidence/benchmark-06b-2026-09-20.json` + verbatim transcript
  `evidence/benchmark-06b-2026-09-20.txt`.
- README / README_CN: per-checkpoint validation-level matrix (Load /
  E2E Generate / Benchmark) directly under the verified-results table,
  separating load validation from functional validation; the model-repo
  row now reads "6 / 6 load-validated"; documented boundary note that 0.6B
  CustomVoice has no instruction control upstream (the installed wrapper
  accepts and nulls `instruct` for 0.6B).
- Multilingual Radeon validation matrix (P0 capability parity, Task 2):
  new harness `scripts/validate_languages.py` + authored fixture
  `tests/data/multilingual_samples.json` exercising every officially
  supported language end to end on the Radeon GPU — 10 CustomVoice + 10
  VoiceDesign cells + 4 representative cross-lingual clone pairs (1.7B
  models, one resident at a time, every generation keyword-first with
  `max_new_tokens=512`). Manifest language names resolve case-insensitively
  onto the installed package's `get_supported_languages()` identifiers
  (recorded per row); drift in either direction is a hard error, never a
  silent skip. The only capability claim is "end-to-end generation
  completed on Radeon" from waveform sanity (finite, non-silent, valid
  sample rate, bounded duration) — no pronunciation-quality claims
  anywhere; cross-lingual clone coverage is stated as representative
  (4 pairs), not exhaustive. The cross-lingual transcript rule is
  structural: `ref_text` is exactly the manifest sentence synthesized into
  the reference audio. 13 CPU unit tests pin the pure helpers. Archived
  run: `evidence/multilingual-matrix.json` + verbatim transcript
  `evidence/multilingual-matrix.txt` (24/24 matrix cells + 4/4 reference
  generations passed, 152.6 s wall). README / README_CN gained the
  "Multilingual capability matrix" section rendered from that evidence.
- First-class Voice Design → reusable-voice workflow (P0 capability parity,
  Task 3): new `qwen3_tts_rocm.voice_workflow` composes ONLY the three
  official inference APIs (`generate_voice_design`,
  `create_voice_clone_prompt`, `generate_voice_clone`) on unmodified
  `loader.load` objects into describe → preview → save → reuse, honouring
  the official model split (the wrapper hard-gates design on the
  VoiceDesign checkpoint and prompt/reuse on Base — probe-proven, so
  `design_voice` takes the Base object as `prompt_model`). The transcript
  rule is enforced by construction: one `text` local feeds both the
  generation and the `ref_text`. Saved voices are official-demo-compatible
  `.pt` files (the `"items"` key is byte-format-identical to upstream's
  `{"items": [asdict(item) ...]}` payload) plus an in-file `voice_meta`
  sidecar that survives `torch.load(..., weights_only=True)` on the
  installed torch 2.12 (the pinned persistence variant; no sibling .json
  needed). The demo gained the ⑥ Voice Studio (音色工坊) tab — one-click
  design/preview/save/reuse with NO download/re-upload hop (the
  saved-voices dropdown replaces the file round-trip; tabs ①–⑤ untouched) —
  backed by new `SynthesisService.voice_studio_design` /
  `voice_studio_save` / `voice_studio_list` / `voice_studio_generate`
  methods (official split preserved through the size-1 model LRU: the
  VoiceDesign weights are evicted exactly when the prompt phase begins).
  `FakeTTSModel`'s prompt-item double gained the official fifth field
  `ref_text` (schema-faithful mirror). Suite grows to 282/282 on the
  validation host (244 CPU + 38 real-GPU: +15 CPU demo/backend/UI tests,
  +4 GPU workflow tests). Archived evidence with the three phases timed
  SEPARATELY: `evidence/voice-workflow-2026-09-20.json` (design 5.47 s,
  prompt 0.31 s, reuse 5.28 s / 6.42 s; every generation
  `max_new_tokens=512`) + verbatim transcript
  `evidence/voice-workflow-2026-09-20.txt`. README / README_CN gained the
  "Voice Design → reusable voice (Voice Studio)" section.
- Fine-tuning execution-validation smoke on ROCm (P0 capability parity,
  Task 4): the OFFICIAL upstream `finetuning/` workflow
  (`prepare_data.py` → `sft_12hz.py` → per-epoch checkpoint →
  `Qwen3TTSModel.from_pretrained` reload → `generate_custom_voice`)
  proven to execute end-to-end on the Radeon GPU. New
  `scripts/make_finetune_dataset.py` + authored manifest
  `tests/data/finetune_manifest.json` build a SELF-GENERATED corpus
  (12 original utterances — 6 Chinese + 6 English, distinct from the
  multilingual samples — plus one shared reference clip, all rendered by
  the official 1.7B CustomVoice model with `torch.manual_seed(1234)`
  before each render and `max_new_tokens=512`; transcript == manifest
  text by construction) in exactly upstream's JSONL schema; 8 CPU tests
  pin its pure helpers. Training smoke: 1.7B Base bf16, batch 2 / lr 2e-5
  (upstream defaults), 2 epochs = 12 optimizer steps (upstream steps per
  microbatch), wall 19.0 s, peak `torch.cuda.max_memory_allocated`
  18.02 GiB, loss lines quoted verbatim with no interpretation; reload
  synthesis of a manifest sentence is finite, 24 kHz, 3.52 s, non-silent
  (`assert_wav_sane` PASS). Scope is EXECUTION VALIDATION ONLY — no
  speaker-similarity / convergence / hyperparameter / stability /
  multi-speaker / scalability claims anywhere (NOT-PROVEN list in the
  doc). Upstream defects + deviations disclosed: (1) `sft_12hz.py`
  hard-codes `flash_attention_2`, which fails at init on the
  flash-attn-less ROCm stack (verbatim ImportError captured); a
  single-line `sdpa`
  override was applied INSIDE the gitignored `.upstream` clone, diff
  recorded, clone restored pristine afterwards (`git status --porcelain`
  empty at the pinned SHA — zero-patch audit); upstream issue filing
  noted as a human-owner action. (2) `github.com:443` was unreachable at
  run time, so the pinned clone was bootstrapped from codeload + the
  GitHub API with the commit object reconstructed SHA-identically to the
  pinned upstream SHA (authenticity by git content-addressing; procedure
  in the transcript). Scratch lives in gitignored `.upstream/` and
  `.work-finetune/` (plus `voices/`); NOTHING upstream or checkpoint
  shaped enters the commit. Archived evidence:
  `evidence/finetune-smoke-2026-09-20.json` + phase-marked verbatim
  transcript `evidence/finetune-smoke-2026-09-20.txt`; new reference doc
  `docs/finetuning-rocm.md` (PROVEN/NOT-PROVEN lists, reproduction
  commands); README / README_CN capability rows mark fine-tuning as
  ✅-scoped execution-only.
- Repository alignment with the North Star (P0 capability parity, Task 5):
  README / README_CN now open with the North Star sentence ("Official
  Qwen3-TTS on AMD Radeon — capability by capability, benchmark by
  benchmark, with zero upstream patches." / CN mirror) followed by a
  consolidated four-state capability matrix (✅ Radeon E2E validated · 🟡
  partial-load-only · ⬜ not validated · 🚫 not exposed upstream or
  intentionally not claimed) covering CustomVoice 1.7B/0.6B, VoiceDesign
  1.7B, Voice Clone 1.7B, Base 0.6B, reusable clone prompts,
  Design→Clone→Reuse, the 12Hz tokenizer, the multilingual matrix,
  fine-tuning (execution-only smoke), vLLM-Omni (🚫 roadmap) and true
  streaming (🚫 roadmap) — every ✅ cell links its evidence file, no
  percentage scores; the existing per-checkpoint and multilingual matrices
  kept as detail views. Conceptual boundary fixed in
  `scripts/download_models.sh` help (CustomVoice was misfiled as
  "reference-voice cloning": it is preset/custom-speaker generation; Base
  is the zero-shot-cloning + fine-tuning family) and stated explicitly in
  both READMEs. Stale suite counts recomputed honestly and identically in
  EN/CN: validation host 252 CPU + 38 GPU = 290/290; the CPU-only CI
  matrix runs the same 252 CPU tests with 1 HIP-gated skip (previously
  contradictory 215/243/216/244/34 figures). A clearly-labelled
  "Roadmap (not yet validated)" section was added to both READMEs: the
  vLLM-Omni-on-ROCm ladder (feasibility first, online serving only when
  upstream supports the required path, no concurrency assumptions on the
  single-GPU iGPU) and true streaming with its required measurements
  (time to first audio, chunk cadence, total RTF, buffer underrun,
  long-text behaviour), noting explicitly that upstream's "97 ms" figure
  is NOT a Radeon number. `src/qwen3_tts_rocm/patch.py` renamed to
  `compat.py` (module docstring now describes what it actually is — a
  version advisory plus a deliberately-empty reserved patch point; the
  old name overstated it) with all internal imports/tests updated and no
  compatibility shim (the only importers were `loader.py` and the test
  suite); full CPU + GPU suites re-run green after the rename.
  `evidence/README.md` gained an errata note for two misleading labels in
  the fine-tuning transcript (line 108's "pristine" label follows a
  mid-bootstrap staged listing; line 1546's `as-is-run-exit=0` is the
  `| tail -40` pipe's exit, not the failed run's) and its
  multilingual-transcript row no longer mentions a `RUN_EXIT=0` token the
  transcript does not contain — transcripts themselves stay byte-for-byte
  unedited per the no-hand-edit policy. Task 1–4 verifier reports copied
  verbatim to `docs/superpowers/reports/`; final program report at
  `docs/p0-parity-report.md`. Two latent `ruff` errors in
  `scripts/make_finetune_dataset.py` fixed (missing executable bit,
  unparenthesized implicit concatenation) so CI's lint job passes.

### Verified

- Claims audit wave (2026-09-21, Task 16, brief 17): the full public-claims
  vocabulary (`validated`, `supported`, `all models`, `all Radeon`,
  `zero patch(es)`, `fine-tuning`, `vLLM`, `streaming`, `305`, `0.2.0`,
  `gfx1151`, `gfx1100`) grepped across README/README_CN/CHANGELOG/docs/
  evidence/src/tests/pyproject (778 hits, every one judged — audit table in
  `evidence/claims-audit-2026-09-21.md`). Suite counts trued to the live
  recount: **313 CPU + 38 GPU = 351/351** (305 superseded; CPU half re-run
  green by the Task 15 verifier, GPU half carried by the p0 Task 5
  verifier's 38/38). PR #373 wording re-verified against the GitHub API:
  still `open`, `merged: false`. Version agreement verified
  programmatically (pyproject / `__version__` / `test_scaffold` /
  CHANGELOG all 0.2.0; no tagged 0.2.0 release exists, and the CHANGELOG
  heading now says Unreleased explicitly). Deferred minors fixed:
  `evidence/upstream-372-root-cause.md` Q2 citations (3-of-4 examples,
  `demo.py:606`), `docs/finetuning-rocm.md` fixed-wording line unwrapped
  with backticks restored, `docs/vllm-omni-rocm.md` §6 median corrected to
  the JSON's pooled 7.72 s (qwen-side 8.28 s), and an erratum appended to
  `evidence/README.md` for the stale `~20 s / ~240 tokens` figure inside
  `vllm-vs-qwen-tts-2026-09-21.json`'s fairness_notes (actual longest
  output 33.52 s ≈ 402 tokens; conclusion — no side hit any cap —
  unchanged; JSON left byte-unedited per the no-hand-edit policy).
- CI-red root cause fixed (found by the same audit): Task 14's
  `tests/test_quality_eval.py` ran 12 cer/wer tests against an unguarded
  `import jiwer` inside `scripts/quality_eval.py`, so every plain `.[dev]`
  CI environment failed with `ModuleNotFoundError` (runs 35588552252,
  35590816441, 35593965878 — 12 failures per Python job at pushes
  `c07055a`/`cfef2e7`/`bbc25db`). The tests now follow the file's own
  resemblyzer convention: a visible skip with the
  `pip install -e '.[quality]'` hint when the extras group is absent
  (verified both ways: 28 passed with the extra, 16 passed + 12 visibly
  skipped with a simulated absence; full host CPU suite unchanged at 313
  passed).
- Cross-day benchmark replication (2026-09-21, Step E): both archived sets
  re-run with unchanged methodology (warmup + n=2/cell,
  `max_new_tokens=512`, git HEAD `6df5e86`) —
  `evidence/benchmark-2026-09-21.{json,txt}` (1.7B set) and
  `evidence/benchmark-06b-2026-09-21.{json,txt}` (0.6B set), delta tables in
  `docs/benchmarks.md` ("Cross-day replication"). Variance observations
  only: the true 09-20 → 09-21 0.6B pair moved −8.7%…+2.5% per cell (load /
  peak memory ±3.5%); 1.7B custom-voice / voice-design −11.6%…+4.5%; the
  1.7B `base` alias improved 19–29% in all four cells — investigated at
  transcript level (baseline cells were measured immediately after the
  1396 s uncapped degenerate warmup, i.e. a heat-soaked iGPU, and its
  cn/medium cell sampled 28.6 s vs 16.2 s of audio); no cell deviates >2×,
  so the variance-investigation re-run budget stayed unused.
- True-streaming probe on gfx1151 (2026-09-21, Step D): the installed
  official `qwen-tts` 0.1.1 Python API exposes **no incremental-audio
  path** — all three generate_* entry points are blocking functions
  returning the complete waveform list at completion (their own
  docstrings state `non_streaming_mode=False` "only simulates streaming
  text input … rather than enabling true streaming input or streaming
  generation"). Measured on the CustomVoice 1.7B path via the new
  `scripts/streaming_probe.py` (4 scenarios x 2 runs: short 17 / long
  205 chars, both `non_streaming_mode` values): every run delivered
  exactly one audio chunk at return — TTFB == total wall (short
  3.4–4.0 s, RTF 1.22–1.25; long 66–79 s wall for 52–59 s audio,
  RTF 1.28–1.34), cadence undefined, 0 underruns only because the whole
  buffer exists at playback start. README/README_CN streaming row stays
  🚫 with the measured finding replacing "no measurements exist";
  upstream's 97 ms figure remains explicitly not-a-Radeon-number —
  `evidence/streaming-2026-09-21.{txt,json}`.
- vLLM-Omni offline-inference feasibility check on gfx1151 (roadmap rung 1):
  **FEASIBLE — single configuration only**. Installed upstream's documented
  ROCm path (`vllm==0.28.0+rocm723` from `wheels.vllm.ai` + `vllm-omni==0.28.0`
  + `onnxruntime-rocm`) in an isolated gitignored `.work-vllm/` venv (the
  validated `.venv` was not touched); the wheel embeds compiled `gfx1151`
  code objects, and the documented offline example
  (`end2end.py --query-type CustomVoice`, byte-unmodified, model loaded from
  the repo's `models/` dir via an HF cache symlink) produced a finite,
  non-silent 6.0 s / 24 kHz WAV in two consecutive runs
  (`PYTHON_EXIT_CODE=0`). README/README_CN roadmap row upgraded 🚫 → 🟡
  (offline feasibility proven only; no serving, perf or quality claims) —
  `evidence/vllm-omni-feasibility-2026-09-21.{txt,json}`.
- Filed upstream issue [QwenLM/Qwen3-TTS#372](https://github.com/QwenLM/Qwen3-TTS/issues/372):
  `finetuning/sft_12hz.py` hard-codes `attn_implementation="flash_attention_2"`, failing
  out-of-the-box on ROCm; requests a CLI-selectable attention implementation or an
  `sdpa` fallback (`evidence/upstream-issue-2026-09-21.txt`).
- Submitted upstream PR [QwenLM/Qwen3-TTS#373](https://github.com/QwenLM/Qwen3-TTS/pull/373)
  "fix(finetuning): make attention implementation configurable" (OPEN at submission,
  2026-09-21; fork `AIwork4me:fix/finetuning-attn-implementation`, commits `0be0026`
  + `48b8644` on base `022e286` — the audited pinned SHA): makes the attention
  implementation configurable in `finetuning/sft_12hz.py` while preserving the
  `flash_attention_2` default for existing users, per the root-cause evidence
  (`evidence/upstream-372-root-cause.md`, Appendix). Fixes #372 **when merged**;
  until a merged fix ships, the published `qwen-tts==0.1.1` fine-tuning path
  still requires the documented temporary workaround (one-token `sdpa` override
  at `sft_12hz.py:51`, `docs/finetuning-rocm.md`).
- First verified CI run on the 0.2.0 suite: push `8815238`, run
  35526426415 — `test (3.10/3.11/3.12)` + `build` all green,
  `251 passed, 1 skipped, 38 deselected` per Python job
  (`evidence/ci-2026-09-21-8815238.txt`); README/README_CN/P0 report now
  cite the run instead of a host-derived expectation.

### Changed

- Fine-tuning claims aligned with the upstream PR state (2026-09-21, Task 10):
  `docs/finetuning-rocm.md` gained an "Upstream fix status" section with the
  fixed wording — ROCm E2E execution proven; upstream portability fix
  submitted as PR #373 (OPEN); current published `qwen-tts==0.1.1` still
  requires the documented temporary workaround — and its stale "issue filing
  remains a human-owner action" limitation was corrected (#372 filed and
  PR #373 submitted the same day, 2026-09-21). README / README_CN
  fine-tuning rows now state the PR #373 status per the same wording and
  link the full PR-validation evidence chain (pristine double reproduction,
  root cause + controlled isolation, 3× loader validation, 2× independent
  E2E runs, default-semantics proof, diff audit, independent verifier PASS)
  instead of under-claiming only the 2026-09-20 smoke-era workaround; no
  claim exceeds "submitted", and the merged-upstream path is explicitly not
  claimed. PROVEN/NOT-PROVEN boundaries unchanged.
- Test-count truing in README / README_CN: current-state claims updated to
  the suite as it stands — 267 CPU + 38 GPU = **305/305** on the validation
  host (CPU-only CI runs the same 267 CPU tests with 1 HIP-gated skip);
  the first-verified-CI-run note (push `8815238`) is now explicitly dated
  to when the suite stood at 252 CPU tests, so its `251 passed, 1 skipped,
  38 deselected` per Python job stays accurate for its date. (Superseded
  later the same day by the claims-audit truing to 313 CPU + 38 GPU =
  351/351 — see the Task 16 entry below.)
- Claims-consistency round: runtime messages no longer make claims the
  project cannot back — the first-load announcement describes cache/kernel
  causes instead of predicting "tens of seconds", and the flash-attn guard
  error (EN+CN) is scoped to the validated wheel stack. CPU CI now runs a
  real Python 3.10 / 3.11 / 3.12 matrix (locally verified on all three
  before authoring), split into test/build jobs with `permissions:
  contents: read` and actions pinned to full commit SHAs. `install.sh`
  fails fast on an unsupported interpreter before the multi-GB ROCm wheel
  downloads (covered by `tests/test_install_sh.py`) and no longer quotes a
  drift-prone wheel size. The hardware-validation issue form warns
  non-gfx1151 testers away from the gfx1151-specific installer and now
  requires exact install commands; Docker size wording names the metric
  behind every number (inspect `.Size` / `docker history` layer /
  `docker system df -v`).

- Public-facing trust pass (EN+CN kept in lockstep): READMEs restructured
  hero-first — verified-results table up top, a minimal ~5 GB quick start
  before the full ~18 GB download, a compatibility matrix that separates the
  validated gfx1151 configuration from unvalidated hardware, collapsed
  five-tab screenshots and a curated generated audio sample linked from the
  hero. Internal development documents (design spec, implementation plan,
  acceptance record) consolidated under `docs/development/`; superseded
  Docker logs and an internal test-infra note removed from `evidence/`, with
  `evidence/README.md` added as a reproducibility index. Stale pre-remote
  notes removed from the CI workflow header; `pyproject.toml` gained
  keywords, classifiers and project URLs. GPU suite re-run green on the
  validation host after the changes (transcript:
  `evidence/gpu-suite-2026-08-29.txt`).

- Documentation and UX pass: the README was restructured with the Quickstart
  moved up front, plus a per-repository model size table, subset-download
  instructions and documented first-run log expectations; the minimal usage
  snippet now saves a wav file; `--help` output no longer renders doubled
  BooleanOptional flags.

### Fixed

- Per-tab automatic model routing: generating from any demo tab now
  auto-selects the matching model family, so a mismatched sidebar switcher
  selection can no longer produce "does not support generate_custom_voice"
  errors — a switch notice is shown instead.
- First-run terminal noise: upstream import banners (SoX, flash-attn) are
  suppressed at the file-descriptor level and sdpa experimental warnings are
  filtered, while the loader prints a bilingual first-load expectation line;
  escape hatches are `QWEN3_TTS_ROCM_VERBOSE_IMPORT=1` and
  `QWEN3_TTS_ROCM_QUIET=1`.
- CLI and script friction: `scripts/download_models.sh` gained a usage line
  and clean unknown-alias errors, `scripts/run_demo.sh` warns when models
  are missing before launch, a busy port yields a bilingual hint suggesting
  `--port`, loader/download/refusal diagnostics carry troubleshooting
  pointers, the startup banner is humanized and `scripts/install.sh` prints
  stage banners with a NEXT hint.

## [Unreleased]

### Added

- **vLLM-Omni current truth run** (v0.2.1 Task 8, 2026-09-24): the dated
  0.28.0 snapshot (2026-09-21, CustomVoice-only) superseded by the CURRENT
  stack — vllm 0.30.0+rocm723 + vllm-omni 0.30.0rc1 + onnxruntime-rocm in a
  fresh isolated venv — running upstream `end2end.py` verbatim at main
  `7e5897b…`: **all three 1.7B task families green** (CustomVoice,
  VoiceDesign, Base voice-clone; finite non-silent 24 kHz WAVs archived).
  Upstream truth captured first (main SHA re-queried live, recipe,
  supported-model table, recent qwen3_tts commits incl. the MRV2 pipeline
  optimization). Evidence:
  `evidence/vllm-omni-current-gfx1151-2026-09-24.{txt,json}` +
  `evidence/vllm-current-wavs/`. Scope unchanged: offline only; no
  serving/streaming/perf claims (Tasks 9–10 pending).

- **Quality benchmark v2** (v0.2.1 Task 7, 2026-09-24): all 10 official
  languages scored for content correctness with whisper-small (GPU-probed
  before use; never compared against v1's whisper-tiny numbers): 9/10 at
  CER/WER 0.00–0.05, German 0.5455 recorded as an honest ASR-agreement
  outlier. Clone speaker-similarity controls: positives 0.63–0.68 >
  negatives 0.58–0.60, no thresholds claimed. New
  `scripts/quality_eval_v2.py` + versioned manifest + `docs/quality-v2.md`
  (explicitly documents what the metrics do NOT measure); 13 WAVs archived
  under `evidence/quality-v2-wavs/`. No composite score, no MOS.

- **0.6B Base fine-tuning execution validation** (v0.2.1 Task 6,
  2026-09-24): the disciplined fine-tuning protocol ran twice, fully
  independently, on Qwen3-TTS-12Hz-0.6B-Base — prep → 12 optimizer steps →
  fingerprinted checkpoints → official-API reload → sane synthesis (both
  runs RELOAD-AND-SYNTHESIS-OK; walls 19.5/26.5 s; peaks 8.65 GiB).
  Execution-only: no convergence, quality, or multi-speaker claims.
  **Second upstream defect root-caused and disclosed**: `sft_12hz.py`
  adds text+codec embeddings without the talker's mandatory
  `text_projection` — a shape RuntimeError by construction on 0.6B
  (text_hidden 2048 ≠ hidden 1024; first attempt failed verbatim exactly
  there). One-line inference-faithful workaround confined to the gitignored
  fix worktree (`512db9b`); pristine clone untouched; upstream issue
  drafted, not filed (user-gated). Evidence:
  `evidence/finetune-06b-gfx1151-run{1,2}-2026-09-24.txt`;
  `docs/finetuning-rocm.md` gained the 0.6B section; fine-tuning rows in
  both READMEs now distinguish 1.7B vs 0.6B execution validation.

- **Long-text + soak stability evidence** (v0.2.1 Task 5, 2026-09-24): new
  `scripts/longtext_ladder.py` (4-tier zh/en ladder on 1.7B CustomVoice:
  8/8 green up to 892 chars / 58 s audio, natural EOS everywhere, no
  maximum-length claim — token counts are not API-observable, cap-vs-EOS
  is a duration inference) and `scripts/soak_test.py` (resident mode:
  60.04 min / **815/815 requests OK, zero failures**, RSS +0.02 GiB
  across the session, torch peak flat; recycle mode: 10/10
  load→generate→unload cycles with RSS plateau after cycle 2 — no
  "memory leak free" claim). Two honest script-bug iterations archived
  (zh `KeyError` mapping bug hit both scripts' first attempts; the stack
  itself was stable throughout). Evidence:
  `evidence/long-text-gfx1151-2026-09-24.{txt,json}`,
  `evidence/soak-gfx1151-2026-09-24.{txt,-resident.jsonl,-recycle.jsonl}`.

- **Benchmark v2 — controlled reproducibility** (v0.2.1 Task 4,
  2026-09-24): new `scripts/benchmark_v2.py` (v1 untouched and still the CI
  replication path) measured all five TTS aliases × cn/en × short/medium at
  n=5 seeded runs per cell — 20 cells, 100 measured generations, zero
  failures — with cold-start load / recorded warmup / warm-model runs
  separated, rocm-smi bookends per alias, and median/min/max/mean/stdev
  (+p10/p90 interpolation hints) per cell. RTF medians 1.04–1.52, per-cell
  stdev ≤ 0.11. New `docs/benchmarks-v2.md` documents methodology and the
  what-these-numbers-do-NOT-measure caveats; evidence
  `evidence/benchmark-v2-gfx1151-2026-09-24.{txt,json}`. Four new CPU unit
  tests pin the stats helper (suite: 317 CPU + 40 GPU).

- **Official API batch inference closure** (v0.2.1 Task 3, 2026-09-24): new
  `scripts/benchmark_batch.py` drives the official qwen-tts list-of-texts
  API (zero custom batching) for all five families — 0.6B/1.7B
  CustomVoice, 1.7B VoiceDesign, and 0.6B/1.7B Base through the reusable
  `create_voice_clone_prompt` broadcast — up the ladder B ∈ {1,2,4,8}:
  20/20 cells green, max validated batch size 8 for every family on this
  host/config (batching amortizes wall time: B=8 RTF 0.36–1.20; peak torch
  allocated up to 9.55 GiB, no OOM). Two new GPU regression tests pin the
  reusable-prompt batch path (`test_batch_clone_with_reusable_prompt`,
  0.6B + 1.7B) — suite grows to **313 CPU + 40 GPU = 353** (CI slice
  counts updated: gpu-short 34/40). Evidence:
  `evidence/batch-inference-gfx1151-2026-09-24.{txt,json}`; capability
  matrix row added in both READMEs.

- **Docker real-GPU E2E closure** (v0.2.1 Task 2, 2026-09-24): a fresh
  `docker build --no-cache` image (tag `qwen3-tts-rocm:gfx1151-e2e`, HEAD
  `488a028`) executed the complete GPU synthesis path inside the container
  on the Radeon 8060S — 17/17 checks: ROCm/HIP/GPU/arch diagnostics,
  finite bf16 matmul + SDPA, torchaudio import, repository-loader 0.6B
  CustomVoice load, one official-API synthesis (3.84 s audio @ 24 kHz,
  non-silent, WAV written), plus a 4-node GPU pytest slice in-container
  (4 passed, 38.33 s). (The first-evidence phrasing said "17/17 checks";
  the probe records 16 — a miscount corrected in the evidence index, with
  an erratum appended to the transcript.) New probe
  `scripts/docker_gpu_e2e.py` is embedded
  in the image at build time. Evidence:
  `evidence/docker-gpu-e2e-gfx1151-2026-09-24.{txt,json}` +
  `evidence/docker-gpu-e2e-gen-2026-09-24.wav` (erratum inside: the JSON's
  `generation.rtf` recorded the reciprocal throughput 0.37; repo-convention
  RTF = 2.73; probe fixed in `96bc051`). README Docker wording upgraded
  from build-validated to GPU-runtime-validated; `docker/README.md` gained
  the GPU runtime validation section.

- **First green `full-weekly` GPU CI run** (v0.2.1 Task 1, 2026-09-24):
  run `35964051504` (`gpu-nightly` / `full-weekly`, event
  `workflow_dispatch`, suite=full-weekly, commit `09ca9fd`) — every step
  green on the self-hosted `gfx1151` runner in ~12½ min: all 38 GPU test
  nodes passed in 314.08 s (including the 5 all-model load smokes + GPU
  fixture smoke that only `full-weekly` runs), `verify_gpu.sh` stack
  sanity SPIKE-GPU-OK, and RTF benchmark replication across the three 1.7B
  aliases (12 cells, median RTF 1.27–1.40). Transcript:
  `evidence/gpu-ci-full-weekly-first-green-2026-09-24.txt`. The runbook's
  go-live checklist step 5 is ✅; both READMEs' GPU CI paragraphs now
  record the full-weekly first green.

### Changed

- **`full-weekly` benchmark output persists as a run artifact.** A new
  `if: always()` step uploads `evidence/benchmark-nightly.json` via
  GitHub's first-party `actions/upload-artifact@v4` (artifact name
  `benchmark-nightly-json`; retention request 365 d is clamped by the
  repository's 90-day maximum — warning disclosed in the transcript).
  Previously the JSON was ephemeral (wiped by the next run's
  `git clean`); the runbook's ephemerality caveat is updated accordingly.
  The JSON is still deliberately not committed back into the repository.
- **GPU CI is live.** The self-hosted Radeon regression workflow
  (`.github/workflows/gpu-nightly.yml`, added prepared-but-blocked in
  0.2.0) executed its first real green run on 2026-09-23: run
  `35857806038` (`gpu-short`, event `workflow_dispatch`, commit `a3a8a75`)
  — 32/38 GPU test nodes in three disjoint slices, all passing, ~18 min
  wall including ~10 min of sync retries through the host's TLS-flaky
  github.com window (transcript:
  `evidence/gpu-ci-first-green-2026-09-23.txt`). The runner
  (amd-HP-ZBook-Ultra, labels `[self-hosted, Linux, X64,
  radeon-gfx1151]`) is a user-level systemd service
  (`~/.config/systemd/user/github-runner.service`, `ExecStart run.sh`,
  `Restart=on-failure`, linger enabled, service user `amd` — not root).
  The same hardening commit `a3a8a75` made real execution reliable:
  retrying `git fetch` sync replacing `actions/checkout` (no third-party
  actions), absolute host `.venv`/`models/` paths, a
  `PYTHONPATH=$GITHUB_WORKSPACE/src` provenance assertion (code under
  test = the pushed commit), and a `gpu-nightly` concurrency group with
  `cancel-in-progress: false`. Nightly window: `0 18 * * *` UTC = 02:00
  local (+08:00); runs when the validation host is powered/online at the
  window — a missed window can be re-dispatched manually. README /
  README_CN gained the GPU CI badge and the live-state paragraph;
  the runbook (`docs/development/gpu-ci-runbook.md`) was corrected to the
  as-built procedure (POST for the registration token, release-asset-API
  digest instead of a nonexistent `.sha256` sidecar, user-systemd + linger
  install with `sudo svc.sh` as the documented alternative,
  personal-repo security posture, current cron, tracked-file-write
  caveat, `benchmark-nightly.json` ephemerality).

## [0.1.0] - 2026-08-27

First public-ready release: a thin, zero-modification shim that runs the
official `qwen-tts` package on AMD ROCm 7.14.0 (gfx1151), with an enhanced
bilingual demo, GPU test suite and measured performance numbers.

### Added

- ROCm 7.14.0 gfx1151 support shim:
  - environment diagnostics with a never-raise guarantee, surfaced through
    the `qwen3-tts-rocm-check` CLI;
  - smart-default model loader (`loader.load()` / `loader.unload()`) that
    returns the native official Qwen3-TTS model object — all synthesis stays
    on unmodified official APIs — with a metadata-backed version probe and a
    silent clean refusal for unsupported combos;
  - dual-source model downloader (ModelScope first, hf-mirror.com fallback,
    per-alias auto mode) with friendly registry aliases for all six official
    checkpoints.
- One-command setup and launch scripts: `scripts/install.sh`
  (`--with-models` optional), `scripts/download_models.sh`,
  `scripts/run_demo.sh`, `scripts/verify_gpu.sh`.
- Enhanced bilingual (中文/English) Gradio demo application with five tabs:
  voice clone (incl. a save/load voice sub-tab), preset speakers, voice
  design, codec (encode→decode roundtrip visualization through the official
  12Hz speech tokenizer) and generation history — backed by a headless
  `SynthesisService` with server-side file validation.
- pytest suite covering unit logic plus on-GPU integration: all-model
  load/unload smoke, tokenizer encode/decode roundtrip, full custom-voice
  matrix, voice design, voice clone workflow (incl. prompt reuse and
  save/load) and waveform sanity helpers.
- Upstream parity proof: GPU integration test demonstrating that the
  upstream demo runs unmodified on this stack (thin-shim guarantee).
- Reproducible RTF benchmark tool (`scripts/benchmark.py`) with published
  methodology; results interpreted in `docs/benchmarks.md` with raw session
  and JSON evidence under `evidence/`.
- Docker image recipe and a CPU-only GitHub Actions CI pipeline
  (ruff + `pytest -m "not gpu"`); container exposes `/dev/kfd`, `/dev/dri`
  device passthrough for ROCm execution.

### Performance

- Measured real-time factors on gfx1151 (Ryzen AI Max+ PRO 395 iGPU,
  bfloat16/sdpa): tuned voices render short/medium sentences at a median
  RTF of 1.3–1.6 (lower is better); zero-shot voice cloning measured
  around RTF ~1.8. Full per-cell tables and caveats in `docs/benchmarks.md`.

### Security

- Model downloads are restricted to the two official registries with pinned
  repository ids; no third-party mirrors, no weights in the repo or wheel.
- Environment probing is strictly read-only and never raises (fail-safe by
  design) — hardware discovery failures degrade to plain reports instead of
  stack traces or partial loads.
- Demo uploads are validated on the backend before use (moved out of the UI
  layer so the guarantee holds for API clients too).
