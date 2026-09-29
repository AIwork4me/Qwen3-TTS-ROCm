# Upstream PR #336 — independent gfx1100 validation (v0.3 Task 14)

PR: QwenLM/Qwen3-TTS#336 — "Fix 0.6B fine-tuning crash: project text
embedding before adding to codec embeddings"

## PR state at validation time (GitHub API, 2026-09-29)

| Field | Value |
|---|---|
| state | **open** (not merged) |
| head | `701938bb6bdf22c091ec0a0952990bf9b7ae457d` (1 commit) |
| base | `022e286b98fbec7e1e916cb940cdf532cd9f488e` (= upstream main at validation time) |
| mergeable | true (`clean`) |
| comments / review comments / reviews | 0 / 0 / 0 |

This document validates the EXACT PR head. It does not create a duplicate
PR and does not claim the PR is merged.

## Validation protocol A–F and evidence (executed 2026-09-29, real GPU)

All evidence: `evidence/finetune-06b-gfx1100-v0.3/` (raw logs + SHA256SUMS).
Hardware: AMD Radeon Pro W7900D (`gfx1100`, 48 GB), Ubuntu 24.04.4,
kernel 6.8.0-79-generic; stack torch 2.12.0+rocm7.14.0 / HIP 7.14.60850
(`device-gfx1100` extras, repo `.venv`). Upstream clones live in the
gitignored `.upstream/`; the repository source was never modified.

- **A. Failure reproduced on the relevant upstream baseline.** Upstream
  `022e286` (= the PR's own base; tree verified 37/37 blobs vs the GitHub
  API) with only the separately-documented attention workaround
  (`flash_attention_2` → `sdpa`, PR #373's issue, diff archived) fails
  0.6B training VERBATIM at `sft_12hz.py:93`:
  `RuntimeError: The size of tensor a (2048) must match the size of tensor
  b (1024) at non-singleton dimension 2` — exactly the defect the PR
  fixes. (`02-baseline-failure.txt`)
- **B. Exact PR head removes the failure.** PR head `701938b` (tree
  verified 37/37 vs the API; the projection fix is inside the committed
  import — `text_projection` applied before the embedding add) + the same
  sdpa workaround trains past the previous crash point. The ONLY local
  change to the PR-head worktree is the sdpa line (diff archived:
  `03-pr336-worktree.diff`).
- **C. Finite steps.** 12 optimizer steps (12-sample self-generated
  dataset, batch 2, 2 epochs, upstream default lr 2e-5); loss lines
  verbatim: `Epoch 0 | Step 0 | Loss: 14.7048`, `Epoch 1 | Step 0 | Loss:
  6.7610`. (`04-pr336-train.txt`)
- **D. Checkpoint save succeeds.** `checkpoint-epoch-0/1` written
  (model.safetensors 1.81 GB each).
- **E. Reload succeeds.** `Qwen3TTSModel.from_pretrained(checkpoint)` via
  the repository loader (sdpa default) loads the trained checkpoint.
- **F. Synthesis sanity passes.** Official-API `generate_custom_voice`
  with the trained speaker: 11.60 s @ 24 000 Hz, peak 1.0000,
  rms 0.3097, `assert_wav_sane` PASS — `RELOAD-AND-SYNTHESIS-OK`.
  (`05-reload.txt`)

Peak training memory 8.66 GiB allocated / 8.99 GiB reserved; wall 20.0 s.
Scope: **execution validation only** — no convergence, quality,
hyperparameter or production-training claims; this is not a subjective
speech-quality assessment.

## Prepared comment (NOT posted — human maintainer decides)

> Independently validated this fix on AMD ROCm hardware (Radeon Pro
> W7900D, gfx1100; ROCm 7.14 wheels, torch 2.12.0+rocm7.14.0, HIP
> 7.14.60850, Python 3.12):
>
> - On this PR's base (`022e286`), 0.6B SFT crashes at
>   `sft_12hz.py:93` with `RuntimeError: The size of tensor a (2048) must
>   match the size of tensor b (1024)` — reproduced verbatim.
> - On this PR's head (`701938b`), the same 12-sample run trains end to
>   end: 12 optimizer steps, finite losses, per-epoch checkpoints saved,
>   checkpoint reload via `Qwen3TTSModel.from_pretrained`, and a sane
>   post-finetune synthesis (24 kHz, non-silent waveform).
> - Peak training memory 8.66 GiB on 0.6B (for reference, 18.02 GiB on
>   1.7B with the same script).
>
> Only deviation: the attention implementation had to be `sdpa` (the
> script's `flash_attention_2` default doesn't work on this stack —
> separate issue #372/PR #373 territory), applied locally outside this PR.
> Execution validation only; no quality claims.

## Provenance chain

- upstream base tree `022e286`: 37/37 blobs == GitHub API tree
- upstream PR-head tree `701938b`: 37/37 blobs == GitHub API tree
- local worktree diff vs PR head: the single sdpa line only
- repo source: `git status --porcelain` clean throughout
- raw transcripts + SHA256SUMS in `evidence/finetune-06b-gfx1100-v0.3/`
