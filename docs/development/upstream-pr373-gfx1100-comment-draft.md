# Upstream PR #373 — independent gfx1100 validation (v0.3 Task 15)

PR: QwenLM/Qwen3-TTS#373 — "fix(finetuning): make attention implementation
configurable" (fixes #372)

## PR state at validation time (GitHub API, 2026-09-29)

| Field | Value |
|---|---|
| state | **open** (not merged) |
| head | `48b8644aac8512e6fdc0f4442788bf399c425fdb` (2 commits) |
| base | `022e286b98fbec7e1e916cb940cdf532cd9f488e` (= upstream main at validation time) |
| mergeable | true (`clean`) |
| comments / reviews | 0 / 0 |

The EXACT PR head was fetched and tree-verified (37/37 blobs == the GitHub
API tree of `48b8644`), imported into the gitignored
`.upstream/Qwen3-TTS-pr373`. **The worktree stayed PRISTINE for the entire
validation — zero local edits** (`git status --porcelain` empty after all
runs), because the PR's own CLI flag selects the attention backend.

## Validation evidence (executed 2026-09-29, real GPU; raw logs +
SHA256SUMS in `evidence/finetune-pr373-gfx1100-v0.3/`)

Hardware/stack: AMD Radeon Pro W7900D (`gfx1100`, 48 GB), Ubuntu 24.04.4,
kernel 6.8.0-79-generic, Python 3.12.3, torch 2.12.0+rocm7.14.0 /
HIP 7.14.60850 (`device-gfx1100` extras).

1. **No flag ⇒ original/default `flash_attention_2` behavior preserved.**
   `python sft_12hz.py …` (defaults) fails on this ROCm stack with the
   verbatim `ImportError: FlashAttention2 has been toggled on, but it
   cannot be used due to the following error: the package flash_attn seems
   to be not installed…` — i.e. the PR did NOT change default semantics;
   the pre-PR failure mode is intact. (`01-default-flashattn.txt`,
   exit 1.)
2. **Explicit `--attn_implementation sdpa` reaches the correct loader
   path.** The same invocation plus the flag drives
   `Qwen3TTSModel.from_pretrained(…, attn_implementation="sdpa")` (source
   line 56 of the PR head) and the **ROCm model load works** — training
   completes the full smoke: 12 optimizer steps (12-sample self-generated
   dataset, batch 2, 2 epochs, upstream-default lr 2e-5), verbatim loss
   lines `Epoch 0 | Step 0 | Loss: 13.5619` / `Epoch 1 | Step 0 | Loss:
   10.7977`, wall 40.1 s, peak 18.01 GiB allocated / 18.40 GiB reserved,
   `TRAIN-OK`. (`02-sdpa-train.txt`)
3. **Checkpoint save + reload succeed.** `checkpoint-epoch-{0,1}` written;
   reload through `Qwen3TTSModel.from_pretrained` (repo loader, sdpa
   default) OK. (`03-reload.txt`)
4. **Synthesis sanity passes.** Official-API synthesis with the trained
   speaker: 3.04 s @ 24 000 Hz, peak 0.3477, rms 0.0504, `assert_wav_sane`
   PASS — `RELOAD-AND-SYNTHESIS-OK`.
5. **Relevant tests.** The repository's own architecture/behavior suites
   remained green throughout (CPU 358 passed; GPU 40 passed during this
   program — see `evidence/gfx1100-v0.3/`); the PR touches only
   `finetuning/sft_12hz.py`, which has no upstream test coverage to run.

Scope: **execution validation only** — no convergence/quality claims, no
claim that the PR is merged.

## Prepared comment (NOT posted — human maintainer decides)

> Independently validated this PR on AMD ROCm hardware (Radeon Pro W7900D,
> gfx1100; ROCm 7.14 wheels, torch 2.12.0+rocm7.14.0, HIP 7.14.60850,
> Python 3.12):
>
> - Default behavior is unchanged: with no flag the script still requests
>   `flash_attention_2` and fails on this stack with the original
>   `ImportError … flash_attn seems to be not installed` (verbatim).
> - With `--attn_implementation sdpa` the full fine-tuning smoke runs end
>   to end on this PR's head (`48b8644`): model load, 12 optimizer steps
>   with finite losses, per-epoch checkpoints saved, checkpoint reload via
>   `Qwen3TTSModel.from_pretrained`, and a sane post-finetune synthesis
>   (24 kHz, non-silent). Peak training memory 18.01 GiB (1.7B Base).
> - No local modifications were needed to run the PR head — the flag alone
>   is enough on ROCm.
>
> Execution validation only; no quality claims. Thanks for the fix.

## Provenance chain

- PR-head tree `48b8644`: 37/37 blobs == GitHub API tree
- worktree: `git status --porcelain` EMPTY after all runs (no edits)
- repo source: clean throughout (`git status --porcelain` empty)
- raw transcripts + SHA256SUMS in `evidence/finetune-pr373-gfx1100-v0.3/`
