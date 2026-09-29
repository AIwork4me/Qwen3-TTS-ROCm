# Task 12 — gfx1100 durability & quality replication (v0.3)

All runs on the real W7900D / gfx1100 with the pinned repo `.venv` stack
(torch 2.12.0+rocm7.14.0 / HIP 7.14.60850), models via the documented
`QWEN3_TTS_ROCM_MODELS_DIR`. gfx1151 thresholds were NOT copied; each
methodology match or difference is stated per run.

## 12-1 long-text ladder — **REPLICATED, same methodology**

`scripts/longtext_ladder.py` (custom-voice 1.7B, seed 20260924, identical
invocation to the gfx1151 2026-09-24 run): **8/8 tiers green** up to
892 chars / 57.84 s audio, natural EOS at every tier (hit_cap=False
everywhere), RTF 1.22–1.24 on the long tiers, peak 5.28 GiB, RSS 2.86–2.91
GiB. (gfx1151 comparand: 8/8 to 892 chars / 58 s.) No max-length claim —
token counts are not API-observable.

## 12-2 load→generate→unload recycle — **REPLICATED, same methodology**

`scripts/soak_test.py --mode recycle --cycles 10`: **10/10 cycles OK**,
per-cycle load ≈ 2.8–2.9 s, wall ≈ 3.9–4.4 s, RSS after unload stable at
3.03–3.04 GiB (plateau, same shape as the gfx1151 evidence; no leak-free
claim).

## 12-3 resident soak — **REPLICATED with disclosed methodology difference**

`scripts/soak_test.py --mode resident`: **307/307 requests OK, zero
failures** in 20.1 minutes, RSS flat at 2.89 GiB, peak 4.06 GiB constant.
**Methodology difference (disclosed):** the gfx1151 reference ran 60
minutes; this run is 20 minutes (request cadence, assertions and script
identical). No cross-duration threshold claims are made.

## 12-4 multilingual matrix — **REPLICATED, same methodology**

`scripts/validate_languages.py` (10 CustomVoice + 10 VoiceDesign cells +
4 cross-lingual clone references, one model resident at a time):
**24/24 matrix cells + 4/4 reference generations passed**, wall 151.2 s.
Waveform-sanity claim only — NOT a pronunciation-quality claim (same
wording as the gfx1151 matrix).

## 12-5 quality benchmark v2 — **BLOCKED in this environment (dependency
unavailable)**

`scripts/quality_eval_v2.py` requires openai-whisper checkpoint weights
(whisper-small). The weight host `openaipublic.azureedge.net` is 403-blocked
by this environment's egress proxy (probe archived in
`05-quality-v2.txt`); reachable mirrors (ModelScope `openai-mirror`,
HuggingFace `openai/whisper-small`) carry HF-format weights
(model.safetensors / pytorch_model.bin) rather than the original
`small.pt` checkpoint openai-whisper loads, and no format-faithful copy of
the original checkpoint is obtainable here. The clone-similarity control
(resemblyzer) alone would be a partial, methodologically different run, so
no quality-v2 numbers are claimed for gfx1100. The gfx1151 quality-v2
results remain historical gfx1151 evidence and are not transferred.

## Summary

| Rung | Status | Methodology |
|---|---|---|
| long-text ladder | ✅ 8/8 | identical |
| recycle cycles | ✅ 10/10 | identical |
| resident soak | ✅ 307/307, 0 fail | 20 min vs reference 60 min — disclosed |
| multilingual matrix | ✅ 24/24 + 4/4 | identical |
| quality v2 | 🚫 blocked (whisper weights unreachable) | n/a — no numbers claimed |

## Errata / hygiene (post-verification, round 1 — appended)

1. "peak 4.06 GiB constant": the full JSONL range is 4.03–4.19 GiB
   (no upward trend; the tail lines read 4.06). Cosmetic over-rounding.
2. The multilingual script's DEFAULT output paths wrote gfx1100 results
   over the worktree copies of the historical
   `evidence/multilingual-matrix.{txt,json}` (gfx1151, 2026-09-20). The
   historical files were restored from git immediately after the round-1
   verification flagged it; the gfx1100 results live ONLY under this
   evidence dir (multilingual-gfx1100-2026-09-29.txt +
   multilingual-matrix.json here, copied before restoration). The
   historical gfx1151 artifacts in git were never modified or committed
   over.
