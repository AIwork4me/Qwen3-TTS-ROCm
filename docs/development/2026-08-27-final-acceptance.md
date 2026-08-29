# Final acceptance — fresh full-suite runs @ 5e3cf37, 2026年 08月 27日 星期四 19:54:15 CST
## CPU gate
.                                                                        [100%]
145 passed, 25 deselected in 6.14s
## GPU integration (all five suites incl parity)
    with gr.Blocks(theme=theme, css=css) as demo:

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
24 passed, 146 deselected, 3 warnings in 186.14s (0:03:06)

## Packaging (v0.1.0)
- python -m build: sdist + wheel OK; twine check: PASSED x2 (dist/qwen3_tts_rocm-0.1.0-*)

## Docker (qwen3-tts-rocm:dev)
- Build EXIT=0 (2.47GB); GPU passthrough probe: SPIKE-GPU-OK inside container
- Bare `docker run` (no args, models mounted ro) serves :8900 → HTTP 200 + valid gradio 6.17.3 config
  (closes T21-review Important finding: container Gradio bind demonstrated)

## Browser E2E (real gfx1151, IAB automation @ http://127.0.0.1:8000)
| Tab | Action | Result |
|---|---|---|
| ② Preset Speakers | fill text → Generate (lazy load ~10s + synth) | Finished. (生成完成), audio player filled, VRAM 4.5 GiB |
| sidebar switch | radio → VoiceDesign, Tab ③ generate | Finished., [voice-design] loaded 已驻留, 5.60s clip, LRU switch verified |
| ① Voice Clone | x-vector only, no ref audio → Generate | graceful bilingual ValueError (Reference audio is required...), server healthy |
| ④ Codec | layout capture | screenshot only (IAB cannot upload files; roundtrip covered by Task-14 GPU tests) |
| ⑤ History | open | 2 entries newest-first (VoiceDesign 5.60s, CustomVoice 7.68s), preview/download/delete present |

Screenshots: docs/img/demo-{clone,customvoice,voicedesign,codec,history}.png (viewport, 1600x1000)
README/README_CN Screenshots sections updated from TODO-release placeholders to live embeds.

## Test totals at tag time
- CPU gate: 146 passed / 25 deselected  ·  GPU integration: 24 passed (fresh runs)
- Known deferred minors: see .superpowers ledger (parked, none release-blocking)
- Owner push steps (network-restricted machine). Placeholder step DONE: owner username
  **AIwork4me** baked into README.md / README_CN.md on 2026-08-28 (commit 199061a).
  1. `git checkout main && git merge --ff-only feat/impl-v0.1.0`
  2. `git remote add origin https://github.com/AIwork4me/Qwen3-TTS-ROCm.git`
  3. `git push -u origin main`
  4. `git push origin v0.1.0`

## Hardening pass (post-final-review, P0-S1..S6)
Five verified batches on feat/impl-v0.1.0 (each: implementer subagent → independent verifier subagent → gates):
- S1 3f793ac hygiene: five-tab wording, models comment, README precision (EN+CN)
- S2 6b74fca test-infra: make_tone/batch/fake-prompt sharp edges + downloader resume=False semantics pinned (146→156)
- S3 a0496a6 downloader: weight-aware is_downloaded(require_weights=) + skip tightening + fallback-into-partial-dir pinned (→162)
- S4 cc5608b+70a2dac+d420a0d backend: per-alias load-ticket thread safety (verified by adversarial interleaving analysis), official-faithful _normalize_audio port (bit-for-bit vs installed), ticket-leak window closed, success-path wake hardened (→170)
- S5 73f2963 docs: troubleshooting aligned to weight-aware semantics + CN polish
P0-S6 independent acceptance: 10/10 gates (ruff / 170 CPU / 24 GPU real / yaml / bash -n / build+twine PASSED / wheel payload complete / docker rebuild+SPIKE-GPU-OK+bare-serve 200 / 54 links / clean tree). Tag re-pointed to the hardened tree.

## UX audit pass (2026-08-28)
Dual-route audit (sandbox fresh-clone CLI journey + live GUI/API journey) → 16 frictions (2 Critical, 6 Annoying, 8 Nit) + 8 pleasant surprises.
Fixes (each: implementer subagent → independent verifier subagent → gates):
- B-4/521f87e per-tab automatic model routing (audited cross-kind error now auto-switches with bilingual notice; E2E-proven)
- U2/07d32e3 CLI friction: download_models.sh usage+clean errors, run_demo preflight WARN, port-busy hint rc2, troubleshooting pointers, human banner, install NEXT hint
- U3a/7df0ad9 --help dedup, loader fd-level banner suppression (VERBOSE_IMPORT/QUIET hatches), per-alias-first refusal
- U3b/1786f0a README restructure (Quickstart early), size table, subset-download docs, snippet save-lines, <OWNER> guard, troubleshooting noise entry
- d59dfb4 CHANGELOG UX notes
Final gates (independent): 10/10 ACCEPTED — 203 CPU / 24 GPU / twine / docker rebuild+SPIKE-GPU-OK / E2E cross-kind auto-switch SUCCESS (1.52s wav) / 0 banners / 0 broken links.
Tag v0.1.0 re-pointed to the UX-hardened tree.
