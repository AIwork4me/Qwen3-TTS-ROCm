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
- Owner push steps (network-restricted machine):
  **Before pushing, replace the `<OWNER>` placeholder in the clone commands in
  README.md / README_CN.md with the real GitHub username.**
  1. `git checkout main && git merge --ff-only feat/impl-v0.1.0`
  2. `git remote add origin https://github.com/<OWNER>/Qwen3-TTS-ROCm.git`
  3. `git push -u origin main`
  4. `git push origin v0.1.0`
