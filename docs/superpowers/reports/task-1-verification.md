## VERIFICATION REPORT — TASK 1
### Verdict
PASS

### Acceptance criteria
* [x] **1. Real GPU tests for 0.6B CustomVoice exist and pass.** `tests/test_generate_custom_voice_06b.py` contains `test_single` (metadata: `>=9` speakers, languages incl. case-insensitive `"auto"`; sane wav with `min_energy_rms=1e-3`), `test_batch`, `test_kwargs_passthrough` (temperature 0.01 vs 1.9, reseeded pair, robust-only shape-or-max-abs distinction), and `test_instruct_behavior` pinning the accept-and-ignore branch (no instruct-driven change asserted, sane wav required). Verifier re-run: 4 passed, exit 0, real generation timings visible (6.9/6.0/4.1/3.8/4.8 s).
* [x] **2. Real GPU tests for 0.6B Base voice cloning exist and pass.** `tests/test_voice_clone_06b.py` is a line-by-line faithful mirror of `tests/test_voice_clone_workflow.py` (verifier diffed both: identical assertions, same REF_TEXT, same `MAX_NEW_TOKENS=512`, same `_asset_path`/`_ref_audio`, same `VoiceClonePromptItem` invariants incl. `x_vector_only_mode is False`/`icl_mode is True`/`ref_code`/`ref_spk_embedding`, same official-format save (`asdict` payload) + `weights_only=True` reconstruct with field-for-field device-agnostic parity, `_distinct` on reuse and batch). Verifier re-run: 5 passed, exit 0 (`test_clone_with_ref_text`, `test_clone_x_vector_only`, `test_create_prompt_reuse`, `test_save_load_roundtrip_parity`, `test_batch_clone`).
* [x] **3. Real generated WAV sanity in every generation test.** Every render in both new modules is gated by `testing.assert_wav_sane` (CV: single/batch/instruct with `sr_expected=sr`, single+instruct additionally `min_energy_rms=1e-3` — stronger than default; clone: all 5 tests gate every wav with `sr_expected=sr`). Distinction assertions are robust-only (`_distinct`, shape-or-value). Nothing weaker than the 1.7B pattern; `git diff HEAD -- tests/` shows 459 insertions, **0 deletions** — purely additive, zero downgrades.
* [x] **4. Benchmark evidence for BOTH 0.6B models.** `evidence/benchmark-06b-2026-09-20.json` parses (verifier script, exit 0): 8 cells (2 aliases × cn/en × short/medium × 2 runs), each carrying per-run `rtf_values`/`wall_s`/`audio_s` + `load_seconds` + `peak_alloc_gb` (RTF ≡ wall/audio verified numerically); `meta.per_alias` = {custom-voice-0.6b: load 4.041 s, peak 2.802 GiB; base-0.6b: load 1.547 s, peak 3.104 GiB}; `meta.git_head` = `f913d4c...` = current HEAD; `meta.upstream_qwen3_tts_sha` = `022e286b...` (Task 0 SHA). The `.txt` transcript matches the JSON on **all 16** per-run `[bench]` lines and the full markdown table; same-run provenance confirmed (header 20:33:16, program date 20:33:18, both +08:00); bulk is genuine MIOpen/HIP kernel-tuning output; exactly 18 `pad_token_id` lines = 2 warmups + 16 measured runs (internal consistency); ends `exit_code=0`.
* [x] **5. Documentation accurately distinguishes load vs functional validation.** README.md:31 "6 / 6 **load-validated**"; per-checkpoint matrix inserted directly below the "Verified, not promised" table with exactly the six specified rows and Load/E2E Generate/Benchmark columns, 0.6B benchmark cells pointing at the JSON; boundary note "0.6B CustomVoice has no instruction control upstream". README_CN.md fully mirrors (已通过加载验证 / 加载 / 端到端合成 / 基准测试 / 边界注记). Counts 250/250 = 216 CPU + 34 GPU and CI 215 + 1 HIP-gated skip consistent in both languages; verifier grep found zero stale 238/213-CPU/212-CPU/25-GPU references and zero unqualified "6 / 6 validated" strings. CHANGELOG.md has `## [0.2.0] - 2026-09-20` describing Task 1 scope. `evidence/README.md` indexes both new artifacts plus the GPU-suite transcript; its quoted ranges (1.03–1.30, 1.19–1.26, loads 4.0/1.5 s, peaks 2.80/3.10 GiB) match the JSON.
* [x] **6. Previous 1.7B tests still pass.** Verifier re-ran `tests/test_generate_custom_voice.py tests/test_voice_design.py tests/test_voice_clone_workflow.py -m gpu -q`: **13 passed in 161.12 s, exit 0**; `tests/test_official_demo_parity.py -q`: **3 passed, exit 0** (incl. one real synthesis through the upstream callback); full CPU suite **216 passed, exit 0**.
* [x] **7. No upstream patch added.** Staged diff touches only the 11 repo files (tests/scripts/docs/evidence) — no site-packages path, no vendored upstream source. Stronger check: verifier hash-verified the installed `qwen_tts` 0.1.1 against pip's RECORD — **25/25 source files match, 0 modified** (18 unhashed rows are only `__pycache__`/RECORD). Ground truth intact: `.venv/.../qwen_tts/inference/qwen3_tts_model.py:799-800` still contains `if self.model.tts_model_size in "0b6": instruct = None`. Zero-patch parity test green.

### Code reviewed
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-20-p0-capability-parity/review-task1-f913d4c-staged.diff` (all 8,645 lines; every file section incl. full JSON and transcript)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/test_generate_custom_voice_06b.py` (via diff + tree, identical)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/test_voice_clone_06b.py` (via diff + tree)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/test_voice_clone_workflow.py` (1.7B reference, mirror comparison)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/test_generate_custom_voice.py` (1.7B reference)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/test_official_demo_parity.py`
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/conftest.py` (`gpu`, `_reseed`, `_timed_generate`, `_distinct`), `src/qwen3_tts_rocm/testing.py` (`assert_wav_sane`)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/scripts/benchmark.py` (modified sections: `build_call`, `with_alias_metrics`, peak helpers, `_git_head`, `collect_meta`, `main`, docstrings)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/test_benchmark.py` (additions)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/README.md`, `README_CN.md`, `CHANGELOG.md`, `evidence/README.md` (staged hunks + grep audits)
- `.venv/lib/python3.12/site-packages/qwen_tts/inference/qwen3_tts_model.py` lines 790–810 + RECORD hash audit
- `evidence/benchmark-06b-2026-09-20.json`, `evidence/benchmark-06b-2026-09-20.txt`, `evidence/gpu-suite-2026-09-20.txt`

### Tests executed
All from `/home/amd/Desktop/Qwen3-TTS-ROCm`, python `.venv/bin/python`:
1. `.venv/bin/python -m pytest -m 'not gpu' -q` → **216 passed, 34 deselected in 7.08s, EXIT=0**
2. `.venv/bin/python -m pytest tests/test_official_demo_parity.py -q` → **3 passed in 12.81s, EXIT=0**
3. `.venv/bin/python -m pytest tests/test_generate_custom_voice_06b.py tests/test_voice_clone_06b.py -m gpu -v -s` → **9 passed in 103.63s, EXIT=0** (4 CV 0.6B + 5 clone 0.6B, all PASSED individually with `[timing]` generation lines)
4. `.venv/bin/python -m pytest tests/test_generate_custom_voice.py tests/test_voice_design.py tests/test_voice_clone_workflow.py -m gpu -q` → **13 passed in 161.12s, EXIT=0**
5. JSON structural audit script (`json.load` + per-cell assertions) → EXIT=0
6. `-m "not gpu and not requires_download" --collect-only -q` → 216 collected (CI host: 215 pass + 1 HIP-gated skip, matching README claim) → EXIT=0
7. pip RECORD hash audit of installed `qwen_tts` 0.1.1 → 25 checked / 0 modified → EXIT=0

### Runtime evidence
- `evidence/benchmark-06b-2026-09-20.json` — parsed; 8 cells each with `rtf_values`/`wall_s`/`audio_s` per run plus `load_seconds`/`peak_alloc_gb`; `meta.per_alias`, `git_head` = `f913d4c111953fb2e2d09e6914bbcb9831d83b53` (= `git rev-parse HEAD` at verification time), `upstream_qwen3_tts_sha` = `022e286b98fbec7e1e916cb940cdf532cd9f488e`; per-alias metrics identical between `meta.per_alias`, every cell, and the printed table.
- `evidence/benchmark-06b-2026-09-20.txt` — all 16 per-run `[bench]` lines cross-checked against the JSON (wall/audio/rtf match to print precision, e.g. cv cn/short run1 3.52s/3.36s/1.05 ↔ JSON 3.522/3.36/1.0481; base en/short run2 8.04s/6.16s/1.31 ↔ 8.039/6.16/1.305); load/peak lines match (`load_seconds=4.0 peak_alloc_gb=2.80 GiB`, `1.5`/`3.10 GiB`); markdown table medians match JSON medians; same-session timestamps; ends `exit_code=0`.
- `evidence/gpu-suite-2026-09-20.txt` — "34 passed, 216 deselected in 296.14s", `exit_code=0`; consistent with the counts the verifier independently reproduced (216 CPU here; 25 of the 34 GPU tests re-run green across commands 2–4).
- `git status --porcelain` — exactly the 11 staged files (7 A / 4 M), nothing unstaged, nothing in site-packages, HEAD `f913d4c`.

### Claims audit
- "6 / 6 load-validated" / "已通过加载验证": the only three "6 / 6" occurrences (README.md:31, README_CN.md:25, CHANGELOG.md:33) all carry the load qualifier; the per-checkpoint matrix separates Load / E2E Generate / Benchmark so no cell implies more than was measured.
- Test counts: 250/250 = 216 CPU + 34 GPU appears identically in both READMEs (hero table, CI paragraph, What-You-Get bullet, re-run proof block); 215 CI + 1 HIP skip verified via collect-only; grep for stale `238`/`213 CPU`/`212 CPU`/`25 GPU`/`25 个 GPU` across READMEs and docs: zero hits.
- Boundary claim: "0.6B CustomVoice has no instruction control upstream; the demo and docs reflect that boundary" (both languages) matches the verified installed-source ground truth (qwen3_tts_model.py:799-800 nulls instruct for 0.6B) and the pinned test behavior; CHANGELOG wording ("accepted but silently ignored by the installed wrapper") is precise.
- Benchmark claims in `evidence/README.md` and the report (median RTF ranges, loads, peaks) all recompute exactly from the JSON; RTF numbers are labeled benchmark measurements, not quality claims.
- No pronunciation/quality claims and no subjective 0.6B-vs-1.7B quality comparison anywhere in README.md, README_CN.md, CHANGELOG.md, evidence/README.md (grep for pronunciation/sounds better/better quality/音质更好/优于 etc.: zero hits). The only 0.6B-vs-1.7B statements are RTF/memory comparisons traceable to archived benchmarks, which is permitted.
- `max_new_tokens=512` is passed on **every** clone/generation call in both new suites (named constant, value 512) and in the benchmark's measured cells (`args.max_new_tokens=512` in JSON meta); calls are keyword-first throughout.

### Regression audit
- 1.7B trio (custom-voice, voice-design, clone workflow): 13/13 green on GPU in this verification (161.12s, exit 0).
- Upstream-parity proof: 3/3 green (exit 0) — stock `qwen_tts.cli.demo.build_demo` constructs, its Gradio-registered closure executes, one real synthesis passes `assert_wav_sane`.
- Full CPU suite: 216/216 green (exit 0); previously-validated 213 + 3 new additive benchmark unit tests, no deletions (0 removed lines under tests/).
- Installed `qwen_tts` 0.1.1 verified byte-identical to pip RECORD (25/25 files) — the zero-patch guarantee holds after Task 1.

### Problems found
1. (Minor, disclosed) `evidence/benchmark-06b-2026-09-20.txt` and `gpu-suite-2026-09-20.txt` have short machine-generated headers around the `tee` capture rather than being raw command output from byte zero; the implementer disclosed this (report §5.7) and the benchmark payload itself (the load-bearing part) is the verbatim piped output. Internal consistency checks (16/16 run lines ↔ JSON, 18 pad_token_id warnings = 2 warmups + 16 runs, matching timestamps) show a genuine single run.
2. (Minor, disclosed) Extra file `evidence/gpu-suite-2026-09-20.txt` beyond the brief's file list — justified by the README count-traceability rule, follows the existing `gpu-suite-*.txt` convention, disclosed as deviation §5.4.
3. (Minor, disclosed) CHANGELOG version heading `0.2.0` chosen by the implementer (brief mandated only "a new version heading"); disclosed as deviation §5.5, orchestrator may re-version.
4. (Observation, no action) The pre-existing 1.7B CV suite does not pass `max_new_tokens` on its generate calls while the new 0.6B suite does; the 1.7B files are untouched by this diff and the constraint binds the new/changed code, which complies. Not a Task 1 defect.
5. (Observation, no action) The verifier re-ran 25 of the 34 GPU tests individually (all Task-1-relevant ones per instructions); the remaining 9 (e.g. loader/tokenizer smoke suites, untouched by this diff) are green in the archived 2026-09-20 transcript.

No discrepancy between the implementer's report and the artifacts was found: every number in the report's §2 table, §3 summary and §4 counts was independently reproduced or traced to a staged artifact.

### Final justification
All seven acceptance criteria are satisfied with independently executed evidence: the new 0.6B suites pass on the real gfx1151 GPU with genuine synthesis timings (9/9), every generation is gated by the full `assert_wav_sane` net with zero assertion downgrades (0 deleted lines under tests/), the benchmark JSON parses and carries every required per-cell and per-alias metric and matches its transcript cell-for-cell from the same exit-0 run, both READMEs and the CHANGELOG now make claims exactly as strong as the archived evidence (load vs E2E vs benchmark separated, no quality or subjective comparison claims, all counts consistent in EN and CN), the 1.7B regression trio plus parity proof plus full CPU suite are green on re-run, and the installed upstream package is byte-identical to pip's RECORD with the documented instruct ground truth intact. The five findings above are minor, disclosed by the implementer, and none weakens an acceptance criterion.
