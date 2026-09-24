# TASK 3 INDEPENDENT VERIFICATION — OFFICIAL API BATCH INFERENCE CLOSURE (v0.2.1)

**Verdict: PASS_WITH_CONCERNS** — every substantive gate reproduces independently (JSON integrity, official-API-only script, live B=1/B=8 rerun on two families, both new GPU tests, all suite counts, evidence-transcript cross-check, no overclaims, historical docs untouched). One minor, purely documentary discrepancy found: the runbook's filter→node mapping table was not updated with the +2 batch tests and still says the clone slice is 10 nodes / 22 slice total / "32 nodes total", contradicting the same file's own updated job table (34 of 40), the workflow yml comments (24), and live collection (24). It understates coverage rather than overstates it, but it is a factual count error inside a tracked doc touched by this task's commit — same class as Task 2's round-1 count discrepancy. Recommended: a one-line fix-round commit truing 10→12, 22→24, 32→34 and adding `test_batch_clone_with_reusable_prompt` to the enumerated clone tests; no code or evidence changes needed.

Verifier: independent subagent, 2026-09-24. Repo at HEAD `15c80c7` (branch main), working tree clean except untracked `.work-docker/`. No tracked files modified, nothing committed. GPU work executed on this validation host (Radeon 8060S, gfx1151).

---

## Check 1 — Evidence JSON integrity and metric recomputation: PASS

Parsed `evidence/batch-inference-gfx1151-2026-09-24.json` with a fresh script (not the implementer's tooling):

- `meta.git_head = 0824e1116584094f2364bdc30bd34f6119621dba` (matches the script's commit `0824e11`), `seed = 20260924`, `max_new_tokens = 512`, `gpu = AMD Radeon 8060S Graphics (gfx1151)`.
- `meta.results_count = 20`, actual `len(results) = 20`; families = {base, base-0.6b, custom-voice, custom-voice-0.6b, voice-design} × batch sizes {1,2,4,8} — exactly 20 distinct cells, `all ok: True`, `per_family_max_stable = {custom-voice-0.6b: 8, custom-voice: 8, voice-design: 8, base-0.6b: 8, base: 8}`.
- Recomputed from raw fields on **all 20 rows** (stronger than the required 3): `rtf == round(wall/total_audio, 2)`, `audio_per_wall_second == round(total/wall, 2)`, `sum(audio_seconds_per_item) ≈ total_audio_seconds`, `len(per_item) == B`, `sample_rate == 24000`, `peak_alloc_gb`/`peak_reserved_gb` present on every row — **zero discrepancies**.
- Duration range across all 75 items: 2.0–40.88 s (within the script's [0.5, 60] gate); the 40.88 s items are the clone-mode items near the 512-token cap, as disclosed.
- Key claim numbers verified verbatim in the JSON: B=8 RTF 0.36 / 0.48 / 0.47 / 1.20 (base-0.6b; its B=4 = 0.63) / 0.50; max peak alloc 9.55 GiB (base B=8); peak reserved 13.72 GiB on the same cell.

## Check 2 — Script audit (`scripts/benchmark_batch.py`): PASS

- **Official APIs only**: every model call is `model.generate_custom_voice` / `model.generate_voice_design` / `model.generate_voice_clone` / `model.create_voice_clone_prompt` / `model.get_supported_speakers` (lines 127–179). No `_tokenize*`, no `_prompt_items_to_voice_clone_prompt`, no direct `qwen_tts` import in the script.
- **No wrapper/proxy**: `loader.load()` (src/qwen3_tts_rocm/loader.py:299) returns `qwen_tts.Qwen3TTSModel.from_pretrained(...)` — the native official object. `compat.apply_compat_patches()` is a version advisory only: `PATCHES = []` (src/qwen3_tts_rocm/compat.py:52), "Nothing in this module vendors, subclasses or monkey-patches upstream code". No monkey-patching or vendored code anywhere on the path.
- **Batching is the official package's own**: verified in the installed official wheel (`.venv/.../qwen_tts/inference/qwen3_tts_model.py`) — `_ensure_list` batching in `generate_custom_voice` (l.796), `generate_voice_design` (l.694), `generate_voice_clone` (l.556), and the 1-item prompt broadcast `prompt_items = prompt_items * len(texts)` (l.574–575) that the reusable-clone-prompt path relies on. This is upstream behavior, not a local batching layer.
- **Ladder policy**: stop-on-first-failure implemented as `break` after appending a failure record carrying `error` + full `error_verbatim` traceback (lines 282–291); `per_family_max_stable` set from the last succeeded size; a family failing at B=1 forces exit 1 (lines 325–328).
- **Sanity gate** (lines 195–205): `len(wavs)==B`, `sr==24000`, all-finite, per-item RMS ≥ 1e-3, durations in [0.5, 60] s — violation raises RuntimeError → recorded verbatim → ladder stops. Warmup per family is discarded and itself sanity-checked; `torch.manual_seed(seed)` before every batch call; `reset_peak_memory_stats()` per cell; rocm-smi VRAM sampled at family boundaries; machine-readable JSON + markdown table out.

## Check 3 — INDEPENDENT RERUN (core gate): PASS

Command: `.venv/bin/python scripts/benchmark_batch.py --families custom-voice,base-0.6b --batch-sizes 1,8 --json-out /tmp/task3-verify-batch.json` → **exit 0**, `BATCH-BENCHMARK-OK`, 4/4 rows ok, `per_family_max_stable = {custom-voice: 8, base-0.6b: 8}` (JSON meta.git_head = 15c80c7). Verbatim key lines:

```
[batch] OK family=custom-voice B=1 wall=3.93s audio_total=3.12s rtf=1.26 peak_alloc=4.06 GiB
[batch] OK family=custom-voice B=8 wall=6.14s audio_total=22.96s rtf=0.27 peak_alloc=4.85 GiB
[batch] OK family=base-0.6b  B=1 wall=8.69s audio_total=8.0s   rtf=1.09 peak_alloc=2.44 GiB
[batch] OK family=base-0.6b  B=8 wall=70.28s audio_total=100.64s rtf=0.7 peak_alloc=7.65 GiB
```

Rerun JSON additionally confirms per-item durations, sr 24000, peak reserved 4.11/5.06/2.55/8.65 GiB. Observations: wall times differ from the archive as forewarned (cv B=8 6.14 s vs 11.13 s; base-0.6b B=8 70.28 s vs 120.60 s — host-load/MIOpen-cache variance), while **audio totals reproduce exactly** (3.12 / 22.96 / 8.0 / 100.64 s, including one 40.88 s clone item near the token cap) because the fixed seed makes generation deterministic — stronger reproducibility than expected. Peak alloc matched the archive within 0.01 GiB on all four cells.

## Check 4 — Independent rerun of the two new GPU tests: PASS

`.venv/bin/python -m pytest -m gpu tests/test_voice_clone_06b.py::test_batch_clone_with_reusable_prompt tests/test_voice_clone_workflow.py::test_batch_clone_with_reusable_prompt -q` → **`2 passed, 2 warnings in 40.66s`** (implementer: 45.90 s). Both tests are real GPU regression tests: B=2 text list over one reused `create_voice_clone_prompt` item, asserting `len(wavs)==2`, per-wav `testing.assert_wav_sane`, and distinct outputs — i.e. official broadcast shape + output correctness, as required.

## Check 5 — Counts: PASS

- `-m gpu --co -q` → **40/353 tests collected (313 deselected)**.
- Slice `custom_voice or voice_design or voice_clone` → **24**; `official_demo or tokenizer` → **6**; `voice_workflow` → **4** (sum 34 = gpu-short; the complementary 6 = 5×`test_loader_all_models` load smokes + `test_gpu_fixture_smoke`, matching the documented full-weekly-only set).
- `-m "not gpu" -q` → **313 passed, 40 deselected in 17.41s**.
- The yml slice expressions (`gpu-nightly.yml` l.137/143/149) are byte-identical to the filters I collected against.

## Check 6 — Docs audit at 15c80c7: PASS except one minor discrepancy (D1 below)

- **README.md:66 / README_CN.md:58 capability row**: "B ∈ {1,2,4,8} all green, max validated B = 8 for all five families (this host/config only)" — accurate vs the JSON, properly host-scoped, no claim of B>8 or other GPUs. Test-count rows "353/353 — 313 CPU + 40 real-GPU (40 since 2026-09-24: +2 reusable-prompt batch tests)" present in both READMEs (README.md:43, README_CN.md:36/156). README.md:177 keeps the historical 38/351 fact with an explicit growth note. Overclaim scan (B=16 / any GPU / other archs) found nothing; the only gfx1100 mentions are historical audit records.
- **Workflow comments** (gpu-nightly.yml l.31–47): "38 GPU nodes on 2026-09-21, 40 since 2026-09-24 — +2 reusable-prompt batch tests"; "gpu-short = 34 of the 40 nodes" with slices 6/24/4 including "both B=2 batch tests in the clone suites" — matches my live collection exactly.
- **Runbook** (docs/development/gpu-ci-runbook.md): job table and header updated to 34-of-40 / 40-nodes (verified via `git show 15c80c7`), timeout note updated. **BUT** the "Suite selection — filter → node mapping" table was left stale: clone row still says 10 nodes enumerating only 5 tests per clone file (no `test_batch_clone_with_reusable_prompt`), and the slice summary still reads "→ **22** nodes (8 CustomVoice + 4 VoiceDesign + 10 clone)" and "disjoint slices, 32 nodes total" — contradicting the same file's own job table (34), the yml (24), and reality (24). See D1.
- **Evidence index** (evidence/README.md:88–89): every number cross-checks against the JSON — 20/20 green, all-8 max, B=8 RTFs 0.36/0.48/0.47/1.20 (B=4 0.63)/0.50, peak 2.18→9.55 GiB, 512-token-cap clone items disclosed, and an explicit "Scope: this host/config only; B > 8 NOT tested".
- **CHANGELOG.md:347–360**: accurate (families, ladder, RTF range 0.36–1.20, 9.55 GiB, 313+40=353, gpu-short 34/40, evidence paths).
- **Historical doc untouched**: docs/radeon-reference-closure-v0.2.md last modified in `d6f9d44` (pre-Task-3) and still states 313+38=351 as its dated record — correctly not rewritten.

## Check 7 — Transcript vs JSON cross-check: PASS

All 20 `[batch] OK family=… B=… wall=… audio_total=… rtf=… peak_alloc=…` lines in `evidence/batch-inference-gfx1151-2026-09-24.txt` match the JSON per-cell values exactly (checked wall_seconds, total_audio_seconds, rtf, peak_alloc_gb for every cell; e.g. base B=8 wall=78.45s rtf=0.5 peak=9.55 both places). The embedded RESULT TABLE is verbatim-identical to the JSON rows; five `max_stable_batch=8` family lines match `per_family_max_stable`; the transcript ends with the same disclosure notes (variable clone durations, VRAM sampling caveat, host-scoped, B>8 not tested). Cosmetic only: the token `BATCH-BENCHMARK-OK` appears twice (stdout tail duplicated by the capture method), and the transcript honestly notes the full raw log lived in a /tmp file with MIOpen chatter omitted.

---

## Discrepancies

- **D1 (minor, docs-only, requires one-line fix round)**: `docs/development/gpu-ci-runbook.md` "Suite selection — filter → node mapping" table is internally inconsistent after 15c80c7's partial update: clone row "10" nodes enumerating 5 tests per clone file, slice-2 summary "**22** nodes", "32 nodes total" — vs the correct 12/24/34 (verified by live collection: the slice includes `test_voice_clone_06b.py::test_batch_clone_with_reusable_prompt` and `test_voice_clone_workflow.py::test_batch_clone_with_reusable_prompt`) and vs the same file's own job-table row "34 of the 40 GPU test nodes". Direction of error: understates coverage; no capability or evidence claim is affected. All other counts everywhere (yml, READMEs, CHANGELOG, evidence index) are consistent.

## Observations

- Determinism bonus: with seed 20260924 the rerun reproduced the archived audio durations exactly (e.g. base-0.6b B=8 per-item [10.8, 8.56, 10.08, 10.48, 5.76, 40.88, 7.12, 6.96]); only wall times moved. The implementer's "stochastic, numbers will differ" caveat was overly conservative — audio content is stable, walls are not.
- base-0.6b B=8 remains the throughput outlier (RTF 1.20 archived / 0.70 in my rerun) because one clone item runs to the 512-token cap (~40.9 s); this is disclosed in the evidence and bounded by the official guardrail, and the peak-reserved headroom (13.72 GiB max vs 32 GiB) leaves margin, but B>8 remains genuinely untested as stated.
- The script's B=1-failure escape hatch (exit 1) and verbatim failure recording were not exercised live (no failures occurred); their correctness is established by code reading only.

---

## Fix round 1 re-verification (D1) — commit `029aa76`

Scoped re-check of the sole discrepancy, performed 2026-09-24 at HEAD `029aa76`:

- **Only the flagged file changed**: `git show 029aa76 --stat` → `docs/development/gpu-ci-runbook.md | 9 +++++----` (5 insertions, 4 deletions), 1 file changed, nothing else touched. `git status` shows no other tracked modifications (only pre-existing untracked scratch + this report).
- **Diff content is exactly the D1 fix**: clone row count 10 → **12** with `test_batch_clone_with_reusable_prompt` now enumerated in both clone files and a dated note "(the last one added 2026-09-24, v0.2.1 Task 3)"; three-step summary "32 nodes total" → "**34** nodes total since 2026-09-24 — 32 before the two v0.2.1 Task 3 batch tests"; slice-2 "**22** nodes (8 CustomVoice + 4 VoiceDesign + 10 clone)" → "**24** nodes (8 CustomVoice + 4 VoiceDesign + 12 clone)". The enumerated per-file test list is now 6 tests × 2 files = 12, arithmetically consistent.
- **Numbers match fresh live collection at this HEAD**: `-m gpu --co -q` → **40/353** collected; slice-2 `custom_voice or voice_design or voice_clone` → **24/353** (grep confirms both `test_batch_clone_with_reusable_prompt` nodes inside the slice); slice-1 `official_demo or tokenizer` → **6/353**; slice-3 `voice_workflow` → **4/353**; 6+24+4 = 34, and 40−34 = 6 = the unchanged full-weekly-only set (5 load smokes + fixture smoke). The runbook's job table (34 of 40), mapping table, summary, and gpu-nightly.yml comments are now mutually consistent and consistent with reality.
- No code, tests, workflow yml, READMEs, CHANGELOG, or evidence files changed in this commit; the round-1 verification results (checks 1–7, independent reruns) are unaffected by a docs-only change.

**Final verdict: PASS** (round-1 concern D1 resolved; all substantive gates had already reproduced independently).
