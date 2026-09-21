# Task 17 (brief 18) — Independent Release Verification — Radeon Reference Closure v0.2

Date: 2026-09-21 (20:24–20:55 +08:00). Verifier role: adversarial — attempt to falsify the
implementer's regression claims; PASS only on executable or runtime evidence.

## 1. Verdict

**PASS** — all 9 acceptance criteria independently confirmed (CPU suite re-run, GPU
representative slice re-run, lint re-run, wheel re-inspected from bytes, upstream clones
re-audited, remote SHA confirmed over the network, fine-tune/vLLM scratch artifacts
re-checked on disk). Zero fabrication indicators; every implementer-report claim traced to
transcript lines, on-disk artifacts, or my own execution.

## 2. Exact commit / working-tree SHA reviewed

- HEAD under review: `d389786edeb21c619d844254be3eb23268def36f` (`test: final full
  regression on gfx1151 before v0.2.0`), parent `20e148e`.
- Diff range reviewed: `20e148e..d389786` = 3 files, 6678 insertions
  (`evidence/README.md` +2, `evidence/benchmark-smoke-2026-09-21.json` +175,
  `evidence/final-regression-2026-09-21.txt` +6501). The review package
  `review-20e148e..d389786.diff` matches the actual `git diff --stat` for the range.
- Working tree during verification: clean before I wrote this report
  (`git status --porcelain` → 0 lines). The transcript's run-start header records HEAD
  `20e148ec7ef8b1d2ba9e4ce9813074b83cef36b` with only the then-untracked evidence file.

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence type |
|---|---|---|---|
| 1 | CPU suite green (313 passed / 38 deselected / exit 0) | **CONFIRMED** | re-executed |
| 2 | GPU suite green (38/38) + representative slice + transcript consistency | **CONFIRMED** | re-executed slice (5 tests) + transcript audit |
| 3 | Fine-tune smoke: attempt-1 driver bug disclosed, corrected, attempt-2 green (12 steps, reload, WAV-SANE); driver bug, not product | **CONFIRMED** | transcript + source read + on-disk artifacts |
| 4 | vLLM path re-run, both sides executed | **CONFIRMED** | transcript + on-disk rows JSONs re-parsed |
| 5 | Lint: `ruff check .` exit 0 (format informational) | **CONFIRMED** | re-executed |
| 6 | Build artifacts fresh, 0.2.0, no `qwen_tts` vendoring | **CONFIRMED** | re-inspected wheel bytes |
| 7 | Zero-patch audit (upstream clones pristine, pinned HEADs, no tracked vendoring) | **CONFIRMED** | re-executed |
| 8 | Tree clean at d389786; pushed to origin/main | **CONFIRMED** | re-executed + `ls-remote` |
| 9 | Concern 2 is metadata-only (pip 0.1.0 vs pyproject/wheel 0.2.0) | **CONFIRMED** | re-executed |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-17-brief.md` (full)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-17-report.md` (full)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-20e148e..d389786.diff` (header, files-changed, evidence/README.md hunk; 6727 lines total)
- `evidence/final-regression-2026-09-21.txt` (6,501 lines — structure map via section headers; full reads of: env header + suites 1–2 (lines 1–151), suite 4 both attempts incl. correction note (1414–1864), suite 5 vLLM (1866–1930), suite 6 lint head (1932–1995), suite 7 build + Step 17.2 zero-patch audit + suite 8 verdict block (6257–6501))
- `evidence/benchmark-smoke-2026-09-21.json` (full, 175 lines)
- `scripts/make_finetune_dataset.py` (path-handling lines 47–49, 139, 148–160 — root-cause check)
- Scratch artifacts (runtime evidence, listed in §7)

## 5. Exact commands executed (by me, this verification)

1. `.venv/bin/python -m pytest -m 'not gpu' -q`
2. `.venv/bin/python -m pytest -m gpu -q -k "(test_generate_custom_voice and test_single) or (test_voice_clone_06b and test_clone_with_ref_text) or (test_official_demo_parity and test_real_synthesis_through_upstream_callback) or (test_tokenizer_codec and test_encode_decode_roundtrip_sane_and_duration_bound)"`
3. Same as (2) with `--collect-only` (to enumerate the slice)
4. `.venv/bin/python -m pytest -m gpu --collect-only -q` (slice selection)
5. `.venv/bin/python -m ruff check .`
6. `.venv/bin/python -m ruff format --check .`
7. `.venv/bin/python -m pip show qwen3-tts-rocm | head -2`
8. `.venv/bin/python -m zipfile -l dist/qwen3_tts_rocm-0.2.0-py3-none-any.whl` (+ `grep -c 'qwen_tts/'`); extracted the wheel to /tmp and read `qwen3_tts_rocm-0.2.0.dist-info/METADATA`
9. `git status --porcelain`; `git rev-parse HEAD origin/main`; `git rev-list --count origin/main..HEAD`; `git log -1 d389786`; `git diff --stat 20e148e..d389786`
10. `git -C .upstream/Qwen3-TTS rev-parse HEAD` + `status --porcelain`; `git -C .upstream/Qwen3-TTS-fix rev-parse HEAD` + `status --porcelain`
11. `git ls-files | grep -i upstream` (18 entries) ; `git ls-files .upstream` (0)
12. `grep -n 'attn_implementation' .upstream/Qwen3-TTS/finetuning/sft_12hz.py` (line 51 = `flash_attention_2`, pristine)
13. `timeout 30 git ls-remote origin refs/heads/main`
14. Dot-count checks on transcript suite-1/suite-2 pytest lines (`sed`/`tr`/`awk`)
15. On-disk artifact checks + JSON re-parses (§7)

## 6. Exit codes (mine, verbatim)

- CPU pytest: `313 passed, 38 deselected, 2 warnings in 17.83s` — **exit 0**
- GPU slice pytest: `5 passed, 346 deselected, 5 warnings in 36.50s` — **exit 0**
- `ruff check .`: `All checks passed!` — **exit 0**
- `ruff format --check .`: `47 files would be reformatted, 46 files already formatted` — **exit 1** (informational, exactly as disclosed; not a CI gate)
- `git status --porcelain` → 0 lines; `git rev-list --count origin/main..HEAD` → 0
- `git ls-remote origin refs/heads/main` → `d389786edeb21c619d844254be3eb23268def36f` (network-confirmed push)
- Wheel `qwen_tts/` grep → 0 entries (grep exit 1 = no matches)

## 7. Runtime evidence inspected

- Transcript suite 1: `313 passed, 38 deselected, 2 warnings in 17.99s`, `exit_code=0`; the four progress lines contain exactly 313 dots (counted).
- Transcript suite 2: `38 passed, 313 deselected, 5 warnings in 312.85s (0:05:12)`, `exit_code=0`; the dots line contains exactly 38 dots (counted); wall window `20:27:47` → `20:33:03` (316 s) vs 312.85 s pytest time — plausible; 3 tests named in the warnings summary (`test_generate_custom_voice.py::test_single`, `test_official_demo_parity.py::test_upstream_blocks_constructed`, and the same custom-voice test again for SDPA warnings).
- Transcript suite 4, attempt 1 (kept verbatim): Phase B `soundfile.LibsndfileError: Error opening '.work-finetune/regression/utt_zh01.wav'`, `prepare-exit=1`; Phases C/D cascade failures (`FileNotFoundError` train jsonl; loader `KeyError: unknown model reference`) — exactly as the correction note describes.
- Transcript suite 4, attempt 2: `make-dataset-exit=0` (12 utt + 1 ref renders), `prepare-exit=0`, `SCHEMA-OK: 12 lines, all with non-empty audio_codes`, `sft-exit=0` (Epoch 0 Step 0 Loss 13.0642 / Epoch 1 Step 0 Loss 9.4369, wall 18.5 s, peak 18.013 GiB reserved-pair consistent), 2 checkpoints (model.safetensors 3,833,402,520 B = 3.83 GB decimal each), `reload-exit=0` `WAV-SANE=True finite=True sr=24000 duration_s=3.28 peak=0.2676 rms=0.05003`, `RELOAD-AND-SYNTHESIS-OK`; clone restored (`porcelain 0`, HEAD `022e286b…`, `sft_12hz.py:51` back to `flash_attention_2`).
- Transcript suite 5: sha256 pairs identical for `ab_qwen.py`/`ab_vllm.py` originals vs regression copies; side A `ab-qwen-exit=0` (p1 rtf 1.2199 ok, p5 rtf 1.2368 ok, rows 2/2); side B `ab-vllm-exit=0` (p1 rtf 1.2466 ok, p5 rtf 1.5316 ok, rows 2/2); vLLM venv env line (vllm 0.28.0, HF_HOME offline snapshot, load 58.67 s).
- On-disk scratch artifacts (today's mtimes, re-listed/re-parsed by me): `.work-finetune/regression/` (12 `utt_*.wav` + `ref_speaker.wav`, `train_raw.jsonl` 12 lines with ABSOLUTE audio paths, `train_with_codes.jsonl` 12 lines all `audio_codes` non-empty — re-parsed, `runs/smoke/checkpoint-epoch-{0,1}/`, `reload_synth_en01.wav`, 4 raw logs); `.work-vllm/regression/` (`ab_qwen_rows_R.json` re-parsed → p1/p5 ok=True; `ab_vllm_rows_R.json` re-parsed → p1/p5 ok=True; both raw logs present).
- `evidence/benchmark-smoke-2026-09-21.json`: 4 cells (cn/en × short/medium), RTFs 1.2894–1.4738 (report's "1.29–1.47" is faithful), load 4.7 s, peak 4.927 GiB, meta git_head `20e148e…`, upstream SHA `022e286b…` — internally consistent with the report and transcript.
- `dist/`: `qwen3_tts_rocm-0.2.0-py3-none-any.whl` (208,284 B) + `qwen3_tts_rocm-0.2.0.tar.gz` (298,638 B), both mtime `2026-09-21 20:43` — fresh (today), matching the transcript's build phase; the Aug-29 `0.1.0` pair also present (pre-existing).

## 8. Regression tests (my re-runs)

- Full CPU suite (criterion 1): 313 passed / 38 deselected / exit 0 / 17.83 s — byte-for-byte the same summary as the implementer's (17.99 s).
- GPU representative slice (criterion 2), 5 of 38 (the `-k` substring `test_generate_custom_voice` also matched the 0.6B variant — a bonus):
  - `tests/test_generate_custom_voice.py::test_single` (CustomVoice 1.7B)
  - `tests/test_generate_custom_voice_06b.py::test_single` (CustomVoice 0.6B)
  - `tests/test_voice_clone_06b.py::test_clone_with_ref_text` (voice-clone, 0.6B)
  - `tests/test_official_demo_parity.py::test_real_synthesis_through_upstream_callback` (official parity + real synthesis through the upstream callback)
  - `tests/test_tokenizer_codec.py::test_encode_decode_roundtrip_sane_and_duration_bound` (tokenizer codec roundtrip)

  Result: `5 passed, 346 deselected, 5 warnings in 36.50s`, exit 0. Spans the required model families (custom-voice 1.7B+0.6B, voice-clone, parity, tokenizer).
- Lint (criterion 5): `ruff check .` exit 0; `ruff format --check .` exit 1 with exactly "47 files would be reformatted" — precisely matching the implementer's informational note.

## 9. Claims audit (implementer report vs my evidence)

- "CPU 313/38 exit 0" — reproduced exactly (17.83 s vs 17.99 s).
- "GPU 38/38 in 312.85 s" — not re-run in full (per verification scope); summary internally consistent (38 dots counted; deselect counts reciprocal 38/313; wall window matches; exit 0; 5 warnings) and 5-test slice green on the same GPU. The `-q` invocation emits dots, not per-test PASSED lines — consistent with the command as recorded; the warnings summary names real tests from the suite.
- "Benchmark smoke 4 cells RTF 1.29–1.47, load 4.7, peak 4.93 GiB" — JSON matches to the digit (1.2894–1.4738, 4.7, 4.927).
- "Fine-tune GREEN attempt 2; attempt 1 driver bug (relative `--output-dir`), method unchanged" — the transcript keeps attempt 1 verbatim with a marked correction note; the root cause is confirmed in source: `scripts/make_finetune_dataset.py` default is `_REPO_ROOT / ".work-finetune" / "data"` (absolute), `out_dir = args.output_dir` used verbatim for jsonl paths; attempt-2 `train_raw.jsonl` on disk carries absolute paths; `prepare_data.py` runs with cwd `.upstream/Qwen3-TTS/finetuning`, which is why the relative attempt-1 paths were unresolvable. This is an invocation bug of the task's scratch driver — the product script's behavior matches its documented default; NOT a product regression. "12 optimizer steps" = launcher arithmetic 12 samples / batch 2 × 2 epochs, corroborated by 2 per-epoch checkpoints and the official script's per-epoch Step-0 loss lines.
- "vLLM both sides exit 0, 2/2 ok rows each, drivers sha256-verified" — transcript plus on-disk rows JSONs re-parsed by me (qwen p1/p5 ok, vllm p1/p5 ok); raw logs present (65,869 B vLLM log consistent with a real 58.7 s load).
- "wheel contains ONLY qwen3_tts_rocm + dist-info; qwen_tts/ vendored = 0; METADATA 0.2.0, Requires-Dist qwen-tts==0.1.1" — reproduced independently from the wheel bytes (zipfile listing + extraction + METADATA grep). sdist listing in the transcript shows only package sources + tests.
- "Upstream clones pristine @ 022e286b / 48b8644, porcelain 0" — reproduced now (post-run), including `sft_12hz.py:51` restored; so the transient sdpa override really was reverted.
- "Pushed 89d0015..d389786 first attempt; HEAD == origin/main" — `git ls-remote` over the network returns `d389786…` for `refs/heads/main`; local `origin/main` ref identical; 0 unpushed commits.
- Concern 2 (`.venv` editable metadata 0.1.0): `pip show` reads `0.1.0` while pyproject says `version = "0.2.0"` and the built wheel METADATA says `0.2.0` — metadata-only staleness of the editable dist-info; code under test is live via the editable path (all suites run against the repo tree). Release-housekeeping, not a regression.

## 10. Problems found

None blocking. Non-blocking observations, for the record:

1. GPU full-suite evidence has dots + warnings-summary only (the `-q` format), not verbose per-test PASSED lines — a mild evidence-quality limitation inherent to the chosen verbosity, mitigated by the exact dot/summary counts and my 5-test slice re-run.
2. `git ls-files | grep -i upstream` returns 18 entries that include, besides docs/evidence, three repo tooling paths (`.github/workflows/upstream-drift.yml`, `scripts/check_upstream_drift.py`, `tests/test_check_upstream_drift.py`). These are the repo's own drift-check tooling, disclosed verbatim in the report; no vendored upstream source. Criterion intent (no vendoring) holds.
3. Stale `dist/` 0.1.0 artifacts (Aug 29) sit alongside the fresh 0.2.0 pair — cosmetic; CI convention evidently does not clean dist/.
4. `pip show` 0.1.0 editable dist-info (concern 2) — confirmed metadata-only; a `pip install -e .` refresh would true it up (implementer already noted this).
5. `ruff format --check` fails on 47 files — disclosed and informational; the repo's CI gate is `ruff check .` only (verified by re-run).

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence I generated or independently re-derived, not
by trusting the implementer: the CPU suite was re-run to the identical 313/38/exit-0 summary;
a 5-test GPU slice spanning custom-voice (1.7B and 0.6B), voice-clone, official parity, and
tokenizer passed on the same GPU with exit 0, and the full-GPU-suite transcript section is
arithmetically and temporally self-consistent (38 dots, 312.85 s inside a 316 s wall window);
lint was re-run (exit 0; the format failure was disclosed and is not a gate); the wheel was
re-inspected from its bytes (only `qwen3_tts_rocm/**` + dist-info, zero `qwen_tts/` entries,
METADATA 0.2.0); both upstream clones are pristine at the pinned HEADs with zero tracked
files under `.upstream/`; the tree is clean at `d389786`, which the network confirms is
`origin/main`'s tip; the fine-tune smoke's attempt-1 failure story was validated against the
actual source of `scripts/make_finetune_dataset.py` (absolute default, verbatim path storage)
and against attempt-2 on-disk artifacts (absolute-path jsonl, 12 coded lines, 2 checkpoints,
sane reload WAV); the vLLM A/B rows JSONs were re-parsed with ok=True on both sides; and the
0.1.0/0.2.0 discrepancy is demonstrably metadata-only (pyproject and wheel both 0.2.0). I
actively looked for fabrication vectors — mismatched counts, post-hoc transcript edits,
unrestored upstream patches, unpushed commits, vendored code in the wheel, a product-code
failure disguised as a driver bug — and every check came back consistent. No criterion
depends on the implementer's word alone.
