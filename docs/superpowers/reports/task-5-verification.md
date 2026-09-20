## VERIFICATION REPORT — TASK 5
### Verdict
PASS

Independent final release verifier, 2026-09-20/21. Repo HEAD during audit:
`91dfb36` (= Task 4 commit); all Task 5 changes staged, nothing committed,
nothing unstaged, nothing untracked (re-confirmed after the audit — my
verification left the tree byte-identical; all my artifacts went to /tmp).
The provided review diff `review-task5-91dfb36-staged.diff` was byte-compared
against a fresh `git diff --cached -U10`: identical (1,586 diff lines).

### Acceptance criteria
* [x] **1. North Star sentence verbatim near the top of README + CN mirror.**
  `README.md:11` (block-quote directly under the tagline, single line):
  `Official Qwen3-TTS on AMD Radeon — capability by capability, benchmark by benchmark, with zero upstream patches.`
  — exact-string grep hit. CN mirror single-line at `README_CN.md:7`
  (`AMD Radeon 上的官方 Qwen3-TTS —— 能力逐项验证、基准逐项实测、零上游补丁。`),
  same position, same meaning; both also grep-able in `docs/p0-parity-report.md` (lines 8–9).
* [x] **2. Capability matrix accurate; 4 states; ≥5 green cells spot-checked.**
  Consolidated matrix (13 rows) inserted as first subsection of "Verified, not
  promised" in both READMEs with the four-state legend (✅ E2E · 🟡 load-only ·
  ⬜ not validated · 🚫 not exposed/not claimed). Seven green cells checked
  against their artifacts (details below): 0.6B CustomVoice & Base, reusable
  clone prompt, Design→Clone→Reuse, multilingual, fine-tuning, 12Hz tokenizer,
  plus the 1.7B trio RTFs recomputed from `benchmark.json`. vLLM-Omni and
  true-streaming rows are 🚫 linking the roadmap, never green; fine-tuning row
  is ✅ **scoped — execution-only smoke … no quality claims** exactly as
  required. All 14 linked evidence files exist. No percentage scores (only
  URL-encoded badge shields match `%`).
* [x] **3. EN/CN consistent.** Both matrices have identical 15 table lines /
  13 data rows and identical state-emoji counts (✅ 39/39, 🟡 1/1, ⬜ 1/1,
  🚫 5/5). Counts identical everywhere current-state: 252 CPU + 38 GPU =
  290/290 (EN lines 42/168-171/293-294; CN 35/149-151/262-263), CI wording
  "same 252 CPU tests with 1 HIP-gated skip" in both, reproduce blocks both
  252/38. Roadmap sections mirrored (`## Roadmap (not yet validated)` /
  `## 路线图（尚未验证）`, anchors verified) incl. the "97 ms is NOT a Radeon
  number" note. I independently collected the CI expression: 252/290 — and
  `test_defaults_sdpa_on_hip` (tests/test_loader.py:66, `_NEEDS_HIP` skipif)
  is the single HIP-gated test, so CI = 252 with 1 skip is arithmetically right.
* [x] **4. No CustomVoice/Base conceptual mistakes remain.** Re-grepped
  `custom.?voice × clone|克隆` and `reference.?voice|参考音频克隆` over
  README/README_CN/docs//scripts/download_models.sh/src/qwen3_tts_rocm/models.py.
  Living-doc hits are all correct usage (multilingual-table "Clone
  (cross-lingual)" column = Base cross-lingual clone; API/file names;
  screenshots) or the new explicit boundary paragraph ("CustomVoice … does
  not clone from reference audio. Base is the zero-shot voice-cloning and
  fine-tuning family"). `models.py` has zero clone prose. Remaining hits live
  only in archival dated records (docs/development/2026-08-27-*, verbatim
  verifier-report copies, the program's own plan/spec) — none misframes
  CustomVoice as a living claim. `download_models.sh` help now reads
  "custom-voice … preset/custom-speaker voice generation", "base … zero-shot
  voice cloning + fine-tuning base".
* [x] **5. No multi-GPU or all-Radeon generalization.** Grep
  `all radeons|all amd|any radeon|every radeon|所有 Radeon|所有 AMD|multi-gpu|多卡|多 GPU`
  over READMEs/docs/CHANGELOG/docker/scripts/src (excluding archival records):
  zero hits. Multilingual ✅ captions explicitly scoped to "the validated
  Radeon 8060S (`gfx1151`) host" in both languages. Community template
  `.github/ISSUE_TEMPLATE/hardware-validation.yml` untouched (0 staged changes)
  and still demands measured results with the confirmation checkbox.
* [x] **6. "Zero upstream patches" literally true.** (a) Staged diff touches
  exactly 14 repo paths — no `.upstream/`, no site-packages, no vendored
  source. (b) `.upstream/Qwen3-TTS`: `rev-parse HEAD` =
  `022e286b98fbec7e1e916cb940cdf532cd9f488e` (exact pin), `status --porcelain`
  empty, `diff --stat HEAD` = 0 lines. (c) pip RECORD hash audit of installed
  `qwen_tts 0.1.1` (my own base64/sha256 re-verification): **26 files checked,
  0 mismatch, 0 missing** (18 no-hash rows = `__pycache__`/RECORD itself).
  Ground truth spot-check: installed `qwen3_tts_model.py` still nulls
  `instruct` for 0.6B — the 🚫 matrix row is accurate.
* [x] **7. Rename behavior-preserving and complete.** Staged diff records
  `R src/qwen3_tts_rocm/patch.py -> src/qwen3_tts_rocm/compat.py` (76%
  similarity). AST comparison of `git show HEAD:…patch.py` vs the new
  `compat.py` after stripping the module docstring: **identical** — only the
  docstring changed (now truthfully describing a version advisory + empty
  reserved patch point). No shim (correct: grep shows zero importable
  references to `qwen3_tts_rocm.patch` / `from .patch` / `import patch`
  anywhere in src/tests/scripts; the only `patch` word-matches in code are
  pytest's `monkeypatch` fixture and docstrings). `loader.py` import + call
  site updated; `tests/test_loader.py` fully migrated (`test_compat_*`,
  `loader.compat` monkeypatch target). No pyproject/__init__/Docker
  references existed. Full suites green post-rename (below).
* [x] **8. Parked findings fixed.** P1: staged `evidence/README.md` row for
  `multilingual-matrix.txt` rewritten — verified `RUN_EXIT` does NOT occur in
  the transcript (grep exit 1) and the file really ends
  `[summary] RESULT: PASS` / `[evidence] wrote …`, exactly as the new row
  states. P2: "Errata — two labels inside finetune-smoke-2026-09-20.txt"
  section added; I confirmed transcript **line 108** is
  `porcelain-exit=0 (empty output above = pristine)` directly under a
  non-empty 37-line staged listing and **line 1546** is `as-is-run-exit=0`
  directly under the flash-attn ImportError — the transcript itself is not in
  the staged diff (byte-for-byte unedited).
* [x] **9. Verbatim report copies.** `diff -q` of
  `docs/superpowers/reports/task-{1,2,3,4}-verification.md` against
  `.superpowers/sdd/2026-09-20-p0-capability-parity/task-N-verification.md`:
  all four byte-identical (md5 08a550d3…, 9eb8bbe6…, 090cd9e4…, 9f254978…).
* [x] **10. p0-parity-report structure and traceability.**
  `docs/p0-parity-report.md` (209 lines) follows the mandated structure
  exactly: North Star / Repository SHAs / Completed (Tasks 0–5) / Capability
  matrix / Benchmarks / Newly proven Radeon value / Remaining gaps / Zero-patch
  audit / Test summary / Changed files / Commits. Verified claims: SHAs
  f913d4c, fb8bc60, 2a90229, d5a74b0, 91dfb36 all match `git log`; upstream
  pin live-verified; host stack live-verified (torch 2.12.0+rocm7.14.0, HIP
  7.14.60850, qwen-tts 0.1.1, Python 3.12.3); all four PASS links resolve;
  changed-files lists match `git show --name-status` for every task commit;
  benchmark table recomputed from the JSONs (see Runtime evidence); Task 5 row
  reads "**PENDING — final verification gate (verdict recorded in the Task 5
  commit)**" — clearly marked as awaiting this gate, as designed.
* [x] **11. All prior tests green — re-run by me.** CPU 252/252, GPU 38/38,
  demo parity 3/3, ruff clean, CI-expression collection 252 — all exit 0
  (exact commands below).

### Code reviewed
- `review-task5-91dfb36-staged.diff` in full (1,620 lines incl. headers) and
  re-derived from the index — identical.
- `README.md`, `README_CN.md` (North Star, matrix, boundary paragraph, counts,
  roadmap, reproduce blocks — both languages side by side).
- `docs/p0-parity-report.md` (all sections, every SHA/link/number).
- `docs/superpowers/reports/task-{1..4}-verification.md` (copy fidelity).
- `evidence/README.md` (P1 row rewrite + P2 errata section, staged version via
  `git show :evidence/README.md`).
- `src/qwen3_tts_rocm/compat.py` vs `git show HEAD:src/qwen3_tts_rocm/patch.py`
  (AST equality), `src/qwen3_tts_rocm/loader.py` hunks, `tests/test_loader.py`
  hunks, `scripts/download_models.sh` help, `scripts/make_finetune_dataset.py`
  lint hunks (EXE001 chmod + ISC004 parens — content-neutral),
  `CHANGELOG.md` Task 5 bullet.
- Evidence artifacts: `benchmark.json`, `benchmark-06b-2026-09-20.json`,
  `multilingual-matrix.json`, `voice-workflow-2026-09-20.json`,
  `finetune-smoke-2026-09-20.json` (+ `.txt` spot lines 71–108, 1543–1548),
  `gpu-suite-2026-09-20.txt`, `gen-customvoice.txt`, `gen-voicedesign.txt`,
  `gen-voiceclone.txt`, `tokenizer-codec.txt`, `multilingual-matrix.txt` tail.
- Installed upstream: `.venv/.../qwen_tts/inference/qwen3_tts_model.py`
  (0.6B instruct null) + RECORD; `.upstream/Qwen3-TTS` git state;
  `.github/ISSUE_TEMPLATE/hardware-validation.yml`.

### Tests executed
Exact commands (cwd `/home/amd/Desktop/Qwen3-TTS-ROCm`) and exit codes:
1. `.venv/bin/python -m pytest -m 'not gpu' -q` → **252 passed, 38 deselected in 15.91s** — **exit 0**
2. `.venv/bin/python -m pytest -m gpu -q` → **38 passed, 252 deselected in 312.04s** (real gfx1151) — **exit 0**
3. `.venv/bin/python -m pytest tests/test_official_demo_parity.py -q` → **3 passed in 12.87s** (incl. real synthesis through the upstream callback) — **exit 0**
4. `.venv/bin/ruff check` → **All checks passed!** — **exit 0**
5. `.venv/bin/python -m pytest -m "not gpu and not requires_download" --collect-only -q` → **252/290 tests collected (38 deselected)** — **exit 0** (proves CI runs the same 252; the 1 HIP-gated member is `test_defaults_sdpa_on_hip`)
6. `git -C .upstream/Qwen3-TTS rev-parse HEAD` → `022e286b98fbec7e1e916cb940cdf532cd9f488e` — **exit 0**; `git -C .upstream/Qwen3-TTS status --porcelain` → **empty** — **exit 0**; `git -C .upstream/Qwen3-TTS diff --stat HEAD | wc -l` → 0
7. pip RECORD hash audit of installed `qwen_tts 0.1.1` (python script, sha256+base64 vs RECORD) → **checked=26 mismatch=0 missing=0, RECORD-INTACT** — **exit 0**
8. `grep -rn "patch" src/ tests/ --include='*.py'` (and the import-pattern greps `qwen3_tts_rocm.patch|from .patch|import patch`) → **zero importable references to the old module**; all word-matches are pytest's `monkeypatch` fixture or docstrings — benign.

### Runtime evidence
**Capability-matrix spot-checks (7 green cells + boundary rows):**
1. **CustomVoice 0.6B** → `gpu-suite-2026-09-20.txt`: header "full on-GPU suite after Task 1 (0.6B E2E parity)", cmd `-m gpu -q`, **34 passed** (all dots, no failures) on Radeon 8060S; per-test proof of the 9 new 0.6B tests is in the verbatim-copied Task 1 verifier report (re-ran them: green within my 38/38).
2. **Base family 0.6B** → same transcript + `benchmark-06b-2026-09-20.json`: per_alias `base-0.6b` load 1.547 s / peak 3.104 GiB; recomputed median RTF range **1.187–1.255** ↔ quoted "1.19 – 1.26" ✓.
3. **CustomVoice 0.6B RTF** → recomputed **1.029–1.300** ↔ quoted "1.03 – 1.30", load 4.041 s / peak 2.802 GiB ↔ "4.0 s / 2.80 GiB" ✓.
4. **Multilingual** → `multilingual-matrix.json`: 10 CustomVoice + 10 VoiceDesign + 4 cross-lingual clone rows, **all `pass: true`**; summary 24/24 cells + 4/4 references, `all_passed: true`; language_ids = exactly the 10 official languages; meta `gfx1151`, `max_new_tokens: 512`, upstream SHA 022e286… ✓.
5. **Design→Clone→Reuse (Voice Studio)** → `voice-workflow-2026-09-20.json`: design 5.469 / prompt 0.307 / reuse 5.283, 6.417 s ↔ README "≈5.5 / ≈0.3 / 5.3–6.4" and p0-report ✓; meta carries gfx1151 + env header.
6. **Fine-tuning** → `finetune-smoke-2026-09-20.json`: `goal = EXECUTION VALIDATION … No quality claims anywhere`; 12 optimizer steps (with the upstream per-microbatch quirk documented), prep exit 0, official-path reload, sane synthesis (24 kHz / 3.52 s / rms 0.03778), scope_guard.not_proven = 6 items — matches the matrix's execution-only wording exactly.
7. **12Hz tokenizer** → `tokenizer-codec.txt`: encode/decode roundtrip suite, **2 passed, EXIT=0** on Radeon 8060S (2026-08-27).
8. **1.7B trio** → `gen-customvoice.txt` (4 passed), `gen-voicedesign.txt` (4 passed), `gen-voiceclone.txt` (5 passed, EXIT=0) on the Radeon; `benchmark.json` (2026-08-27) median-of-medians recomputed: custom-voice **1.307–1.513** ↔ "1.31 – 1.51", voice-design **1.275–1.617** ↔ "1.27 – 1.62", base **1.713–1.878** ↔ "1.71 – 1.88" ✓.
9. **Boundary rows**: 0.6B instruct 🚫 confirmed against installed wrapper source (`if self.model.tts_model_size in "0b6": instruct = None`); vLLM-Omni and true streaming 🚫 → roadmap links, never green.

**Live environment:** torch 2.12.0+rocm7.14.0, HIP 7.14.60850, qwen-tts 0.1.1,
Python 3.12.3 — matches every documented claim.

### Claims audit
- North Star: verbatim EN + faithful CN mirror, top-of-README in both, and in the p0 report. ✓
- Counts: every current-state count in both READMEs + p0 report is 252 CPU + 38 GPU = 290/290, CI same-252-with-1-HIP-skip; independently reproduced (252 collected for the CI expression; 252+38 run green). No stale 215/216/243/244/34 current-state claims anywhere in living docs (swept READMEs, docs/, docker/, CONTRIBUTING). CHANGELOG keeps as-of-then counts (250 → 282 → 290; CI 215/243) in dated per-task bullets — historically accurate progression, disclosed by the implementer, and the new Task 5 bullet states the final numbers.
- Benchmarks: every range/figure in READMEs and the p0 report recomputes from the archived JSONs (see above); n=2 ranges with iGPU-variance caveats, no overstated precision, no cross-session A/B implied (different days explicitly noted).
- Voice-workflow timings: JSON ↔ READMEs ↔ p0 report ↔ CHANGELOG all agree.
- Zero-patch: three independent proofs re-run live (diff paths, upstream clone, pip RECORD) — all clean; the Task 4 sdpa deviation is documented as clone-local + reverted + errata'd.
- p0 report changed-files lists match `git show --name-status` for all five task commits; commits list matches `git log`; PASS links resolve; Task 5 row PENDING-by-design, clearly marked.
- No pronunciation/quality claims, no percentage scores, no all-Radeon/multi-GPU claims (greps clean in both languages).

### Regression audit
- Full CPU suite 252/252 and full GPU suite 38/38 re-run by me on the real gfx1151 — the exact numbers every document now claims.
- The rename is provably behavior-preserving (AST-identical modulo docstring) and import-complete (zero dangling references; `pyproject`/`__init__`/Docker never referenced the module).
- `tests/test_official_demo_parity.py` 3/3 — the upstream-unchanged proof still holds, backed by the RECORD audit.
- `ruff check` clean, including the two disclosed Task-4-leftover lint fixes in `scripts/make_finetune_dataset.py` (mode 100644→100755 + parenthesized concatenation; `tests/test_finetune_dataset.py` 8/8 green inside the 252).
- Working tree after my audit: identical 14 staged paths, 0 unstaged, 0 untracked — the gate modified nothing.

### Problems found
1. (Observation, no action) The four-state legend defines 🟡 partial-load-only and ⬜ not validated, but no row in the consolidated matrix currently uses them (all rows are ✅ or 🚫). Nothing is mislabeled — the load-level distinction lives in the "6 / 6 load-validated" hero row and the per-checkpoint detail view — so this is vocabulary breadth, not an error.
2. (Observation, no action) The 0.6B rows' transcript evidence (`gpu-suite-2026-09-20.txt`) is a `-q` whole-suite run (34 passed) that does not name individual tests; the per-test 0.6B proof lives in the Task 1 verifier report, which is now verbatim-copied into `docs/superpowers/reports/` and linked from the p0 report. The chain is complete inside the repo.
3. (Observation, no action) The p0 report's zero-patch audit quotes "25/25 and 26/26 source files" — faithfully reporting what the Task 1 vs Task 2/4 verifiers counted; my own recount finds 26 hashed RECORD rows (Task 1's script evidently counted 25). 0 modified either way; immaterial.
4. (Observation, no action) Historical CHANGELOG bullets retain then-current counts (250/282, CI 215/243) as a dated progression while all current-state claims are uniform 252/38/290 — internally consistent and disclosed; a reader comparing old bullets to today's numbers must notice the dates.
5. (Cosmetic) Rounding in the p0 report's benchmark table (4.041→"4.0 s", 3.104→"3.10 GiB") is correct to the stated precision.
6. (Process note) The 0.6B `instruct` 🚫 row's evidence link is the Task 0 audit doc rather than a test transcript; the behavior is additionally pinned by `test_instruct_behavior` inside the green GPU suite — acceptable and consistent with the 🚫 state semantics.

No discrepancy between the implementer's report and the artifacts was found:
every number, link, SHA and count in the Task 5 report was independently
reproduced or traced to a staged artifact.

### Final justification
Every one of the eleven acceptance criteria was verified against artifacts I
inspected and commands I executed myself, not against the implementer's
claims: the North Star sentence sits verbatim at the top of both READMEs with
a faithful CN mirror; the consolidated capability matrix uses the four states
consistently, never marks vLLM-Omni or true streaming green, scopes
fine-tuning to execution-only, links only existing evidence, and contains no
percentage scores — with seven-plus green cells validated against their
underlying JSON/transcript artifacts, including full recomputation of every
quoted benchmark range; EN and CN are structurally and numerically identical
(counts re-derived live: 252 CPU + 38 GPU = 290/290, CI same-252 + 1
HIP-gated skip); the CustomVoice/Base conceptual boundary is correct
everywhere in living docs (and the installed wrapper source confirms the 🚫
instruct row); nothing generalizes gfx1151 to other Radeons or claims
multi-GPU; the zero-upstream-patch guarantee holds under three independent
live proofs (staged paths, pristine pinned clone, 26/26 pip RECORD hashes);
the patch.py→compat.py rename is AST-identical modulo its docstring with zero
dangling references and both full suites green after it; both parked findings
are fixed with the transcript left byte-unedited; the four verifier reports
are byte-identical copies; the p0 report follows the mandated structure with
every claim traced and its Task 5 row clearly PENDING for this gate; and I
re-ran the entire verification surface myself — 252 CPU, 38 GPU, 3 demo
parity, ruff clean, all exit 0. The six findings above are observations or
cosmetics that falsify no criterion. This gate protects a push that is
documented exactly as strongly as its evidence. **PASS.**
