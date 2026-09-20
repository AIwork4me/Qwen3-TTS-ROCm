# VERIFICATION REPORT — TASK 3

*Independent release verifier, 2026-09-20. Repo HEAD during audit:
`2a902291c4a60ca1ca9d64ef9bc39c9a860259e5` (= the meta recorded in the
evidence JSON). Working tree == staged diff (nothing unstaged, nothing
untracked; no `voices/` dir in the repo). Verifier modified nothing.*

## Verdict

**PASS**

## Acceptance criteria

* [x] **1. Real one-click end-to-end workflow in the UI.** The ⑥ Voice
  Studio tab (`src/qwen3_tts_rocm/demo/ui.py` ~L823-919) wires
  design-click → preview Audio + `gr.State` voice-id → save-click (outputs
  `[status, saved-voices dropdown]` via `gr.update(choices=...)`) →
  generate-click from the dropdown: no `gr.File` anywhere in the flow, no
  retype (one `text` variable feeds generation and `ref_text` by
  construction in `design_voice`). Structurally pinned by
  `test_build_ui_voice_studio_adds_no_file_components` (exactly the 4
  pre-existing `gr.File`s remain) — passed in my run.
* [x] **2. Official APIs only.** Grep of `src/qwen3_tts_rocm/voice_workflow.py`
  for generation-ish tokens yields exactly three model calls:
  `model.generate_voice_design(...)` (L157),
  `prompt_model.create_voice_clone_prompt(...)` (L171),
  `model.generate_voice_clone(...)` (L206). No transformers import, no
  `.forward`, no `AutoModel`/`from_pretrained`, no model `__call__`. The
  only private-name import is `from .loader import
  _suppress_fd_stdout_stderr` — our own package's fd-noise suppressor
  around `import qwen_tts` (same pattern `backend.py` L287 already uses);
  it cannot reach or silence the official `tts_model_type` gate, which
  lives inside the official method bodies.
* [x] **3. Official-compatible reusable voice data.** Installed official
  `qwen_tts/cli/demo.py::save_prompt` (L516-521) writes
  `{"items": [asdict(it) ...]}` via `torch.save`;
  `::load_prompt_and_gen` (L534-538) reads **only** `payload["items"]` and
  never touches any other key — the additive `voice_meta` is provably
  ignored, and it roundtrips through `weights_only=True` (dict/str/bool
  allowlisted; GPU roundtrip test passed). The walk's actual artifact
  `/tmp/t3_evidence_voices_4bgihjcq/evidence_bright_female.pt` (mtime
  22:41, matching the JSON timestamp) loads under
  `torch.load(weights_only=True)` with item keys exactly
  `{ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text}`
  (== the official dataclass, verified in
  `qwen_tts/inference/qwen3_tts_model.py` L40-51), tensors verbatim
  (`ref_code` int64 (36,16), `ref_spk_embedding` bf16 (2048,)) — i.e. the
  `items` content is exactly what `asdict` of an official item produces;
  the ndarray→list conversion in `save_voice` fires only for fake
  (ndarray-carrying) items, never for real ones.
* [x] **4. Real GPU integration test passing.** Re-ran myself:
  `4 passed in 28.73s` on the real gfx1151 (exit 0), covering
  design→sane preview+prompt (transcript rule + ICL invariants), reuse
  across two sentences (both sane, `_distinct`), save/load roundtrip then
  reuse (device-agnostic field parity + sane regeneration), official
  schema preserved.
* [x] **5. Reusable prompt generates multiple new sentences.** Walk
  transcript: two reuse renders from the saved voice
  (`generate_s=5.3s/6.4s`, wav durations 3.52s/4.24s — distinct),
  `[walk] WALK-OK (two reuse renders from the saved voice)`; JSON
  `reuse_s_1=5.283`, `reuse_s_2=6.417`; GPU test additionally asserts
  `_distinct(wav1, wav2)`.
* [x] **6. Evidence archived with three separate phase timings.**
  `evidence/voice-workflow-2026-09-20.json` parses with all four keys
  (design_s=5.469, prompt_s=0.307, reuse_s_1=5.283, reuse_s_2=6.417) plus
  meta (env header: Radeon 8060S/gfx1151, torch 2.12.0+rocm7.14.0, hip
  7.14.60850, qwen-tts 0.1.1; git_head = actual HEAD; upstream SHA
  022e286b…; max_new_tokens=512). The .txt has phase 1 (pytest tee,
  `4 passed ... 29.53s`, `PYTEST_EXIT=0`), a marked separator, and phase
  2 behind it ending `WALK_EXIT=0`; the `[walk] RESULT: design_s=5.47
  prompt_s=0.31 reuse_s_1=5.28 reuse_s_2=6.42` line is consistent with
  the JSON (exact 2-dp roundings). Phases never collapsed.
* [x] **7. Standalone VoiceDesign and Voice Clone paths intact.**
  `tests/test_voice_design.py` + `tests/test_voice_clone_workflow.py` are
  byte-identical to HEAD (zero diff); re-ran myself: `9 passed` exit 0.
  Full `-m gpu` re-run: `38 passed` exit 0 (= the README's 38-GPU claim).

## Code reviewed

Full staged diff (13 files, +3183/−29) plus targeted reads of
`voice_workflow.py`, `demo/backend.py`, `demo/ui.py`, `testing.py`
(current and `git show HEAD:` versions), the installed official
`qwen_tts/cli/demo.py` and `qwen_tts/inference/qwen3_tts_model.py`,
`loader.py` (`_suppress_fd_stdout_stderr`, `unload`, size-1 LRU in
`get()`), and the evidence driver `/tmp/t3_evidence_driver.sh` + probe
log `/tmp/t3_probe_design_voice.log`.

**Orchestrator-ruling compliance (adjudicated deviation):** applied
exactly as ruled. `design_voice(model, *, text, language, description,
max_new_tokens=512, prompt_model=None, **gen_kwargs)` — `model` drives
`generate_voice_design`; `prompt_model` (object OR zero-arg lazy factory,
honored at L167-169) drives `create_voice_clone_prompt`; `prompt_model=None`
defaults to `model`, which propagates the official ValueError honestly
(no bypass, no silencing, no tampering with `tts_model_type`). All
brief-pinned call shapes remain valid (same keyword names, positional
model first, 512 default). The ruling's factual premise is genuine:
probe log shows the installed wrapper raising
`ValueError: ... tts_model_type: voice_design does not support
create_voice_clone_prompt` (qwen3_tts_model.py:402), and I confirmed the
three hard gates in the installed source (L401 base-required for
`create_voice_clone_prompt`, L548 base for `generate_voice_clone`, L686
voice_design for `generate_voice_design`).

**testing.py adversarial audit:** the hunk is exactly 3 change blocks —
docstring text, `ref_text: Any = None` (a field WITH default, so all
existing 4-arg/4-kwarg constructions still work), and `ref_text=txt` in
the fake's `create_voice_clone_prompt`. `assert_wav_sane` and every other
sanity gate are untouched (verified by full-file diff vs HEAD). The RNG
seed expression already included the transcript (`self._seed(..., repr(audio),
txt, xvec)`), so determinism is unchanged; the CPU suite count went
229→244 with zero pre-existing test outcomes altered (my run: 244 passed).

**UI hunks:** the ui.py hunks are docstrings, additive callback keys, the
new ⑥ tab block (inserted after ⑤ History's components), the new ⑥ event
wiring, and one new `demo.load` targeting only `vs_saved` (a second,
independent `.load` alongside the pre-existing status-line one — legal in
Gradio). No ①–⑤ component or wiring line was modified; the single changed
existing test updates an exact dropdown count 4→6 for the additive tab
(still asserts ALL language dropdowns keep `allow_custom_value`) — a
correction forced by addition, not a weakening.

## Tests executed

All in repo cwd, `.venv/bin/python`:

| Command | Result | Exit |
|---|---|---|
| `.venv/bin/python -m pytest -m 'not gpu' -q` | **244 passed, 38 deselected** in 15.59s | 0 |
| `.venv/bin/python -m pytest tests/test_voice_workflow.py -m gpu -v -s` | **4 passed** in 28.73s (timing lines reproduced: design 5.4s/0.5s, reuse 5.6s/4.3s/4.2s) | 0 |
| `.venv/bin/python -m pytest tests/test_voice_design.py tests/test_voice_clone_workflow.py -m gpu -q` | **9 passed** in 131.18s | 0 |
| `.venv/bin/python -m pytest tests/test_demo_backend.py tests/test_demo_ui.py -q` | **100 passed** in 14.24s | 0 |
| `.venv/bin/python -m pytest tests/test_official_demo_parity.py -q` | **3 passed** in 13.15s | 0 |
| `.venv/bin/python -m pytest -m gpu -q` (extra: verifies README 38-GPU claim) | **38 passed, 244 deselected** in 307.46s | 0 |
| JSON parse + field check of `evidence/voice-workflow-2026-09-20.json` | all 4 timing keys positive floats + full meta; git_head == `git rev-parse HEAD` | 0 |
| grep of `voice_workflow.py` for non-official generation paths | only the 3 official method calls (see criterion 2) | 0 |
| official schema check vs installed `qwen_tts/cli/demo.py` (L501-572) + `VoiceClonePromptItem` dataclass + walk `.pt` artifact | items-format claim confirmed | 0 |

## Runtime evidence

Phase 1 (pytest tee): `4 passed, 2 warnings in 29.53s`, `PYTEST_EXIT=0`,
with in-transcript `[timing]` lines for design (5.7s/0.3s), two-sentence
reuse (5.2s/4.9s) and roundtrip reuse (4.7s). Phase 2 (heredoc walk
behind the marked separator): env header comments identical to the JSON
meta; three generation markers (`Setting pad_token_id` ×3 at txt L652/
L668/L1205 = design + 2 reuses); two reuse timing prints (5.3s/6.4s,
wavs 3.52s/4.24s @ 24 kHz); `RESULT` line consistent with the JSON;
`WALK-OK`; `WALK_EXIT=0`. The driver script
(`/tmp/t3_evidence_driver.sh`) confirms the walk calls the REAL
`voice_workflow.design_voice(vd, prompt_model=bc, …, max_new_tokens=512)`
→ `assert_wav_sane` → `save_voice` → `loader.unload(vd)` → `load_voice`
→ `reuse_voice` ×2 (`TEXT`, then `ALT_TEXT`), writing the JSON itself —
fully machine-generated, phases timed separately.

## Claims audit

* README/README_CN "282/282 on validation host — 244 CPU + 38 real-GPU":
  verified by my runs (244 + 38, both exit 0).
* README "CPU-only CI … 243 CPU tests (244 on AMD hosts)": collect-only
  of `-m 'not gpu and not requires_download'` gives 244 collected on this
  AMD host; CI skips the 1 HIP-gated test → 243. Arithmetic consistent.
* README/CN latency claims "design ≈ 5.5 s · prompt ≈ 0.3 s · reuse ≈
  5.3–6.4 s": match the archived JSON (5.469/0.307/5.283/6.417).
  CHANGELOG "design 5.47 s, prompt 0.31 s, reuse 5.28/6.42": exact.
* "items key byte-format-identical to upstream's payload / the stock demo
  can load them too": verified against the installed demo source (items
  read exclusively; tensors pass through verbatim; the extra `voice_meta`
  is ignored by the official loader) and against the walk's real `.pt`.
* "Six-tab", "no download/re-upload hop", transcript-rule-by-construction:
  verified in code + pinned by tests.
* No quality overclaims: waveform sanity only; no pronunciation/quality
  language anywhere in the new sections.
* `evidence/README.md` index rows for both new artifacts present and
  accurate. Report's numbers all trace to the archived artifacts; the
  report's disclosed earlier run (6.11/0.51/5.40/4.83) is corroborated by
  the second temp voices dir in /tmp (mtime 22:34 vs final 22:41).

## Regression audit

* Standalone GPU suites untouched (zero diff vs HEAD) and green (9/9).
* Full GPU suite 38/38; full CPU suite 244/244 (was 229 before this task
  per the orchestrator's baseline; +15 new CPU demo/backend/UI tests).
* `assert_wav_sane` & friends untouched; FakeTTSModel change is
  additive-with-default; no existing assertion weakened (the one edited
  UI test changes an exact count 4→6, still exact).
* Demo parity suite (upstream-unchanged proof) still 3/3.
* Backend's size-1 model LRU semantics preserved: the studio's lazy
  `prompt_model=lambda: self.get(base_alias)` evicts VoiceDesign exactly
  at the prompt phase (pinned by
  `test_voice_studio_design_routes_off_mismatched_radio` /
  LRU-no-reload test). No `voices/` dir or stray artifacts in the repo.

## Problems found

1. **(Minor, evidence completeness — not a criterion failure)** Three
   walk prints are missing from the phase-2 transcript:
   `[timing] design_voice …`, `[walk] preview … sane …`, and
   `[walk] saved <path> (bytes)`. Root cause (verified in
   `src/qwen3_tts_rocm/loader.py` L146-155): `_suppress_fd_stdout_stderr`'s
   finally-block deliberately drains Python's buffered stdout into
   /dev/null before restoring the fds, and `load_voice`'s
   `with _suppress_fd_stdout_stderr(): from qwen_tts import …` consumed
   the buffer holding those three lines (printed between `design_voice`
   and `load_voice`). Everything printed after `load_voice` survived.
   Impact is cosmetic: the phase timings themselves are carried in
   `res.timings` → `RESULT` line → JSON, the driver script proves
   `design_voice` was really called, and the saved `.pt` exists in
   `/tmp/t3_evidence_voices_4bgihjcq/` with correct schema/content and a
   timestamp matching the JSON. No number in the JSON or docs depends on
   the lost lines. (Pre-existing loader behavior interacting with the new
   `load_voice`; fixing it is out of scope for a read-only audit — a
   `flush=True` or a stderr print in future evidence drivers would avoid
   it.)
2. **(Observation, adjudicated)** The brief's single-model
   `design_voice(vd_model, …)` sketch now raises the official ValueError
   when called without `prompt_model` on a VoiceDesign object — inherent
   to the orchestrator's accepted resolution (the official gate must
   propagate honestly); documented in the module docstring; all
   brief-pinned keyword shapes remain valid.
3. **(Observation)** `design_voice`/`reuse_voice` accept `**gen_kwargs`
   beyond the brief sketch, so a caller can still override
   `max_new_tokens` — this matches the repo's pre-existing, README-
   documented "override knowingly via the Advanced accordion" guardrail
   policy (`DEFAULT_GEN_KWARGS = {"max_new_tokens": 512}`), and every
   workflow/test/evidence generation pins 512 (CPU tests assert
   `gen_kwargs == {"max_new_tokens": 512}`).

## Final justification

Every acceptance criterion was independently re-verified on the real
gfx1151 GPU and against the installed official `qwen_tts` source, not the
implementer's claims: the one-click no-hop UI flow is structurally pinned
(zero new `gr.File`), the workflow module contains exactly the three
official method calls with the adjudicated `prompt_model` split and no
gate bypass, the persisted payload is the official `items` format with an
additive sidecar the official loader provably ignores (confirmed on a real
saved artifact), the new GPU suite (4/4), the standalone regressions
(9/9), the full CPU (244) and GPU (38) suites, and the demo/parity suites
all pass under my own execution with exit 0, and the archived three-phase
evidence is machine-generated, internally consistent, and traceable to a
real run at the recorded HEAD. The only defect found is a cosmetic
transcript gap (three buffered walk prints drained to /dev/null by
pre-existing loader fd-suppression) that does not affect any recorded
number, test, or documented claim. PASS.
