# P0 Capability Parity Program — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved 5-task gated program that turns Qwen3-TTS-ROCm from "1.7B generation validated" into "capability-by-capability Radeon validation with 0.6B E2E parity, a multilingual matrix, a first-class design→clone→reuse workflow, a fine-tuning smoke, and an evidence-aligned narrative."

**Architecture:** Subagent-driven gated execution in `/home/amd/Desktop/Qwen3-TTS-ROCm` (branch `main`). Per gated task: the orchestrator dispatches a fresh implementation subagent carrying that task's plan section verbatim; the implementer never commits; a fresh independent verification subagent re-runs tests and audits evidence; the orchestrator commits only after PASS. Task N+1 starts only after Task N's verifier returns PASS. All pushes happen once, at the very end.

**Tech Stack:** Python 3.12 venv (`.venv/`), `qwen-tts==0.1.1` (official package, never modified), `torch 2.12.0+rocm7.14.0`, pytest with `gpu`/`requires_download` markers, Gradio demo, AMD gfx1151 (Radeon 8060S, unified memory).

**Spec:** `docs/superpowers/specs/2026-09-20-p0-capability-parity-design.md` (commits `e6e0b80` + `5ebb88b`). The plan argues from the spec; executors read both.

## Global Constraints

- **Zero upstream patches.** Never modify `qwen_tts` package source or upstream `QwenLM/Qwen3-TTS`. Upstream is cloned only in Task 4 into gitignored `.upstream/` at a pinned SHA; nothing upstream is committed to this repo.
- **Official APIs only.** All generation flows through the official object returned by `qwen3_tts_rocm.loader.load(alias)` calling `generate_custom_voice` / `generate_voice_design` / `generate_voice_clone` / `create_voice_clone_prompt` — keyword-first calls everywhere (the installed wrapper's positional order differs from early upstream sketches).
- **Latency guardrail:** every clone/long generation passes official kwarg `max_new_tokens=512` (a 2048-token stochastic runaway measured ~23 min on this iGPU).
- **Loader defaults:** bf16 + SDPA HIP defaults via `loader.load`; flash-attn is absent from the validated stack — never request it.
- **Claim vocabulary:** ✅ Radeon E2E validated / 🟡 partial-load-only / ⬜ not validated / 🚫 not exposed upstream or intentionally unclaimed. "Supported" ≠ "validated". No pronunciation-quality claims from waveform sanity. No subjective 0.6B-vs-1.7B quality claims. Never reuse upstream's "97 ms" streaming figure as a Radeon number.
- **Evidence policy:** every GPU capability produces `evidence/<task>-<date>.{json,txt}` with date, git HEAD at run time, upstream SHA, qwen-tts/python/torch/ROCm/GPU/gfx, exact commands, exit codes, results. Transcripts are never hand-edited. Update `evidence/README.md` index for every new evidence file.
- **Test style (binding, matches existing suite):** `pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]`; module-scoped model fixtures with `loader.unload` teardown; primary net `testing.assert_wav_sane(wav, sr_expected=sr)`; distinction assertions robust-only via `conftest._distinct`; `_reseed()` before stochastic comparison pairs; seeds reduce variance and are never quality validation. Genuine failures are product bugs — never weaken a test silently; any downgrade is recorded in the task report and verifier briefing.
- **Honest boundaries:** if GPU evidence cannot be produced for a step, stop that step at `BLOCKED: hardware evidence unavailable` — never convert missing validation into a green claim.
- **Gating:** one logical commit per verified task (messages in each task's final step); verifier PASS precedes every commit; push happens exactly once after Task 5 + P0 report.
- **CPU suite command:** `.venv/bin/python -m pytest -m 'not gpu' -q` — must stay green after every task. GPU suite: `.venv/bin/python -m pytest -m gpu -q`.

## Verification Protocol (applies to Tasks 1–5; referenced by every gate step)

Dispatch a fresh general-purpose subagent with this contract verbatim, plus the task's acceptance criteria and file list:

> You are the independent release verifier. Assume the implementing agent may be wrong. Review the git diff, acceptance criteria, tests, raw evidence and documentation claims. Try to falsify the implementation. Do not modify the code unless explicitly asked. Return PASS or FAIL only after checking every required criterion.

The verifier must produce a report with exactly these sections: **Verdict** (PASS/FAIL); **Acceptance criteria** (checkbox per criterion); **Code reviewed** (exact files); **Tests executed** (exact commands and exit codes — the verifier re-runs the full CPU suite, the zero-patch parity test, and this task's GPU tests itself); **Runtime evidence** (exact evidence files reviewed); **Claims audit** (README/docs no stronger than measured evidence); **Regression audit** (previously green capabilities remain green); **Problems found** (every issue); **Final justification**. A PASS without test/evidence details is invalid. On FAIL: orchestrator dispatches a fix round carrying the findings, then a NEW verifier. Repeat until PASS.

## File Structure (new files created by this plan)

```
evidence/ground-truth-2026-09-20.md            (Task 0, audit output)
tests/test_generate_custom_voice_06b.py        (Task 1A)
tests/test_voice_clone_06b.py                  (Task 1B)
evidence/benchmark-06b-2026-09-20.json/.txt    (Task 1C, benchmark run artifacts)
tests/data/multilingual_samples.json           (Task 2A manifest)
scripts/validate_languages.py                  (Task 2E harness)
tests/test_validate_languages.py               (Task 2E CPU unit tests)
evidence/multilingual-matrix.json/.txt         (Task 2E run artifacts)
src/qwen3_tts_rocm/voice_workflow.py           (Task 3A backend)
tests/test_voice_workflow.py                   (Task 3D GPU integration test)
evidence/voice-workflow-2026-09-20.json/.txt   (Task 3E artifacts)
scripts/make_finetune_dataset.py               (Task 4A dataset generator)
tests/data/finetune_manifest.json              (Task 4A utterance manifest)
.upstream/Qwen3-TTS/                           (Task 4, gitignored pinned clone)
evidence/finetune-smoke-2026-09-20.json/.txt   (Task 4 artifacts)
docs/finetuning-rocm.md                        (Task 4E)
docs/p0-parity-report.md                       (final deliverable)
```

Modified: `scripts/benchmark.py` (0.6B aliases + memory/load metrics), `README.md` + `README_CN.md` (matrices, vocabulary), `CHANGELOG.md`, `src/qwen3_tts_rocm/demo/backend.py` + `demo/ui.py` (Voice Studio tab), `src/qwen3_tts_rocm/patch.py` → `compat.py` (Task 5C, with shim), `.gitignore` (`.upstream/`), `evidence/README.md` (index).

---

### Task 0: Ground-truth audit (prerequisite — its own chore commit, no verifier gate)

**Files:**
- Create: `evidence/ground-truth-2026-09-20.md`

**Interfaces:**
- Produces: the upstream HEAD SHA + finetuning layout facts that Tasks 1–4 briefs cite; nothing code-level.

- [ ] **Step 0.1: Dispatch the audit subagent** (general-purpose, read-only intent). Brief must instruct it to: record `git -C /home/amd/Desktop/Qwen3-TTS-ROCm rev-parse HEAD`; fetch upstream `git ls-remote https://github.com/QwenLM/Qwen3-TTS HEAD` and record the SHA; record `.venv/bin/pip show qwen-tts` version; inspect upstream `finetuning/` tree (via GitHub API `https://api.github.com/repos/QwenLM/Qwen3-TTS/contents/finetuning`) and record exact script filenames + README-reported entry points + requirements; confirm from the installed package (`.venv/lib/python3.12/site-packages/qwen_tts/`) the runtime `get_supported_languages()` list and the actual 0.6B-CustomVoice `instruct` behavior (grep the wrapper source for how `instruct` is handled — record whether it raises, warns, or is silently ignored); inventory `evidence/` (which capabilities already have real GPU transcripts); write everything to `evidence/ground-truth-2026-09-20.md` with a header (date, HEAD, upstream SHA, versions).
- [ ] **Step 0.2: Orchestrator reviews the audit** — confirms it answers every fact above; any gap → one follow-up message to the same subagent.
- [ ] **Step 0.3: Commit**

```bash
git add evidence/ground-truth-2026-09-20.md evidence/README.md
git commit -m "chore(evidence): record ground-truth audit (upstream SHA, finetuning layout, 0.6B instruct reality)"
```

---

### Task 1: Complete 0.6B end-to-end parity

**Files:**
- Create: `tests/test_generate_custom_voice_06b.py`, `tests/test_voice_clone_06b.py`
- Modify: `scripts/benchmark.py` (`build_call`, `DEFAULT_ALIASES` docstring, per-alias load timing + peak memory)
- Create: `evidence/benchmark-06b-2026-09-20.json`, `evidence/benchmark-06b-2026-09-20.txt`
- Modify: `README.md`, `README_CN.md`, `CHANGELOG.md`, `evidence/README.md`

**Interfaces:**
- Consumes: `loader.load("custom-voice-0.6b")`, `loader.load("base-0.6b")` (aliases exist in `models.REPOS`); `conftest` helpers; `testing.assert_wav_sane`; the bundled ref asset pattern from `tests/test_voice_clone_workflow.py`.
- Produces: the per-checkpoint validation matrix used by README (Task 5 refines it further).

- [ ] **Step 1.1: Write `tests/test_generate_custom_voice_06b.py`** — modeled on `tests/test_generate_custom_voice.py`, alias `custom-voice-0.6b`:

```python
# tests/test_generate_custom_voice_06b.py
"""Task 1A: GPU integration -- 0.6B CustomVoice E2E parity (real weights).

Upstream documents NO instruction control for 0.6B CustomVoice (1.7B has it).
The instruct test pins the INSTALLED package's actual behavior as discovered in
the Task 0 ground-truth audit; docs and UI communicate that boundary.
Keyword-first calls; robust-only distinction; assert_wav_sane primary net.
"""
from __future__ import annotations

import pytest
from conftest import _distinct, _reseed, _timed_generate

from qwen3_tts_rocm import loader, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

TEXT = "今天的天气真不错，适合去公园散步。"

@pytest.fixture(scope="module")
def cv06_model(gpu):
    m = loader.load("custom-voice-0.6b")
    yield m
    loader.unload(m)

def test_single(cv06_model):
    spks = cv06_model.get_supported_speakers()
    langs = cv06_model.get_supported_languages()
    assert len(spks) >= 9 and langs
    assert "auto" in [str(lang).lower() for lang in langs]
    wavs, sr = _timed_generate(cv06_model, "generate_custom_voice",
                               text=TEXT, language="Auto", speaker=spks[0],
                               max_new_tokens=512)
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr, min_energy_rms=1e-3)

def test_batch(cv06_model):
    wavs, sr = _timed_generate(
        cv06_model, "generate_custom_voice",
        text=[TEXT, TEXT], language="Auto",
        speaker=cv06_model.get_supported_speakers()[0], max_new_tokens=512,
    )
    assert len(wavs) == 2
    for w in wavs:
        testing.assert_wav_sane(w, sr_expected=sr)

def test_kwargs_passthrough(cv06_model):
    spk = cv06_model.get_supported_speakers()[0]
    _reseed()
    lo, lo_sr = _timed_generate(cv06_model, "generate_custom_voice", text=TEXT,
                                language="Auto", speaker=spk,
                                temperature=0.01, max_new_tokens=512)
    _reseed()
    hi, hi_sr = _timed_generate(cv06_model, "generate_custom_voice", text=TEXT,
                                language="Auto", speaker=spk,
                                temperature=1.9, max_new_tokens=512)
    testing.assert_wav_sane(lo[0], sr_expected=lo_sr)
    testing.assert_wav_sane(hi[0], sr_expected=hi_sr)
    assert lo[0].shape != hi[0].shape or float(abs(lo[0] - hi[0]).max()) > 0
```

Plus `test_instruct_behavior(cv06_model)`: written AFTER reading the Task 0 audit finding. If the wrapper raises on `instruct` for 0.6B → `pytest.raises` pinning the exception type; if it accepts-and-ignores → assert the call completes with a sane waveform and record in the module docstring "instruct accepted but has no documented effect on 0.6B (upstream: no instruction control)". Either way the observed behavior is pinned, never assumed.

- [ ] **Step 1.2: Run the new 0.6B CustomVoice tests**

Run: `.venv/bin/python -m pytest tests/test_generate_custom_voice_06b.py -m gpu -v -s`
Expected: all PASS (first run of new code may reveal genuine product issues → debug systematically, never loosen).

- [ ] **Step 1.3: Write `tests/test_voice_clone_06b.py`** — mirror of `tests/test_voice_clone_workflow.py` for alias `base-0.6b`: same `_asset_path`/`_ref_audio` helpers, same REF_TEXT (`"This tiny synthetic voice was cloned for automated testing."`), same MAX_NEW_TOKENS=512 guardrail, module fixture `bc06_model`; tests: `test_clone_with_ref_text`, `test_clone_x_vector_only`, `test_create_prompt_reuse` (asserts `VoiceClonePromptItem` invariants exactly as the 1.7B test does, reuse across TEXT/ALT_TEXT with `_distinct`), `test_save_load_roundtrip_parity` (official `{"items": [asdict(it) ...]}` save + `weights_only=True` reconstruct, field-for-field device-agnostic parity, then sane regeneration), `test_batch_clone`.

- [ ] **Step 1.4: Run the 0.6B clone tests**

Run: `.venv/bin/python -m pytest tests/test_voice_clone_06b.py -m gpu -v -s`
Expected: all PASS.

- [ ] **Step 1.5: Extend `scripts/benchmark.py`** — in `build_call`, treat 0.6B aliases as their family's entry point:

```python
if alias in ("custom-voice", "custom-voice-0.6b"):
    ...
if alias == "voice-design":
    ...
if alias in ("base", "base-0.6b"):
    ...
raise KeyError(f"unknown alias {alias!r}; benchmark supports: custom-voice, "
               f"custom-voice-0.6b, voice-design, base, base-0.6b")
```

Add per-alias metrics recorded into each cell/result: `load_seconds` (time `loader.load(alias)` → `loader.unload` around the alias's whole run, printed and stored in meta per alias) and `peak_alloc_gb` (`torch.cuda.reset_peak_memory_stats()` after load; `torch.cuda.max_memory_allocated()/2**30` read at alias end; None-guarded `if torch.cuda.is_available()` so CPU unit tests of pure helpers stay lightweight). Unit-test the new pure helpers if any are extracted; extend `tests/test_benchmark.py` for the widened alias set.

- [ ] **Step 1.6: Run the 0.6B benchmark and archive evidence**

Run (repo root, ~30–60 min GPU):
```bash
.venv/bin/python scripts/benchmark.py --aliases custom-voice-0.6b,base-0.6b \
    --json-out evidence/benchmark-06b-2026-09-20.json 2>&1 \
    | tee evidence/benchmark-06b-2026-09-20.txt
```
Expected: exit 0; JSON holds per-cell RTF/wall/audio-duration records plus load_seconds/peak_alloc_gb; `.txt` holds the verbatim transcript. Also record `git rev-parse HEAD`, upstream SHA (from Task 0), and env versions into the JSON `meta` if not already present via `collect_meta`.

- [ ] **Step 1.7: Documentation (1D)** — in `README.md` replace the "Official model repositories | 6 / 6 validated" row's ambiguity: keep the row as "6 / 6 load-validated" and add a per-checkpoint matrix directly below the "Verified, not promised" table:

```markdown
| Model | Load | E2E Generate | Benchmark |
|---|---|---|---|
| 1.7B CustomVoice | ✅ | ✅ | ✅ |
| 1.7B VoiceDesign | ✅ | ✅ | ✅ |
| 1.7B Base (clone) | ✅ | ✅ | ✅ |
| 0.6B CustomVoice | ✅ | ✅ | ✅ (see `evidence/benchmark-06b-2026-09-20.json`) |
| 0.6B Base (clone) | ✅ | ✅ | ✅ (same) |
| 12Hz Tokenizer | ✅ | codec ✅ | n/a |
```

Mirror the same table and wording in `README_CN.md` (加载 / 端到端合成 / 基准测试). Add a note: "0.6B CustomVoice has no instruction control upstream; the demo and docs reflect that boundary." Add a CHANGELOG entry under a new version heading describing Task 1 scope. Update `evidence/README.md` index with the two new artifacts.

- [ ] **Step 1.8: Full CPU suite**

Run: `.venv/bin/python -m pytest -m 'not gpu' -q`
Expected: all PASS (previous 213/214 + any new CPU tests).

- [ ] **Step 1.9: Verification gate** — dispatch verifier per §Verification Protocol with Task 1 acceptance criteria: real GPU tests for 0.6B CustomVoice; real GPU tests for 0.6B Base cloning; real generated WAV sanity; benchmark evidence for both 0.6B models; docs distinguish load vs functional validation; previous 1.7B tests still pass (verifier re-runs `tests/test_generate_custom_voice.py`, `tests/test_voice_design.py`, `tests/test_voice_clone_workflow.py` with `-m gpu`); no upstream patch added (verifier confirms `tests/test_official_demo_parity.py` + zero-patch check green).
- [ ] **Step 1.10: On PASS, commit**

```bash
git add -A
git commit -m "feat: validate Qwen3-TTS 0.6B generation on Radeon (E2E parity + benchmarks + validation-level matrix)"
```

---

### Task 2: Multilingual Radeon validation matrix

**Files:**
- Create: `tests/data/multilingual_samples.json`, `scripts/validate_languages.py`, `tests/test_validate_languages.py`
- Create: `evidence/multilingual-matrix.json`, `evidence/multilingual-matrix.txt`
- Modify: `README.md`, `README_CN.md`, `CHANGELOG.md`, `evidence/README.md`

**Interfaces:**
- Consumes: `loader.load("custom-voice")`, `loader.load("voice-design")`, `loader.load("base")`; official `get_supported_languages()` (runtime identifiers are lowercase, incl. `"auto"`).
- Produces: `validate_languages.load_manifest() -> list[Sample]` and `validate_languages.evaluate_waveform(wav, sr) -> CheckResult` (pure, CPU-testable); evidence schema consumed by README matrix.

- [ ] **Step 2.1: Write `tests/data/multilingual_samples.json`** — original authored sentences (no copyrighted prose), one per language:

```json
{
  "samples": [
    {"language": "Chinese",     "text": "秋风吹过湖面，带来一丝清凉的气息。"},
    {"language": "English",     "text": "The blue kite rises slowly over the quiet harbor."},
    {"language": "Japanese",    "text": "春の朝、小鳥たちが窓の外で楽しそうに鳴いています。"},
    {"language": "Korean",      "text": "가을 바람이 불면 단풍잎이 조용히 흔들립니다."},
    {"language": "German",      "text": "Der schnelle Zug hält morgen am kleinen Bahnhof."},
    {"language": "French",      "text": "La lumière du matin traverse les grandes fenêtres."},
    {"language": "Russian",     "text": "Утренний дождь оставил капли на зелёных листьях."},
    {"language": "Portuguese",  "text": "O barco pequeno volta ao porto antes do pôr do sol."},
    {"language": "Spanish",     "text": "El café de la esquina abre temprano todos los días."},
    {"language": "Italian",     "text": "La piazza vecchia è silenziosa nel primo pomeriggio."}
  ],
  "voice_design_descriptions": {
    "Chinese": "年轻女性，声音清亮，语速轻快",
    "English": "A calm middle-aged male voice, warm and unhurried",
    "Japanese": "若い女性の明るく澄んだ声、やや速いテンポ",
    "Korean": "젊은 남성의 차분하고 낮은 목소리",
    "German": "Eine freundliche junge Frauenstimme, klar und lebhaft",
    "French": "Une voix féminine douce et posée, débit modéré",
    "Russian": "Молодой мужской голос, спокойный и чистый",
    "Portuguese": "Uma voz masculina madura, grave e tranquila",
    "Spanish": "Una voz femenina joven, brillante y amable",
    "Italian": "Una voce maschile matura, calda e pacata"
  },
  "cross_lingual_clone_pairs": [
    {"ref_language": "English",  "target_language": "Chinese"},
    {"ref_language": "Chinese",  "target_language": "English"},
    {"ref_language": "Japanese", "target_language": "English"},
    {"ref_language": "French",   "target_language": "Chinese"}
  ]
}
```

Language names are canonical manifest keys; the script resolves each to the runtime identifier via case-insensitive match against `model.get_supported_languages()` and records the resolved value (drift between manifest and installed package = hard error, not silent skip).

- [ ] **Step 2.2: Write `scripts/validate_languages.py`** — lightweight import (stdlib-only at import time, GPU deps lazy, mirroring `benchmark.py` so CPU unit tests stay cheap). Structure:

```python
def load_manifest(path) -> Manifest          # parse + schema-validate (pure)
def resolve_language(canonical: str, supported: list[str]) -> str
    # case-insensitive match; raise KeyError naming both sides on miss (pure)
def evaluate_waveform(wav: "np.ndarray", sr: int) -> dict
    # finite / non-silent (rms>1e-3) / sr>0 / bounded duration (pure)
def run_custom_voice_matrix(model, manifest, ...) -> list[dict]   # GPU, lazy imports
def run_voice_design_matrix(model, manifest, ...) -> list[dict]   # GPU
def run_cross_lingual_clone(model, manifest, ...) -> list[dict]   # GPU:
    # per pair: ref = generate_custom_voice(ref_language sentence, speaker[0],
    #   max_new_tokens=512); prompt = create_voice_clone_prompt(
    #   ref_audio=ref, ref_text=<exact manifest sentence>);   # transcript rule
    #   out = generate_voice_clone(target sentence, language=target,
    #   voice_clone_prompt=prompt, max_new_tokens=512)
def main(argv=None) -> int
    # loads each model once (custom-voice -> voice-design -> base, unload between),
    # times every cell with time.perf_counter, aggregates pass/fail per language,
    # writes evidence/multilingual-matrix.json (env header: timestamp, GPU name,
    # gfx, ROCm/torch/qwen-tts versions, git HEAD, upstream SHA, model alias per
    # section; rows: language, wall_s, audio_s, checks, pass, error) and the
    # verbatim .txt transcript; exit 0 iff every cell passed
```

All GPU calls keyword-first with `max_new_tokens=512`; failures recorded as failed rows with `error` text, never crash the whole run.

- [ ] **Step 2.3: Write `tests/test_validate_languages.py`** (CPU, no gpu marker) — pure-logic tests with real fixtures: manifest schema validation rejects missing keys/unknown languages; `resolve_language` matches case-insensitively and raises on drift; `evaluate_waveform` flags NaN/silent/wrong-sr/over-long synthetic numpy arrays and passes a sane synthetic sine burst.

- [ ] **Step 2.4: Run CPU unit tests**

Run: `.venv/bin/python -m pytest tests/test_validate_languages.py -q`
Expected: PASS.

- [ ] **Step 2.5: Run the full GPU matrix and archive evidence** (~1–2 h; 10 CV + 10 VD + 4 clone pairs, one model resident at a time):

```bash
.venv/bin/python scripts/validate_languages.py 2>&1 | tee evidence/multilingual-matrix.txt
```
Expected: exit 0; JSON + verbatim transcript written; every cell `pass: true`.

- [ ] **Step 2.6: Documentation (2F)** — README (EN+CN): new "Multilingual capability matrix" section near the capability tables:

```markdown
| Language | CustomVoice | VoiceDesign | Clone (cross-lingual) |
|---|---|---|---|
| Chinese | ✅ | ✅ | ✅ (ref: en, fr) |
| English | ✅ | ✅ | ✅ (ref: zh, ja) |
| ... all 10 rows from evidence ... |
```

With the caption: "✅ = end-to-end generation completed on Radeon (waveform sanity: finite, non-silent, valid sample rate, bounded duration) — see `evidence/multilingual-matrix.json`. This is NOT a pronunciation-quality claim. Cross-lingual clone coverage is representative (4 pairs), not exhaustive." Update `evidence/README.md` + CHANGELOG.

- [ ] **Step 2.7: Full CPU suite** — Run: `.venv/bin/python -m pytest -m 'not gpu' -q` — Expected: PASS.
- [ ] **Step 2.8: Verification gate** — §Verification Protocol with Task 2 acceptance criteria: all current official major languages exercised through at least CustomVoice (verifier cross-checks JSON rows against runtime `get_supported_languages()`); machine-readable evidence exists and is internally consistent with the `.txt` transcript; no pronunciation-quality overclaim anywhere; representative cross-lingual clone coverage present; docs traceable to evidence; prior GPU suite still green (verifier re-runs `-m gpu` for Tasks' relevant modules: 1.7B trio + Task 1's two 0.6B modules); no upstream patches.
- [ ] **Step 2.9: On PASS, commit**

```bash
git add -A
git commit -m "feat: add Radeon multilingual capability matrix (10 languages, CV+VD+representative cross-lingual clone)"
```

---

### Task 3: First-class Voice Design → Clone → Reuse workflow

**Files:**
- Create: `src/qwen3_tts_rocm/voice_workflow.py`, `tests/test_voice_workflow.py`
- Modify: `src/qwen3_tts_rocm/demo/backend.py`, `src/qwen3_tts_rocm/demo/ui.py` (new tab), plus `src/qwen3_tts_rocm/demo/__init__.py` exports if pattern requires
- Create: `evidence/voice-workflow-2026-09-20.json`, `evidence/voice-workflow-2026-09-20.txt`
- Modify: `README.md`, `README_CN.md`, `CHANGELOG.md`, `evidence/README.md`

**Interfaces:**
- Consumes: official `generate_voice_design` / `create_voice_clone_prompt` / `generate_voice_clone`; the official payload format `{"items": [asdict(item) ...]}` + `torch.save`/`torch.load(weights_only=True)` reconstruction (proven in `tests/test_voice_clone_workflow.py`).
- Produces (exact API later tasks/demo rely on):

```python
# src/qwen3_tts_rocm/voice_workflow.py
def design_voice(model, *, text: str, language: str, description: str,
                 max_new_tokens: int = 512) -> DesignResult
    # 1) official generate_voice_design(text=..., instruct=description,
    #    language=..., max_new_tokens=...) -> (wavs, sr)
    # 2) official create_voice_clone_prompt(ref_audio=(wavs[0], sr),
    #    ref_text=text)  -- transcript rule: ref_text IS the design text
    # 3) returns DesignResult(preview=(wav, sr), prompt_items=items,
    #    description=..., language=..., ref_text=text,
    #    timings={"design_s": ..., "prompt_s": ...})
def reuse_voice(model, *, prompt_items, text: str, language: str,
                max_new_tokens: int = 512) -> tuple["np.ndarray", int, float]
    # official generate_voice_clone(..., voice_clone_prompt=prompt_items);
    # returns (wav, sr, generate_s)
def save_voice(result: DesignResult, path) -> None
    # official-compatible payload ONLY: {"items": [asdict(it) ...], plus a
    # sidecar "voice_meta" dict (description/language/ref_text); torch.save
def load_voice(path) -> DesignResult
    # torch.load(weights_only=True) may reject non-tensor sidecar strings --
    # if so, store meta in a sibling .json next to the .pt (decide by TEST,
    # Step 3.2 pins the working variant; do not guess)
```

- [ ] **Step 3.1: Write `src/qwen3_tts_rocm/voice_workflow.py`** per the interface above; keyword-first, `max_new_tokens=512` default, no proprietary voice representation — the `.pt` payload's `items` key is byte-format-identical to the official demo's.
- [ ] **Step 3.2: Write `tests/test_voice_workflow.py`** (GPU):

```python
pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]
# module fixtures: vd_model = loader.load("voice-design") ... then swap-unload
# and bc_model = loader.load("base") ... unload (design runs on VoiceDesign,
# clone reuse runs on Base -- exactly like the official demo's model split)

def test_design_voice_creates_prompt_and_preview(...):
    res = voice_workflow.design_voice(vd_model, text=TEXT, language="Auto",
                                      description="年轻女性，声音清亮，语速轻快")
    testing.assert_wav_sane(res.preview[0], sr_expected=res.preview[1])
    assert isinstance(res.prompt_items[0], VoiceClonePromptItem)
    assert res.prompt_items[0].ref_text == TEXT          # transcript rule
    assert res.timings["design_s"] > 0 and res.timings["prompt_s"] > 0

def test_reuse_across_two_sentences(...):
    # reuse_voice on bc_model with res.prompt_items for TEXT and ALT_TEXT;
    # both sane; _distinct; returns generate_s separately recorded

def test_save_load_roundtrip_then_reuse(...):
    # save_voice -> load_voice (variant pinned by this test per Step 3.1 note)
    # -> field-for-field parity (device-agnostic, official-fields equality)
    # -> reuse_voice still generates a sane waveform

def test_official_schema_preserved(...):
    # payload["items"][0] keys == official dataclass field names exactly
```

(Also the standalone regression: existing `tests/test_voice_design.py` + `tests/test_voice_clone_workflow.py` still pass — they are the proof the standalone paths remain intact.)
- [ ] **Step 3.3: Run the workflow GPU tests**

Run: `.venv/bin/python -m pytest tests/test_voice_workflow.py -m gpu -v -s`
Expected: PASS.

- [ ] **Step 3.4: Demo integration** — `backend.py`: add `SynthesisService` methods `voice_studio_design(text, language, description, gen_kwargs)` (calls `voice_workflow.design_voice`, keeps the DesignResult in a per-session store keyed by generated voice id, returns preview audio + timings), `voice_studio_save(voice_id, name)` (persists via `save_voice` under a user-data dir, default `<cwd>/voices/<name>.pt` + sidecar), `voice_studio_list()` / `voice_studio_generate(name, text, language, gen_kwargs)` (load + `reuse_voice` on the Base model). `ui.py`: new top-level tab `⑥ Voice Studio (音色工坊)` after ⑤ History, following existing tab patterns (gr.Row blocks, `build_callbacks` wiring, bilingual labels):

```
[voice description textbox | language dropdown | design button]
  -> preview Audio component + design_s/prompt_s timing labels
[voice name textbox | save button]  -> saved-voices dropdown refresh
[saved voice dropdown | target text textbox | language dropdown | generate button]
  -> output Audio + generate_s label
```

No download/upload/retype hop anywhere in the flow. Existing tabs ①–⑤ untouched.
- [ ] **Step 3.5: CPU demo tests** — extend `tests/test_demo_backend.py` (and `test_demo_ui.py` if the tab adds UI-level units) following the file's existing mocking conventions for the three new service methods.
- [ ] **Step 3.6: Run CPU suites** — Run: `.venv/bin/python -m pytest -m 'not gpu' -q` — Expected: PASS.
- [ ] **Step 3.7: Evidence run** — a small driver script or `pytest -s` teed:

```bash
.venv/bin/python -m pytest tests/test_voice_workflow.py -m gpu -v -s 2>&1 \
  | tee evidence/voice-workflow-2026-09-20.txt
```
plus a hand-run snippet (recorded in the same `.txt` via append with a marked separator, or a `scripts/`-less `python - <<'EOF'` block in the transcript) that walks design → save → load → reuse twice and prints the three phase timings; JSON artifact `evidence/voice-workflow-2026-09-20.json` records `{design_s, prompt_s, reuse_s_1, reuse_s_2, meta:{env header, HEAD, upstream SHA}}`. The three phases are recorded separately — never collapsed.
- [ ] **Step 3.8: Docs** — README (EN+CN): feature section describing the Voice Studio workflow (describe → preview → save → reuse) linking evidence; CHANGELOG entry; `evidence/README.md` index.
- [ ] **Step 3.9: Verification gate** — §Verification Protocol with Task 3 acceptance criteria: real one-click workflow with no manual download/re-upload hop (verifier inspects the tab wiring); official APIs only (verifier greps `voice_workflow.py` for any non-official generation path — must be none); official-compatible reusable data (payload schema check vs official dataclass fields); real GPU integration test passing; reusable prompt generates multiple new sentences (evidence shows two reuse renders); three-phase timing evidence archived; standalone VoiceDesign + Voice Clone paths intact (verifier re-runs `tests/test_voice_design.py` + `tests/test_voice_clone_workflow.py`).
- [ ] **Step 3.10: On PASS, commit**

```bash
git add -A
git commit -m "feat: add voice-design-to-reusable-voice workflow (Voice Studio tab + official-prompt persistence)"
```

---

### Task 4: ROCm fine-tuning smoke + reproducible reference

**Files:**
- Create: `scripts/make_finetune_dataset.py`, `tests/data/finetune_manifest.json`
- Create (gitignored): `.upstream/Qwen3-TTS/` (pinned clone), `.work-finunete/` scratch
- Create: `evidence/finetune-smoke-2026-09-20.json`, `evidence/finetune-smoke-2026-09-20.txt`, checkpoint artifacts path recorded in evidence
- Create: `docs/finetuning-rocm.md`
- Modify: `.gitignore` (add `.upstream/`, `.work-finetune/`, `voices/`), `README.md`/`README_CN.md` (capability row), `CHANGELOG.md`, `evidence/README.md`

**Interfaces:**
- Consumes: upstream `finetuning/` scripts (exact entry points recorded by Task 0 audit; GitHub is reachable from this host — verified `git ls-remote` works).
- Produces: `docs/finetuning-rocm.md` PROVEN/NOT PROVEN lists + evidence package that Task 5's capability table cites.

- [ ] **Step 4.1: Pin + clone upstream (no vendoring)**

```bash
UP_SHA=$(git ls-remote https://github.com/QwenLM/Qwen3-TTS HEAD | cut -f1)
git clone https://github.com/QwenLM/Qwen3-TTS.git .upstream/Qwen3-TTS
git -C .upstream/Qwen3-TTS checkout "$UP_SHA"
grep -q '^\.upstream/' .gitignore || echo '.upstream/' >> .gitignore
grep -q '^\.work-finetune/' .gitignore || echo '.work-finetune/' >> .gitignore
grep -q '^voices/' .gitignore || echo 'voices/' >> .gitignore
```
Record `$UP_SHA` — it goes into every Task 4 evidence header. If it differs from Task 0's SHA, note both and proceed with the newer pinned one.
- [ ] **Step 4.2: Inspect upstream `finetuning/`** — read its README/scripts (Task 0 recorded the layout; re-verify against the pinned tree): identify (a) the data-preparation entry point and expected input format, (b) the training entry point and its step/batch/output/checkpoint args, (c) the documented reload path. Write the exact discovered commands + file list into the head of `evidence/finetune-smoke-2026-09-20.txt` BEFORE running anything. If the audited layout no longer matches, stop and record the delta — do not guess.
- [ ] **Step 4.3: Generate the self-made dataset (4A)** — write `scripts/make_finetune_dataset.py` + `tests/data/finetune_manifest.json`: manifest = 12 utterances (6 Chinese + 6 English short original sentences, distinct from Task 2's); generator loads `custom-voice` (1.7B) via our loader, `_reseed()` before each render for variance control, renders each utterance (`max_new_tokens=512`), writes `wav` + exact text transcript pair into `.work-finetune/data/` in the format upstream's preparation step expects (per Step 4.2 findings). Run it; transcript goes to evidence. No third-party speech anywhere.
- [ ] **Step 4.4: Official data preparation (4B)** — run upstream's preparation entry point on the dataset exactly as its README prescribes (tokenizer → audio codes → training format). Tee output to evidence. Any upstream defect found: document + minimal reproducer in the evidence file; prefer reporting/fixing upstream; NO permanent local fork (P2).
- [ ] **Step 4.5: Training smoke (4C)** — run upstream's training script on **1.7B Base** (fallback 0.6B Base only if memory/time-bound — record which and why) with a deliberately small configuration: the smallest officially-supported batch size, ~10–50 optimizer steps (pick the smallest that exercises save), seq limits per upstream defaults, output to `.work-finetune/runs/smoke/`. Tee the full transcript to evidence. Capture into `evidence/finetune-smoke-2026-09-20.json`: model, precision, GPU, ROCm, torch, qwen-tts version, upstream SHA, batch size, seq limits, step count, wall time, peak memory (from `torch.cuda.max_memory_allocated` if the script exposes hooks, else rocm-smi sampling noted as observation), final loss values verbatim (no interpretation).
- [ ] **Step 4.6: Checkpoint reload + synthesis (4D, mandatory)** — follow upstream's documented reload path exactly (per Step 4.2c). Load the saved checkpoint, run one `generate_voice_clone`-equivalent official inference on a manifest sentence, `assert_wav_sane`-style waveform checks, tee everything. A checkpoint that merely exists is NOT sufficient — synthesis must succeed.
- [ ] **Step 4.7: Write `docs/finetuning-rocm.md` (4E)** — sections: Overview (goal = execution validation, not convergence); Environment (pinned versions + upstream SHA); Reproduction (exact commands, manifest, dataset generator); What was run (steps, wall time, loss verbatim); **PROVEN**: preprocessing works / training runs N steps / checkpoint saves / checkpoint reloads / inference produces sane audio — each linked to evidence; **NOT PROVEN**: speaker similarity quality / convergence quality / optimal hyperparameters / long-run stability / multi-speaker training / production scalability; Limitations (synthetic self-generated dataset — no quality claims).
- [ ] **Step 4.8: Docs touchpoints** — README/README_CN capability rows for fine-tuning become 🟡→✅-scoped per the doc's PROVEN list wording; CHANGELOG; `evidence/README.md` index. The scratch dirs stay gitignored; NOTHING from `.upstream/` or checkpoints enters the commit (verifier checks `git status` clean of them).
- [ ] **Step 4.9: Full CPU suite** — Run: `.venv/bin/python -m pytest -m 'not gpu' -q` — Expected: PASS (plus any new CPU tests for the dataset-generator's pure helpers if extracted).
- [ ] **Step 4.10: Verification gate** — §Verification Protocol with Task 4 acceptance criteria: official fine-tuning workflow used (verifier confirms the commands in evidence match upstream's scripts — no forked trainer); real ROCm training steps executed (loss/step lines in transcript); checkpoint artifact exists (verifier inspects the recorded path); checkpoint reloaded and post-finetune inference produced a sane waveform (transcript); raw evidence archived; no unsupported quality claim anywhere in docs; zero upstream patches (upstream tree still pristine at pinned SHA — verifier runs `git -C .upstream/Qwen3-TTS status --porcelain` expecting empty).
- [ ] **Step 4.11: On PASS, commit**

```bash
git add -A
git commit -m "feat: validate Qwen3-TTS fine-tuning on ROCm (official-workflow smoke: prep, N steps, save, reload, sane synthesis)"
```

---

### Task 5: Align the repository with the North Star

**Files:**
- Modify: `README.md`, `README_CN.md`, `CHANGELOG.md`, `src/qwen3_tts_rocm/patch.py` → `src/qwen3_tts_rocm/compat.py` (+ shim if public), possibly `src/qwen3_tts_rocm/__init__.py`, `scripts/download_models.sh` (help text), `docs/troubleshooting.md` (vocabulary), `evidence/README.md`
- Create: `docs/p0-parity-report.md`

**Interfaces:**
- Consumes: every prior task's evidence + verifier PASS records (SHAs, test counts, benchmark numbers).
- Produces: the public capability narrative + the final report; nothing code-behavioral except the rename.

- [ ] **Step 5.1: README capability table (5A)** — insert directly under the hero/tagline: the North Star sentence verbatim, then the four-state legend and a consolidated capability matrix (CustomVoice 1.7B/0.6B, VoiceDesign 1.7B, Voice Clone 1.7B, Base 0.6B, reusable clone prompt, Design→Clone→Reuse, 12Hz tokenizer, multilingual matrix, fine-tuning, vLLM-Omni 🚫→roadmap, true streaming 🚫→roadmap). Every ✅ cell links its evidence file. No percentage scores. Mirror in README_CN.
- [ ] **Step 5.2: Conceptual correctness audit (5B)** — `grep -rn -iE "clone|克隆" README.md README_CN.md docs/ scripts/download_models.sh src/qwen3_tts_rocm/models.py` and fix every description that frames CustomVoice as reference-audio cloning; CustomVoice = preset-speaker/custom-voice generation; Base = zero-shot cloning + fine-tuning family. EN and CN fixed identically.
- [ ] **Step 5.3: `patch.py` → `compat.py` (5C)** — first `grep -rn "qwen3_tts_rocm.patch\|from .patch\|import patch" --include='*.py' --include='*.md' .` to map consumers. `git mv src/qwen3_tts_rocm/patch.py src/qwen3_tts_rocm/compat.py`; keep a shim at the old path ONLY if external consumers exist (tests/docs/README references): shim = `from qwen3_tts_rocm.compat import *  # noqa: F401,F403` plus explicit `__all__` re-export and a docstring pointing to `compat`. Update every internal import/test/reference. Full CPU suite after.
- [ ] **Step 5.4: Vocabulary sweep (5D)** — `grep -rn -iE "all models validated|6 / 6 validated|全部.*验证|models are validated" .` (excluding evidence transcripts — those are immutable) and replace with the four-state vocabulary per the spec. Every replacement keeps surrounding meaning intact.
- [ ] **Step 5.5: Hardware humility (5E)** — verify no claim generalizes gfx1151 to all Radeons; compatibility statements reference the evidence-driven matrix; community-report template unchanged in its measured-report requirements.
- [ ] **Step 5.6: Follow-up roadmap (5F)** — append a clearly-labeled "Roadmap (not yet validated)" section to README/README_CN: vLLM-Omni on ROCm (feasibility-first; ladder: PyTorch/qwen-tts ROCm → offline inference → performance characterization → online serving when upstream supports it → concurrency → production guidance) and True Streaming (required measurements: time-to-first-audio, chunk cadence, total RTF, buffer underrun, long-text; explicit note: upstream's 97 ms is not a Radeon number).
- [ ] **Step 5.7: Final full regression** — Run ALL of:
```bash
.venv/bin/python -m pytest -m 'not gpu' -q
.venv/bin/python -m pytest -m gpu -q          # full GPU suite: original 25 + Tasks 1-3 additions
git -C .upstream/Qwen3-TTS status --porcelain  # empty
git status --porcelain                          # only intended Task 5 changes uncommitted
```
Expected: every suite green; upstream pristine.
- [ ] **Step 5.8: Write `docs/p0-parity-report.md`** — per the spec's final-deliverable structure: North Star; repository SHAs (project final + upstream pinned); Tasks 1–5 each with verifier PASS; capability matrix (Capability × Model × Radeon status × Evidence link); benchmarks summary (1.7B prior + 0.6B new, no overstated precision); newly proven Radeon value beyond upstream; remaining gaps (vLLM-Omni, true streaming, other Radeon architectures, long-run fine-tuning, quality evaluation); zero-patch audit (explicit confirmation upstream modifications = 0); test summary (CPU count / GPU count / subagent verification 5/5 PASS); changed files summary; task commit list.
- [ ] **Step 5.9: Final verification gate** — §Verification Protocol with Task 5 acceptance criteria: North Star visible in README; capability matrix matches evidence exactly (verifier spot-checks ≥5 cells against evidence files); EN/CN consistent (verifier compares tables/wording); no CustomVoice/Base conceptual mistakes remain (verifier re-greps); no multi-GPU generalization claims; "zero upstream patches" still literally true; ALL suites green (verifier re-runs both CPU and full GPU suites itself); every green capability links to evidence.
- [ ] **Step 5.10: On PASS, commit**

```bash
git add -A
git commit -m "docs: align project with capability-first Radeon validation (North Star, 4-state matrix, vocabulary, p0 report)"
```

- [ ] **Step 5.11: The single push** (orchestrator, after the report is summarized to the user):

```bash
git push origin main
```

---

## Self-Review (completed)

1. **Spec coverage:** Task 0 = §Task 0; Tasks 1A/1B/1C/1D → plan Task 1 Steps 1.1–1.7; 2A–2F → Task 2 Steps 2.1–2.6; 3A–3E → Task 3 Steps 3.1–3.8; 4A–4E → Task 4 Steps 4.3–4.7 (+4.1/4.2 execution approach); 5A–5F + final report + push → Task 5 Steps 5.1–5.11. Verification protocol, regression gates, evidence policy, git discipline all in Global Constraints. Gaps: none found.
2. **Placeholder scan:** two deliberate discovery points remain — 0.6B `instruct` behavior (Step 1.1 pins it from Task 0's audit with both branches specified) and upstream finetuning entry-point names (Step 4.2 requires recording them into evidence before running, with a stop-and-delta rule). Both are concrete actions with defined outputs, not TBDs. `load_voice` sidecar decision is resolved by test in Step 3.2, not guessed.
3. **Type consistency:** `DesignResult` fields (`preview`, `prompt_items`, `description`, `language`, `ref_text`, `timings`) used identically in Steps 3.1, 3.2, 3.4; `reuse_voice` returns `(wav, sr, generate_s)` consistently; evidence filenames identical between task steps and the File Structure map.
