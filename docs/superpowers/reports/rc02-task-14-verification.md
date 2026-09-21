# Task 14 (Quality benchmark foundation, brief 15) — Independent Release Verification

## 1. Verdict

**FAIL** — one acceptance-criterion element is not met and is contradicted by
the evidence file itself: the pesq ITU IPR notice was **not captured verbatim**
in the committed transcript (the Section A capture step failed at execution and
was never corrected), yet three artifacts (transcript Section B,
`evidence/README.md`, the task report) claim it was "captured verbatim", and
the Step 14.6 self-check printed `[PASS] ITU IPR notice captured verbatim` over
that gap. Every other criterion (2–9) is verified PASS with independent
runtime/executable evidence, including criteria the implementer's report got
wrong in both directions (the push actually succeeded; the verbatim capture
actually failed).

## 2. Exact commit / working-tree SHA reviewed

- Commit under review: `c07055a2956875f072564f7c9d39e679fc2c6636`
  ("feat(eval): automated TTS quality benchmark foundation (CER/WER + speaker
  similarity)"), parent `7cffe845841c6951ee703aadceabc5feff320d19`.
- Working tree during verification: `git rev-parse HEAD` =
  `c07055a2956875f072564f7c9d39e679fc2c6636`; `git status --porcelain` empty
  (0 lines) — tree clean.

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Licensing log complete, per-candidate verdicts, license names as verified at execution, resemblyzer MIT→Apache-2.0 correction recorded, pesq REJECTED with IPR notice **captured verbatim**, pesq never installed | **FAIL (partial)** | Verdict table complete with all 5 candidates and licenses-at-execution; resemblyzer correction explicitly recorded (transcript lines 168–171: "BRIEF CORRECTION: brief sketch said 'MIT'; actual at execution is Apache-2.0"); `pip show pesq` rc 1 in `.venv`; pesq not a dependency in pyproject. BUT the verbatim IPR capture is ABSENT: transcript line 153 "verbatim header of one bundled ITU source file:" is followed by the probe's failure output — `file: ` (empty) + `(no .c file at top level; listing tree)` — and no corrected follow-up was ever appended (unlike smoke-1b and venv-import-verify-2, which did get corrections). Section B line 210 claims "(captured verbatim in Section A, pesq/pesqmod.c)" — contradicted by Section A itself. |
| 2 | Only ACCEPTed deps in `[project.optional-dependencies] quality`, all installed in `.venv` | PASS | pyproject quality = exactly `jiwer==4.0.0`, `resemblyzer==0.1.4`, `openai-whisper==20250625`, `pystoi==0.4.1`; `pip show` rc 0 for all four in `.venv`; pesq appears in pyproject only inside the documenting comment (line 52), not as a dependency. |
| 3 | Unit tests green; full CPU suite 295 passed / 38 deselected | PASS | Ran myself: `tests/test_quality_eval.py -v` → 28 passed, 2 warnings, 2.88 s; `-m 'not gpu' -q` → **295 passed, 38 deselected** in 17.66 s; re-ran excluding the new file → 267 passed / 38 deselected, confirming the +28 delta exactly. |
| 4 | Evidence rows match manifest (5 cases; clone speaker_sim + ref exists; seed recorded); values verbatim, no interpretive adjectives | PASS | Cross-checked programmatically: row ids/texts == manifest ids/texts in order; seed 20260921 in both; clone row carries `speaker_sim_ref_vs_output: 0.6728`; manifest `ref` resolves to existing `src/qwen3_tts_rocm/demo/assets/ref_en.wav` (134444 bytes). Adjective grep (good/bad/poor/excellent/acceptable/garbled/unintelligible/hallucinat/degraded/perfect/clean/…) over the JSON: single hit = "clean-reference/degraded-signal pair" inside the STOI scope note — standard STOI terminology, not a quality interpretation; rows themselves are pure numbers/verbatim transcripts. |
| 5 | Performance vs quality separation; `scripts/benchmark.py` untouched | PASS | `git diff --stat 7cffe84..c07055a -- scripts/benchmark.py` → 0 lines (last touched by older commit fb8bc60); grep for rtf/wall/latency/speed/faster/slower over `evidence/quality-benchmark-2026-09-21.json` → no matches (rc 1); no composite score key; gen/load timings exist only as console progress lines, never in rows or scores. |
| 6 | Hand-check one CER and one WER example | PASS | Independently recomputed with pure-Python Levenshtein (no jiwer): CER("你好世界","你号世界") = 1/4 = 0.25; CER("天气很好","天气很好好") = 1/4 = 0.25; WER("hello world","hello there world") = 1 insertion / 2 ref words = 0.5; WER 1-of-5 substitution = 0.2. Also recomputed the evidence rows: zh-mid CER = 4 subs / 20 chars = 0.2 and clone-ref CER = 28 edits / 10 chars = 2.8 — both match the JSON verbatim. |
| 7 | Manifest clone-ref asset exists on disk and is the same asset used by `tests/test_voice_clone_workflow.py` | PASS | Both reference `qwen3_tts_rocm/demo/assets/ref_en.wav`; the clone test resolves it via `importlib.resources` at `tests/test_voice_clone_workflow.py:82` and its docstring lines 27–28; file exists at `src/qwen3_tts_rocm/demo/assets/ref_en.wav`. |
| 8 | Whisper model choice + device recorded; GPU phases rocm-smi-wrapped | PASS (minor note) | JSON `meta.asr` = {model: "tiny", temperature: 0, device: "cuda if torch.cuda.is_available() else cpu"}; the tiny-on-GPU probe (Section C smoke 4a: loaded on cuda, transcribe 5.00 s, rc 0, rocm-smi PRE/POST) plus `meta.gpu` (gfx1151) document tiny on GPU with no fallback needed; the Step 14.5 evidence run is wrapped by rocm-smi PRE/POST snapshots. Minor: `asr.device` records the policy expression, not the runtime-resolved string ("cuda"); resolved-device is established by the gate probe + meta.gpu, so the criterion substance holds. |
| 9 | Commit c07055a pushed; tree clean | PASS | `git ls-remote origin main` timed out (github.com:443 currently blocked, rc 124 — the known host flakiness), but: (a) GitHub API over the reachable api.github.com host returns `"sha": "c07055a2956875f072564f7c9d39e679fc2c6636"` for `commits/main`; (b) reflog shows `refs/remotes/origin/main@{2026-09-21 18:24:13 +0800}: update by push` → c07055a (a server-accepted push is the only thing that writes that). Remote main == local main == c07055a; porcelain empty. |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-14-brief.md` (task brief)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-14-report.md` (implementer report)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-7cffe84..c07055a.diff` (full 2043-line review package, read end to end)
- `scripts/quality_eval.py` (via diff, 557 lines)
- `tests/test_quality_eval.py` (via diff, 268 lines) + executed
- `tests/data/quality_benchmark_manifest.json` (via diff + loaded programmatically)
- `evidence/quality-benchmark-2026-09-21.json` (via diff + loaded/inspected on disk)
- `evidence/quality-benchmark-2026-09-21.txt` (988 lines — inspected on disk, not only via diff: Section A lines 118–155, verdict table 157–233, tail 959–988)
- `pyproject.toml` quality extras block
- `tests/test_voice_clone_workflow.py` (asset reference, lines 27–28, 82)
- `.work-quality/pesq-0.0.4/pesq/pesqmod.c` (untracked scratch sdist, for independent IPR verification)
- `evidence/README.md` (new rows)

## 5. Exact commands executed

1. `git log --oneline -5`; `git status --porcelain`; `git rev-parse HEAD`
2. `git show --stat c07055a | head -40`
3. `grep -n -i "psytechnics|opticom|intellectual property|IPR|commercial use" evidence/quality-benchmark-2026-09-21.txt`
4. `grep -n "verbatim" evidence/quality-benchmark-2026-09-21.txt`
5. `ls .work-quality/ ; ls .work-quality/pesq-0.0.4/pesq/`
6. `for p in jiwer resemblyzer pystoi pesq: .venv/bin/pip show $p` (+ `openai-whisper`)
7. `head -60 .work-quality/pesq-0.0.4/pesq/pesqmod.c | grep -i …`; `grep -rn -il "intellectual property" .work-quality/pesq-0.0.4/pesq/`
8. `.venv/bin/python -m pytest tests/test_quality_eval.py -v`
9. `.venv/bin/python -m pytest -m 'not gpu' -q`
10. `.venv/bin/python -m pytest -m 'not gpu' -q --ignore=tests/test_quality_eval.py`
11. Pure-Python Levenshtein re-computation of 4 unit-test expectations + 2 evidence-row CERs (heredoc script, no jiwer)
12. `git diff --stat 7cffe84..c07055a -- scripts/benchmark.py | wc -l`; `git log --oneline -1 -- scripts/benchmark.py`; perf-keyword grep over the evidence JSON
13. Interpretive-adjective grep over the evidence JSON
14. Python cross-check of manifest vs evidence rows (ids/texts/seed/speaker_sim)
15. `timeout 60 git ls-remote origin main`; `git rev-parse origin/main`; `git status --porcelain | wc -l`
16. `git reflog show origin/main --date=iso`
17. `timeout 30 curl -s https://api.github.com/repos/AIwork4me/Qwen3-TTS-ROCm/commits/main | grep -m1 '"sha"'`
18. `sed -n` over pyproject optional-dependencies block; `grep -n pesq pyproject.toml`

## 6. Exit codes

- All `pip show` (jiwer/resemblyzer/pystoi/openai-whisper): rc 0; `pip show pesq`: rc 1 (absent — required).
- `pytest tests/test_quality_eval.py -v`: rc 0 (28 passed).
- `pytest -m 'not gpu' -q`: rc 0 (295 passed, 38 deselected).
- `pytest -m 'not gpu' -q --ignore=tests/test_quality_eval.py`: rc 0 (267 passed, 38 deselected).
- `git ls-remote origin main`: rc 124 (timeout — github.com blocked at verification time; superseded by api.github.com rc 0 and the push reflog).
- `curl api.github.com .../commits/main`: rc 0, sha c07055a2956875f072564f7c9d39e679fc2c6636.
- Perf-keyword and interpretive-adjective greps: rc 1 (no matches) for perf; rc 0 for the adjective sweep with a single benign STOI-terminology hit (see criterion 4).
- Everything else (git log/status/rev-parse/reflog/show/diff, ls, sed, greps over the transcript): rc 0.

## 7. Runtime evidence inspected

- Transcript (988 lines) on disk: Section A license probes (PyPI license_expression/classifier for all five candidates captured before verdicts), failed verbatim-header capture at lines 153–155, Section B verdict table, Section C scratch-venv installs + smokes (jiwer hand-computed 0.5/0.25, pystoi synthetic pair, resemblyzer cosine 1.000/0.322, whisper tiny GPU 5.00 s / CPU 4.81 s, all with RC lines), Section D `.venv` install of exactly the four ACCEPTed + `pip check` clean + pesq-absence rc 1, Step 14.5 evidence run with rocm-smi PRE/POST and per-case verbatim transcripts, JSON sha256 `757314bc…` captured in-transcript, Step 14.6 self-check (initial FAIL on 5 verdict needles, corrected follow-up PASS — both honestly kept).
- Evidence JSON on disk: meta (git_head 7cffe84 pre-commit as designed, versions incl. "pesq: not installed (REJECTED…)"), omitted_metrics (pesq REJECT + stoi scope), seed, 5 rows verbatim.
- `.venv` package state (pip show) — runtime.
- Test execution — runtime (twice for the count, once for the delta).
- Remote state: api.github.com commits/main + `origin/main` push reflog — runtime.
- `.work-quality/pesq-0.0.4/pesq/pesqmod.c` — runtime inspection of the still-present sdist: the "PESQ Intellectual Property Rights Notice" is real and reads exactly as Section B paraphrases/quotes (owners "British Telecommunications plc … assigned to Psytechnics Limited" and "Royal KPN NV … assigned to OPTICOM GmbH"; RESTRICTIONS "alter, duplicate, modify, adapt, or translate …" and "sell, hire, loan, distribute, dispose or put to any commercial use …").

## 8. Regression tests

- Targeted: `tests/test_quality_eval.py` → 28/28 passed (cer/wer hand-computed math, normalization pins, empty-hyp=1.0, empty-ref raises, speaker_sim on synthetic sines incl. path/memory parity, manifest schema incl. 7 rejection cases, pesq/stoi documentation contracts).
- Full CPU suite: 295 passed, 38 deselected — no regressions vs the pre-task baseline (267 + exactly the 28 new).
- Note: the CPU suite count claim in the transcript's self-check (295/38) reproduces exactly on my run.

## 9. Claims audit (implementer report vs verified reality)

| Claim | Verified? |
|---|---|
| Verdict table complete, decided before install, all five candidates | Yes (transcript structure + timestamps) |
| resemblyzer license is Apache-2.0, brief's MIT corrected at execution | Yes (Section A PyPI classifier + GitHub API probe lines; correction stated in Section B) |
| pesq REJECTED, never installed, not in pyproject | Yes (pip show rc 1 in `.venv`; pyproject comment-only) |
| "IPR notice … captured verbatim into the transcript" (report) / "captured verbatim in Section A" (transcript) / "captured verbatim from the sdist" (README) | **NO — false.** Section A's verbatim-capture step failed at execution and printed nothing; the notice exists in the transcript only as quoted fragments inside the Section B summary. I verified the underlying facts myself from the still-on-disk sdist, but the committed evidence does not contain a verbatim capture. |
| Self-check "[PASS] ITU IPR notice captured verbatim for pesq REJECT" | Not substantiated — the needle evidently matched Section B's summary text, not a verbatim capture. |
| 28 new tests, CPU suite 295 passed / 38 deselected | Yes (reproduced exactly) |
| Evidence rows verbatim, no quality interpretation | Yes (rows pure; only STOI technical terminology in the scope note) |
| Performance/quality separation; benchmark.py untouched | Yes (0-line diff; JSON perf-grep empty) |
| Push FAILED, main ahead 2 | **NO — stale.** Push succeeded at 2026-09-21 18:24:13 +0800 (reflog "update by push" → c07055a); api.github.com confirms remote main = c07055a. In the user's favor. |
| README says transcript is "950 lines" | Minor inaccuracy — file is 988 lines. |

## 10. Problems found

1. **(Blocking for this verification) Verbatim IPR capture missing from the committed evidence.** `evidence/quality-benchmark-2026-09-21.txt` lines 153–155 show the capture probe failing ("file: " + "(no .c file at top level; listing tree)"); no corrected follow-up was appended. The criterion requires the IPR notice evidence captured verbatim; a fresh clone of the repository cannot see the notice verbatim (the sdist lives only in gitignored `.work-quality/`). The decision itself is sound (I verified the notice text independently), but the required evidence artifact is absent and is falsely claimed present in three places (transcript Section B line 210, `evidence/README.md` line 77, task report line 16), plus the unsubstantiated self-check PASS line. Remediation is small and precedented: append a corrected verbatim-capture block (the sdist is still on disk at `.work-quality/pesq-0.0.4/pesq/pesqmod.c`, lines 1–~60) to the transcript `tee -a`-style exactly as was done for smoke-1b and venv-import-verify-2, and fix the README claim (and its "950 lines" count).
2. (Minor, non-blocking) `meta.asr.device` records the policy expression `"cuda if torch.cuda.is_available() else cpu"` rather than the runtime-resolved device string; the resolved fact (tiny on cuda, gfx1151) is nonetheless documented via the gate probe and `meta.gpu`.
3. (Minor, non-blocking) Report claims push failed; the push actually succeeded (18:24:13 +0800 reflog + api.github.com). Stale report, not a code defect.
4. (Minor, non-blocking) `evidence/README.md` says the transcript has 950 lines; it has 988.

## 11. Why FAIL is justified

The instruction for this verification is explicit: PASS only when every
acceptance criterion is backed by executable or runtime evidence. Criterion 1
requires the pesq IPR notice "captured verbatim" — the committed transcript
does not contain it (its own capture step demonstrably failed and was never
corrected), so the criterion has no evidence in the repository, and worse, the
transcript, the evidence README, the implementer report, and the self-check all
assert the capture exists. That is precisely the class of claim-versus-artifact
gap an independent verifier exists to catch, in a program whose method is
verbatim machine capture. The failure is narrow: criteria 2–9 all pass on
independent runtime evidence I re-derived myself (test counts, package states,
pure-Python recomputation of the CER/WER arithmetic including two evidence-row
values, manifest/row cross-check, separation greps, zero-diff on
scripts/benchmark.py, and network+reflog proof the commit is pushed and the
tree clean), and the pesq REJECT decision itself is factually correct — I
verified the real IPR notice from the still-present sdist. One small
append-only transcript correction (plus README wording) converts this to PASS;
until that evidence exists in the repository, the verdict is FAIL.
