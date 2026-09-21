# Task 7 independent verification — Rounds D+E (default semantics + diff audit)

Program: Radeon Reference Closure v0.2. Independent release verifier run 2026-09-21.
Mandate: falsify the claimed result; every acceptance criterion backed by executable or runtime evidence. No subagents dispatched; nothing modified except this report file.

## 1. Verdict

**PASS** — all 8 acceptance criteria independently reproduced or byte-verified. One minor, non-blocking prose inaccuracy found in the implementer report (Section 10, P1).

## 2. Exact commit / working-tree SHA reviewed

- Downstream `main` @ `bee03ab14169021c62e28118372cd025e07a9d04` (= review range head 1925540..bee03ab; single commit, 3 files, 498 insertions). Working tree clean (`status --porcelain` = 0 lines), so committed blobs == on-disk evidence files.
- FIX worktree `.upstream/Qwen3-TTS-fix`, branch `fix/finetuning-attn-implementation` @ `48b8644aac8512e6fdc0f4442788bf399c425fdb`, pristine (`status --porcelain` = 0 lines).
- Pinned upstream main `022e286b98fbec7e1e916cb940cdf532cd9f488e`; `merge-base(022e286, HEAD) == 022e286` (branch is a linear descendant: 022e286 → 0be0026 → 48b8644).

## 3. Acceptance criteria checklist

| # | Criterion | Result | Independent evidence |
|---|---|---|---|
| 1 | Round D unit leg: 5/5 OK with command+exit visible; default-value tests assert flash_attention_2 | **PASS** | Transcript lines 45–71: verbatim command `(cd .upstream/Qwen3-TTS-fix && .venv/bin/python -m unittest tests.test_sft_attn_implementation -v)`, `Ran 5 tests ... OK`, `unittest-exit=0`. I read `tests/test_sft_attn_implementation.py` myself: line 40 `assertEqual(_parse([]).attn_implementation, "flash_attention_2")`; lines 67–82 `test_default_value_reaches_model_loader` runs full `train()` with mocked loader and asserts `captured["attn_implementation"] == "flash_attention_2"`. I also re-ran the suite myself: OK, exit 0 |
| 2 | Round D runtime leg: default-flag load transcript with FlashAttention ImportError + non-zero exit, matching pristine failure class | **PASS** | Transcript lines 98–150: parse with NO `--attn_implementation` (verified by reading `.work-finetune/round_d_default_load.py`, sha256 `fbebafd2…` matches transcript pin), prints `parsed attn_implementation='flash_attention_2' (parser DEFAULT -- no flag given)`, real `Qwen3TTSModel.from_pretrained`, ImportError traceback, `python-exit=1`. I re-derived `diff <(grep -m1 "^ImportError: FlashAttention2" evidence/upstream-372-pristine-failure-run1.txt) <(grep -m1 ... round_d_default_load_stdout.txt)` → byte-identical; both tracebacks terminate at `transformers/modeling_utils.py:2422 _flash_attn_2_can_dispatch`; pristine run1 records `sft-as-is-exit=1` (same class, same exit) |
| 3 | Verbatim statement recorded; NOWHERE claims CUDA validation | **PASS** | Statement verbatim at evidence line 189 and in the README row. CUDA scan over both Task-7 evidence files + README: only hits are `cuda_available=True` (an environment probe of `torch.cuda.is_available()` on the HIP stack — a fact, not a validation claim) and the statement itself; README row states "CUDA runtime explicitly NOT claimed" |
| 4 | Round E: per-hunk verdict table + full diff; re-derived diff byte-compares; 4 hunks / 2 files; every changed line categorized | **PASS** | My own `git -C .upstream/Qwen3-TTS-fix diff 022e286..HEAD` (151 lines, 4 `@@` hunks, 2 `diff --git` files) `cmp`-byte-identical to BOTH the audit doc's appended diff (extracted from `evidence/upstream-372-diff-audit.txt`) and `.work-finetune/fix-branch.diff` (sha256 `274761d4…` matches the audit's pinned hash). Verdict table H1–H4 present. All 5 deletions + 115 insertions accounted for and each verified against the diff text I read in full: 3 verbatim moves (`def train():` / `    global target_speaker_embedding` / blank), 1 replaced-equivalent parse, 1 call-site kwarg, 1 new `add_argument(..., default="flash_attention_2")`, `def build_arg_parser():` / `return parser` / 2 blanks, 105 test lines |
| 5 | Zero prohibited patterns; moved lines byte-identical to 022e286 | **PASS** | My own scan over changed lines (audit's pattern set): 0 hits. Wider scan of whole diff for `hip\|rocm\|amd\|radeon\|gfx\|cuda\|platform\|sys\.platform\|os\.name\|uname`: 0 hits. `diff -w --stat` == `diff --stat` (zero whitespace-only churn). Moved lines byte-identical: `grep -x` exact matches at 022e286 lines 31–33 == HEAD lines 44–46; `cat -A` on 022e286 shows no hidden whitespace. `--name-only` = exactly the 2 files (no vendoring, no dependency files); added imports are stdlib only (`sys`, `unittest`, `pathlib.Path`, `unittest.mock`) |
| 6 | Disclosed driver re-run: archived transcript a single clean run; no residue | **PASS** (with P1 note) | Transcript internally consistent as one run: exactly 1 task header (15:06:32), 1 COMPONENT A, 1 COMPONENT B, 1 closeout/date-end (15:06:43), 1 `ROUND_D_RESULT`; the versions probe in-transcript is the working `importlib.metadata` form. Residue check: `evidence/` dir listing shows no discarded/orphan captures; files modified in the 15:04–15:10 window under `.work-finetune/` are exactly the in-transcript-referenced working files (scratch stdout overwritten by the clean run at 15:06:41) |
| 7 | Worktree states: FIX pristine @ 48b8644; pristine clone clean; downstream @ bee03ab clean | **PASS** | All three verified by my own `rev-parse` + `status --porcelain | wc -l` = 0 for each (pristine clone `.upstream/Qwen3-TTS` on `main` @ `022e286`, clean) |
| 8 | CPU suite green (267 passed, 38 deselected) | **PASS** | I ran `.venv/bin/python -m pytest -m 'not gpu' -q` myself: `267 passed, 38 deselected in 16.04s`, exit 0 — exactly the expected counts |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-7-brief.md`
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-7-report.md`
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-1925540..bee03ab.diff`
- `evidence/upstream-372-default-semantics.txt` (sha256 `0e6c1723525f34e57001ca69a252b4d3ed985d0c504a4bdac582e07750d47107`, matches report claim `0e6c1723…`)
- `evidence/upstream-372-diff-audit.txt` (sha256 `a54aa8ac1e489bc3a2c97087fe030c8241280689bd1c5608b250dd22a0753d0d`)
- `evidence/upstream-372-pristine-failure-run1.txt` (failure class + exit)
- `evidence/README.md` (Task 7 rows)
- `.upstream/Qwen3-TTS-fix/tests/test_sft_attn_implementation.py` (read in full, 105 lines)
- `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py` (read head; compared region vs `git show 022e286:...`)
- `.work-finetune/round_d_driver.sh`, `.work-finetune/round_d_default_load.py`, `.work-finetune/round_d_default_load_stdout.txt`, `.work-finetune/fix-branch.diff`

## 5. Exact commands executed (verifier's own)

1. `git -C .upstream/Qwen3-TTS-fix diff 022e286..HEAD > /tmp/verifier-fix-diff.txt` (+ `wc -l`, `grep -c "^@@"`, `grep "^diff --git" | wc -l`)
2. `git rev-parse --abbrev-ref HEAD; git rev-parse HEAD; git status --porcelain | wc -l; git log --oneline -3` (downstream; same trio for `.upstream/Qwen3-TTS-fix` and `.upstream/Qwen3-TTS`)
3. `sha256sum .work-finetune/fix-branch.diff` ; `cmp .work-finetune/fix-branch.diff /tmp/verifier-fix-diff.txt`
4. `awk`-extract of the audit doc's appended diff → `cmp` vs re-derived diff
5. `grep -E "^(\+|-)" /tmp/verifier-fix-diff.txt | grep -v -E "^(\+\+\+|---)" | grep -c -i -E "torch\.version\.hip|version\.hip|\brocm\b|\bamd\b|is_hip|device_map|CUDA_VISIBLE|qwen3-tts-rocm|radeon"` → 0
6. `grep -n -i -E "hip|rocm|amd|radeon|gfx|cuda|platform|sys\.platform|os\.name|uname" /tmp/verifier-fix-diff.txt` → 0 hits
7. `git -C .upstream/Qwen3-TTS-fix diff -w 022e286..HEAD --stat` ; `diff <(git diff 022e286..HEAD) <(git diff 022e286...HEAD)` → identical
8. `git -C .upstream/Qwen3-TTS-fix show 022e286:finetuning/sft_12hz.py | sed -n '25,60p' | cat -A` ; `grep -n -x` exact-line matching of the moved lines in both versions
9. `grep -n -m1 "^ImportError: FlashAttention2" evidence/upstream-372-pristine-failure-run1.txt` ; re-derived class-match `diff` vs `.work-finetune/round_d_default_load_stdout.txt`
10. `grep -n -i "cuda"` over both Task-7 evidence files + `evidence/README.md`
11. `ls -la --time-style=full-iso` of `evidence/` and `.work-finetune/`; `find .work-finetune -maxdepth 1 -newermt "2026-09-21 15:04" ! -newermt "2026-09-21 15:10"` (residue window)
12. `grep -c` single-run-consistency counters on the Round D transcript
13. `sha256sum .work-finetune/round_d_default_load.py` (= `fbebafd2…`, matches transcript pin)
14. `git diff 1925540..bee03ab` vs the review package body (see Section 9)
15. `git -C .upstream/Qwen3-TTS-fix log --oneline --graph 022e286..HEAD` ; `merge-base 022e286 HEAD`
16. Independent re-run: `(cd .upstream/Qwen3-TTS-fix && .venv/bin/python -m unittest tests.test_sft_attn_implementation -v)`
17. Independent re-run: `.venv/bin/python -m pytest -m 'not gpu' -q`

## 6. Exit codes (verifier-observed)

- Re-derived FIX diff: 0. Unittest re-run (mine): 0 (`OK`). pytest CPU suite (mine): 0 (267 passed, 38 deselected). All `cmp`/`diff` byte-compares: 0 (identical). Prohibited-pattern greps: exit 1 = zero matches. `git status --porcelain | wc -l` = 0 for all three trees. Evidence-recorded exits: `unittest-exit=0`, `python-exit=1` (expected ROCm failure), pristine run1 `sft-as-is-exit=1`.

## 7. Runtime evidence inspected

- Round D transcript (`evidence/upstream-372-default-semantics.txt`): environment header (python 3.12.3, torch 2.12.0+rocm7.14.0 / HIP 7.14.60850, transformers 4.57.3, qwen-tts 0.1.1, flash_attn=ABSENT probe, Radeon 8060S gfx1151, both repo SHAs, FIX pristine); unit suite 5/5 OK exit 0; rocm-smi snapshots wrapping component B; default-flag load traceback ending in the FlashAttention ImportError, `python-exit=1`; in-transcript class-match `CLASS-MATCH-VERBATIM`; closeout criteria + verbatim statement.
- Round E audit (`evidence/upstream-372-diff-audit.txt`): pinned SHAs, saved-diff sha256, machine checks [A]–[I], per-hunk verdict table H1–H4, changed-line accounting, prohibited-change checklist, appended diff.
- Pristine failure run1 (`evidence/upstream-372-pristine-failure-run1.txt`): same failure point (`modeling_utils.py:2422`), same ImportError line, `sft-as-is-exit=1`.
- Driver sources read: `round_d_driver.sh` (structure matches transcript exactly; exit codes taken as the process's own via `$?`/`PIPESTATUS[0]`) and `round_d_default_load.py` (parse list contains NO `--attn_implementation`; asserts default is flash_attention_2; real, uncaught `from_pretrained`).

## 8. Regression tests

- Upstream FIX-branch unittest suite re-run by verifier: 5 tests, OK, exit 0. The two default-value tests exist and genuinely assert flash_attention_2 (default parse; default reaching `from_pretrained` through a full `train()` with intercepted loader).
- Downstream CPU suite re-run by verifier: 267 passed, 38 deselected, exit 0 — no regressions from commit bee03ab.
- Runtime regression anchor: default-path failure on this flash-attn-less host is byte-identical (ImportError line) to the pinned pristine-upstream capture, i.e. the fix did not alter default-path behavior on ROCm.

## 9. Claims audit (report vs evidence vs my own execution)

- "5/5 OK, exit 0, both default tests green" — TRUE (transcript + my re-run + my reading of the test source).
- "Default-flag load fails with python-exit=1, CLASS-MATCH-VERBATIM" — TRUE (re-derived myself; traceback endpoint and ImportError byte-identical to pristine run1; pristine exit also 1).
- "Verbatim statement recorded; CUDA never claimed" — TRUE (grep-audited).
- "Diff: 151 lines, 4 hunks, 2 files, 115+/5−; appendix byte-identical" — TRUE (independently re-derived and `cmp`-verified; sha256 matches).
- "Two-dot ≡ three-dot (linear descendant)" — TRUE (`merge-base` = 022e286; diffs identical under my own comparison).
- "Zero prohibited patterns / zero whitespace churn / stdlib-only additions" — TRUE (my own greps and `-w --stat` comparison).
- "FIX pristine before/after, pristine clone clean, downstream clean" — TRUE (current state verified clean at the pinned SHAs).
- "Report sha256 claims (`0e6c1723…`, `fbebafd2…`, `274761d4…`)" — all three verified exact.
- Review package `review-1925540..bee03ab.diff`: its diff body differs from my `git diff 1925540..bee03ab` ONLY in the README hunk's context width (review shows `@@ -62,20 +62,22 @@` with extra unchanged table rows; git default shows `@@ -69,6 +69,8 @@`); same 2-line insertion, and the two evidence-file sections are byte-identical. Benign generation-context difference, committed content unaffected (working tree clean at bee03ab).
- Report claim "run once cleanly, 15:05:58→15:06:43" / "the single clean 15:05:58 run" — see P1 below.

## 10. Problems found

- **P1 (minor, non-blocking, prose only):** the implementer report describes the archived Round D transcript as "the single clean 15:05:58 run (15:05:58→15:06:43)". The archived transcript's own first timestamp is `2026-09-21T15:06:32+08:00` (closeout 15:06:43). The 15:05:58 time corresponds to the discarded defective attempt (driver script last modified 15:06:30 when the versions probe was replaced; `round_d_default_load.py` created 15:05:11). The archived evidence itself is internally consistent as one clean run and the disclosure is otherwise accurate; only the report's start-timestamp attribution is off by ~34 s. No effect on any acceptance criterion.
- No other problems. No hand-edit artifacts detected; hashes, mtimes, transcript structure, and git states are mutually consistent.

## 11. Why PASS is justified

Every acceptance criterion was backed by evidence I reproduced independently rather than accepted: the diff-audit's central artifact (the FIX diff) was re-derived from git and byte-compared identical in three places (git, saved file, audit appendix); hunk/file/line accounting was recomputed and each changed line re-read against the 022e286 original (verbatim moves confirmed by exact-line matching); prohibited-pattern and whitespace-churn scans were re-run with the audit's pattern set plus a wider net (0 hits both); the failure-class match was re-derived against the pristine capture (byte-identical, same traceback endpoint, both exit 1); the unit suite and the CPU suite were re-executed by me (5/5 OK exit 0; 267 passed / 38 deselected exit 0); all three worktrees were checked at their pinned SHAs and clean; the verbatim no-CUDA-claim statement is recorded and a full-text scan found no CUDA-validation claim; and the disclosed re-run note checks out with a single-run-consistent transcript and no on-disk residue. The only defect found is a 34-second start-timestamp mislabeling inside the implementer's prose report (P1), which does not touch the evidence, the commit, or any criterion.
