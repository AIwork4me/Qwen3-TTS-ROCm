# RC v0.2 Task 3 — Independent Release Verification

**Program:** Radeon Reference Closure v0.2
**Task:** 3 (Design decision + minimal fix on FIX branch)
**Verifier role:** independent falsifier — assume the implementer may be wrong
**Date:** 2026-09-21

---

## 1. Verdict

**PASS.**

## 2. Exact commit / working-tree SHA reviewed

- FIX branch commit reviewed: `0be0026ee99ae90173009e4a6540b336df360f22` (short `0be0026`), subject `fix(finetuning): make attention implementation configurable`, author/committer `amd <amd@localhost>` (one-shot `-c` identity, matches implementer disclosure).
- Parent: `022e286b98fbec7e1e916cb940cdf532cd9f488e` (pinned upstream main) — verified via `git log --format='%H %P' -1`.
- Baseline blob `022e286:finetuning/sft_12hz.py` sha256 `74d4359bff1ac5eddaca99f0cb5a8b76a55ffbd94af0ab50bffd49aee0e5c473` — **byte-identical to Task 0's recorded sha256** (ground-truth capture §8), tying the review to the verbatim upstream file.
- Patched blob `0be0026:finetuning/sft_12hz.py` = working-tree file (both sha256 `c7182a8dd062ebb1ad6c15764182c5671e08dfdfa499e55919aa156f0e0b3db3`) — no uncommitted edits.
- Diff index blob SHAs `c1f3f46..d4d63e4` verified against `git ls-tree` at both commits.

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Diff contains ONLY configurability change + minimal parser factoring; existing argument lines byte-unchanged as context; exactly one new `add_argument("--attn_implementation", ...)`; `def build_arg_parser():` + `return parser`; `train()` calls `build_arg_parser().parse_args()`; kwarg change | **PASS** | Reconstruction proof (below): applying exactly those changes to the verbatim baseline reproduces the patched file byte-for-byte. Committed diff `022e286..0be0026` is byte-identical to the review-package diff. All seven pre-existing `add_argument` lines appear as context lines (unchanged) in the diff hunks. |
| 2 | Default behavior unchanged (flash_attention_2) | **PASS** | Runtime: `parse_args(['--train_jsonl','/tmp/x.jsonl'])` → `default: flash_attention_2`; explicit `--attn_implementation flash_attention_2` → `flash_attention_2`. |
| 3 | Upstream style respected: no help strings, no reformatting, no whitespace churn, moved lines byte-identical | **PASS** | `cat -A` on baseline vs patched moved lines: byte-identical (`$` endings, no trailing spaces). Whole-file trailing-whitespace grep: zero matches. New argument line has no `help=`. Reconstruction proves zero untouched-line drift. |
| 4 | No prohibited change (AMD branches, `torch.version.hip`, changed global default, vendoring, architecture/optimizer/dataset/checkpoint logic, branding, new deps) | **PASS** | `grep -niE 'amd|hip|rocm|radeon|torch\.version|os\.environ|getenv|platform\.|sys\.platform'` on the patched file → 0 matches. `git diff --stat 022e286 0be0026` → 1 file, 10 insertions / 5 deletions (matches report). Lines 58–166 (dataset/optimizer/checkpoint logic) untouched per reconstruction. No dependency files touched. |
| 5 | Smoke check prints `sdpa` | **PASS** | Exact criterion-5 command from the worktree root, with dummy `--train_jsonl /tmp/x.jsonl`: printed `sdpa`, exit 0. |
| 6 | Worktree state: FIX clean, `0be0026` on `022e286`; downstream clean, HEAD `faf99b8` | **PASS** | FIX `status --porcelain` empty (checked before and after verification activity; `__pycache__/` is pre-existing gitignored, line 1 of the worktree `.gitignore`); `log --oneline -2` → `0be0026` / `022e286`; branch `fix/finetuning-attn-implementation`; exactly 1 commit in `022e286..0be0026`. Downstream `/home/amd/Desktop/Qwen3-TTS-ROCm`: clean, `main` @ `faf99b8dc6d041623aeafe7c61d4fe7483c096bf`. |

## 4. Files reviewed

- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-3-brief.md` (task brief)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-3-report.md` (implementer report)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-task3-fix-branch.diff` (review package)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/evidence/reference-closure-ground-truth-2026-09-21.md` (Task 0 verbatim capture, §8 lines 206–375, §9 lines 377–387)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py` (patched file, 166 lines, read in full to line 80; remainder covered by reconstruction equality)
- Git objects: `022e286:finetuning/sft_12hz.py`, `0be0026` commit + tree

## 5. Exact commands executed

All by the verifier, all from absolute paths (cwd reset between calls; smoke checks prefixed with `cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix`):

1. `git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix status --porcelain` (run twice: pre- and post-activity)
2. `git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix log --oneline -3`
3. `git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix branch --show-current`
4. `git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix log --format='%H %P' -1`
5. `git -C /home/amd/Desktop/Qwen3-TTS-ROCm status --porcelain` + `rev-parse HEAD` + `log --oneline -1` + `branch --show-current`
6. `git -C ...fix diff 022e286 0be0026 > /tmp/actual-commit.diff` then `diff /tmp/actual-commit.diff <review-package.diff>`
7. `git -C ...fix show 0be0026 --format='...' --no-patch`
8. `git -C ...fix show 022e286:finetuning/sft_12hz.py | sha256sum` and `git ls-tree` at both commits
9. `git -C ...fix show 0be0026:finetuning/sft_12hz.py | sha256sum` vs `sha256sum` of the working-tree file
10. `grep -n 'attn_implementation' ...fix/finetuning/sft_12hz.py` (+ `-c` count)
11. `grep -niE 'amd|hip|rocm|radeon|torch\.version|os\.environ|getenv|platform\.|sys\.platform'` on the patched file
12. `grep -nP ' +$|\t+$'` (trailing-whitespace audit) on the patched file
13. Python reconstruction script (see §7) comparing baseline+described-changes vs actual file
14. **Criterion-5 smoke check (exact command from the acceptance criteria):** `cd ...fix && /home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python -c "import sys; sys.path.insert(0,'finetuning'); import sft_12hz; a=sft_12hz.build_arg_parser().parse_args(['--attn_implementation','sdpa','--train_jsonl','/tmp/x.jsonl']); print(a.attn_implementation)"` → **`sdpa`**
15. Default-behavior check: `parse_args(['--train_jsonl','/tmp/x.jsonl'])` and explicit `flash_attention_2` variant → both `flash_attention_2`
16. Literal brief command (no `--train_jsonl`) → exit 2, error `the following arguments are required: --train_jsonl` (verifies the implementer's disclosure)
17. `cat -A` comparison of moved lines (baseline `sed -n '31,33p;35,41p'` vs patched `sed -n '44,46p;32,39p'`)
18. `/home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python -m py_compile finetuning/sft_12hz.py`
19. `git -C ...fix rev-list --count 022e286..0be0026` and `git show --stat 0be0026`

## 6. Exit codes

| Command | Exit |
|---|---|
| FIX worktree `status --porcelain` (both runs) | 0, empty output |
| Downstream `status --porcelain` | 0, empty output |
| `diff /tmp/actual-commit.diff review-package.diff` | 0 (byte-identical) |
| `grep -n 'attn_implementation'` | 0 (2 matches) |
| Prohibited-pattern grep | 1 (no matches — desired) |
| Trailing-whitespace grep | 1 (no matches — desired) |
| Reconstruction script | clean run, prints byte-identical |
| Criterion-5 smoke check | **0**, stdout `sdpa` |
| Default check | 0, `flash_attention_2` |
| Literal brief command (no dummy arg) | 2, `--train_jsonl` required |
| `py_compile` | 0 |

## 7. Runtime evidence inspected

- **Smoke check stdout:** `sdpa` (preceded by pre-existing environmental noise: SoX-not-found and flash-attn-not-installed warnings from the module import chain — same noise the Task 0/2 environment shows; non-fatal, unrelated to this diff).
- **Default check stdout:** `default: flash_attention_2` and `explicit-fa2: flash_attention_2`.
- **Literal brief command stderr (tail):** `-c: error: the following arguments are required: --train_jsonl` — confirms the plan defect is the pre-existing upstream `required=True` on `--train_jsonl` (present verbatim in the Task 0 capture, line 37), not anything this patch introduced.
- **Reconstruction proof (strongest falsification attempt):** extracted the baseline from git (`sha256 74d4359b…` == Task 0 capture's recorded sha256), asserted the exact baseline lines at every touched position (old lines 31–33, 34, 41–42, 51 match the ground-truth capture verbatim), then applied only the three described changes — (a) `def train():`/`global`/blank → `def build_arg_parser():`, (b) `args = parser.parse_args()` → new-argument + `return parser` + two blanks + re-homed `def train():`/`global`/blank + `args = build_arg_parser().parse_args()`, (c) `attn_implementation="flash_attention_2",` → `attn_implementation=args.attn_implementation,` — and compared the result to the actual patched file: **byte-identical**. This mechanically excludes any other edit, whitespace churn, reformatting, or subtle change anywhere in the 166-line file, and proves the moved lines are byte-identical to the Task 0 capture.
- **`cat -A` moved-line comparison:** baseline `def train():$` / `    global target_speaker_embedding$` / `$` and all seven `parser.add_argument(...)$` lines are byte-equal (including indentation and no trailing spaces) to their patched-file counterparts.
- **`grep -c 'attn_implementation'` = 2**, at lines 40 (definition, default `flash_attention_2`) and 56 (kwarg use) — exactly the two expected occurrences.

## 8. Regression tests

No test suite exists for this change yet (Task 4 builds the tests on top of `build_arg_parser()`). Executed in its place:
- Import + parser construction + parse without executing training (smoke checks above).
- `python -m py_compile` — passes.
- Default-value equivalence check (default == previously hard-coded string) — passes.
- Byte-level no-regression proof vs the Task 0 verbatim capture (reconstruction equality) — passes; this covers lines 58–166 (model loading body, dataset, optimizer, training loop, checkpoint logic) as untouched without needing to eyeball them.

## 9. Claims audit

| Implementer claim | Verifier finding |
|---|---|
| Commit `0be0026`, parent `022e286`, 1 file, +10/−5 | Confirmed exactly (`ls-tree`, `show --stat`, `rev-list --count` = 1). |
| Baseline matched Task 0 capture (parser inline in `train()`, FA2 hard-code at line 51) | Confirmed; baseline sha256 equals the capture's recorded sha256; hard-code at old line 51 matches capture §9. |
| Review-package diff == committed diff | Confirmed byte-for-byte via `diff`. |
| Literal brief command fails exit 2 on `--train_jsonl` required | Reproduced: exit 2, argparse error names only `--train_jsonl`; that argument is `required=True` in the verbatim upstream capture, so the failure is a pre-existing plan defect, correctly disclosed. Criterion-5's corrected command (with dummy value, per the disclosed ruling) passes. |
| Default remains `flash_attention_2` | Confirmed at runtime. |
| No whitespace churn / moved lines unchanged / no help strings | Confirmed via reconstruction equality + `cat -A` + whole-file trailing-whitespace grep (zero). |
| Only `finetuning/sft_12hz.py` changed; post-commit status clean | Confirmed (`--stat`, `status --porcelain` empty). |
| Downstream main untouched at `faf99b8` | Confirmed (clean, `main` @ `faf99b8dc6d041623aeafe7c61d4fe7483c096bf`). |
| Author identity supplied via one-shot `-c` flags; no config modified | Consistent with observation; `git config --list` not re-audited (non-substantive for acceptance criteria). |

## 10. Problems found

None blocking. Two non-blocking observations:
1. **Plan defect (already disclosed and ruled on):** the brief's literal Step 3.3 command cannot pass against any parser retaining upstream's `--train_jsonl required=True`. The implementer disclosed this honestly; the failure reproduces identically against the unmodified upstream parser, so it is not a defect of this change. The verifier's criterion-5 command (with dummy `--train_jsonl`) passes and prints `sdpa`.
2. **Environmental noise only:** SoX and flash-attn warnings appear on import of `sft_12hz` — pre-existing environment properties (flash-attn absent is in fact the very condition of issue #372), unrelated to the diff.

## 11. Why PASS is justified

Every acceptance criterion is backed by executable or runtime evidence, not by trusting the implementer: the committed diff was independently re-derived from git and shown byte-identical to the review package; the patched file was proven (by programmatic reconstruction from the sha256-matched verbatim baseline) to contain exactly the three described changes and nothing else, which mechanically establishes criteria 1, 3, and 4 — including that the moved parser lines are byte-identical to the Task 0 capture and that the default remains the previously hard-coded string; the default was additionally confirmed at runtime (criterion 2); the exact criterion-5 smoke command was executed by the verifier and printed `sdpa` with exit 0 (criterion 5); and both worktrees' git state was checked directly and matches the required topology — FIX clean at `0be0026` on `022e286`, downstream clean at `faf99b8` (criterion 6). The only deviation from the brief's literal smoke command is the pre-disclosed dummy `--train_jsonl`, whose necessity I reproduced and whose cause is upstream's pre-existing `required=True`, not this patch. No falsification attempt succeeded.
