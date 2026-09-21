# Task 10 independent verification — downstream fine-tuning claim cleanup

Program: Radeon Reference Closure v0.2. Verifier role: independent release
verifier; implementer's claims assumed possibly wrong; every criterion
attacked with executable or runtime evidence re-derived this session. No
subagents dispatched; nothing modified except this report file.

## 1. Verdict

**PASS.**

## 2. Exact commit / working-tree SHA reviewed

- Reviewed commit: `6f509aa5f41988ce0698add7ff05eb2242834dab` (`docs: align
  fine-tuning claims with upstream PR status (#372)`, author amd, 2026-09-21
  15:59:05 +0800) — equals working-tree HEAD; `git status --porcelain` empty
  (clean) before and after verification.
- Diff range re-derived myself: `git diff 008ca7c..6f509aa --stat` = 5 files,
  +81/−7 (CHANGELOG.md 14+, README.md 2×2~ changed lines, README_CN.md,
  docs/finetuning-rocm.md +64/−2, evidence/README.md 1 row) — matches the
  review package byte-for-byte in structure and stat.

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Fixed wording substance wherever fine-tuning upstream state is summarized (EN+CN) | PASS | 5 placements verified (§8 below); mandated sentence matches verbatim modulo hard line-wrap and the intended `#XXX`→`#373` substitution |
| 2 | No forbidden claim anywhere; every "merge" occurrence a disclaimer | PASS | Repo-wide greps (§8); all 9 merge-mentions in changed files are disclaimers; only forbidden-phrase hits are the rulebook quoting its own prohibition (untouched by diff) |
| 3 | PR facts correct vs live state | PASS | `gh pr view 373` → OPEN; title/URL match docs exactly (§7) |
| 4 | Stale "human-owner action" line corrected | PASS | diff + current `docs/finetuning-rocm.md:297-300` |
| 5 | PROVEN/NOT-PROVEN + disclosed-workaround history intact | PASS | byte-identical section diff old vs new (§8) |
| 6 | All newly-linked evidence files exist | PASS | 17/17 paths exist (§8) |
| 7 | CN/EN same substance | PASS | spot-translation of both row pairs (§8) |
| 8 | 6f509aa pushed to origin/main; tree clean; CPU suite green | PASS | remote `refs/heads/main` = `6f509aa…`; suite `267 passed, 38 deselected`, exit 0 (§7) |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-10-brief.md`
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-10-report.md`
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-008ca7c..6f509aa.diff`
- `docs/finetuning-rocm.md` (full, current HEAD)
- `README.md` (both fine-tuning rows + repo-wide greps)
- `README_CN.md` (both rows + repo-wide greps)
- `CHANGELOG.md` ([Unreleased] Verified + Changed sections)
- `evidence/README.md` (changed row)
- Secondary (claim-context checks): `docs/p0-parity-report.md`,
  `docs/development/2026-08-27-{design-spec,final-acceptance,implementation-plan}.md`,
  `evidence/upstream-372-root-cause.md` (PR appendix),
  `docs/superpowers/reports/rc02-task-8-upstream-372-verdict.md`,
  `docs/superpowers/specs|plans/2026-09-21-radeon-reference-closure-v0.2*.md`
  (rulebook only, to classify forbidden-phrase hits).

## 5. Exact commands executed

1. `git rev-parse HEAD` / `git status --porcelain` / `git log --oneline -5` / `git branch -vv`
2. `git rev-parse origin/main`
3. `git ls-remote origin main` (retry loop ×3 after GnuTLS flake)
4. `gh pr view 373 --repo QwenLM/Qwen3-TTS --json number,state,url,title` (retry loop ×3)
5. `git diff 008ca7c..6f509aa --stat`; `git show 6f509aa --format=… --no-patch`
6. `grep -rn "zero-patch fine-tuning fully closed\|fully closed" README.md README_CN.md CHANGELOG.md docs/ evidence/README.md`
7. `grep -rni "fixed upstream\|fixes upstream\|upstream fixed" …` (same scope)
8. `grep -rn -i "merge" README.md README_CN.md CHANGELOG.md docs/finetuning-rocm.md evidence/README.md` (and repo-docs variant)
9. `grep -rn "only path\|唯一的路径\|唯一路径" …`; `grep -n "合并" README_CN.md CHANGELOG.md docs/finetuning-rocm.md`
10. `grep -rn "runs unmodified\|run unmodified\|merged upstream\|merged and\|零补丁" …`
11. Existence loop over all 17 linked evidence/report paths (`test -f`)
12. `git show 008ca7c:docs/finetuning-rocm.md` piped through `sed`+`diff` vs current file for the PROVEN→Limitations and workaround sections (byte-compare)
13. Mandated-wording comparison: design-spec line 84 vs `docs/finetuning-rocm.md:103-105`, newline-joined and whitespace-normalized
14. `.venv/bin/python -m pytest -m 'not gpu' -q` (run twice)
15. `sed -n`/`grep` context reads of CHANGELOG.md:330-348, p0-parity-report, root-cause appendix, task-8 verdict

## 6. Exit codes

- (1)(2)(5)(11)(12)(13)(15): 0 (12: diff reported no differences = identical)
- (3): 1st attempt failed `GnuTLS recv error (-110)` (known host network flake,
  same class the implementer documented); 2nd attempt exit 0
- (4): 0 (succeeded first attempt)
- (6)(7)(10): grep found matches only in rulebook/historical files (audited,
  benign); "only path" grep exit 1 (no hits)
- (14): exit 0, both runs — `267 passed, 38 deselected in 16.11s` and
  `267 passed, 38 deselected in 15.95s`

## 7. Runtime evidence inspected

- **Live PR state (criterion 3):** `gh pr view 373 --repo QwenLM/Qwen3-TTS`
  → `{"number":373,"state":"OPEN","title":"fix(finetuning): make attention
  implementation configurable","url":"https://github.com/QwenLM/Qwen3-TTS/pull/373"}`.
  Docs claim OPEN, not merged, and quote the title identically — match.
- **Remote push (criterion 8):** `git ls-remote origin main` →
  `6f509aa5f41988ce0698add7ff05eb2242834dab refs/heads/main` — the remote
  actually holds the reviewed commit; local `origin/main` ref equals HEAD.
- **Test suite (criterion 8):** `.venv/bin/python -m pytest -m 'not gpu' -q`
  → `267 passed, 38 deselected` — exactly the expected count, exit 0.
- **Task-8 verdict artifact:** `rc02-task-8-upstream-372-verdict.md` §1 reads
  "**PASS** …" — the "independent chain-verifier PASS" claim in README rows
  points at a real PASS verdict.
- **PR appendix reality:** `evidence/upstream-372-root-cause.md:434` has
  "## Appendix — Upstream PR (2026-09-21)" with the verbatim `gh pr view`
  record (`state: OPEN`) — the appended clause in `evidence/README.md`'s
  changed row describes something that exists.

## 8. Claims audit (criterion-by-criterion attack)

**Criterion 1 — fixed wording placements (5 found, all verified):**
1. `docs/finetuning-rocm.md:103-105` — bold, verbatim after newline-joining;
   vs the design-spec template the only deltas are `#XXX`→`#373` (mandated
   substitution) and dropped backticks around `qwen-tts==0.1.1` (renders the
   same; the acceptance wording quotes it without backticks).
2. `README.md:67` capability row — same substance + "(OPEN)" + anchor
   `docs/finetuning-rocm.md#upstream-fix-status` (heading
   `## Upstream fix status` exists → anchor valid).
3. `README.md:324` comparison row — same substance + validation-chain summary
   + "Until it merges, … still requires the documented temporary workaround".
4. `README_CN.md:59` — 「ROCm 端到端执行已验证。上游可移植性修复已作为
   Qwen3-TTS PR #373 提交（OPEN）；当前已发布的 `qwen-tts==0.1.1` 仍需按文档
   使用临时变通方案」+ same anchor + same 8 evidence links.
5. `README_CN.md:292` + `CHANGELOG.md:220-229` (Verified entry) and
   `CHANGELOG.md:238-251` (new Changed bullet) — same three clauses.
Other fine-tuning mentions in the READMEs (EN:62/CN:54 "Base family …
fine-tuning base" inference row; EN:75/CN:66 concept sentence) carry no
upstream-state summary. `docs/p0-parity-report.md` is explicitly date-pinned
("As-of 2026-09-21", "authoritative current capability state is README.md")
and makes only smoke-era execution claims — no upstream-PR statement, so
nothing stale or forbidden there.

**Criterion 2 — forbidden claims:** "zero-patch fine-tuning fully closed" /
"fully closed": single hit at
`docs/superpowers/specs/…-design.md:84` — the rulebook *defining* the
prohibition (quoted phrase), untouched by this diff. "fixed upstream":
hits only in `docs/superpowers/specs|plans` prohibition statements, likewise
untouched (diff touches exactly the 5 brief-listed files). All 9
merge/merged mentions in the five changed files are disclaimers:
README.md:324 "Until it merges"; CHANGELOG.md:226 "when merged", :227 "until
a merged fix ships", :250 "merged-upstream path is explicitly not claimed";
finetuning-rocm.md:109 "OPEN, not merged", :112 "when merged", :113 "Until a
merged fix ships", :149-150 "No claim … depends on … being merged, and the
merged-upstream path has NOT been run"; :299 "until it merges".
CN: README_CN.md:292 「在合并之前…仍需…临时变通」(disclaimer); :143 「绝不合并」
is benchmark-stage wording, unrelated. "runs unmodified" hits
(CHANGELOG.md:342, dev-plan:797) are the v0.1.0 *inference-demo thin-shim*
claim, not fine-tuning/merged-upstream. No statement anywhere that merged
upstream runs unmodified; finetuning-rocm.md explicitly denies it.

**Criterion 4:** diff replaces "Upstream issue filing remains a human-owner
action." with the #372-filed / #373-submitted correction + link; current
file confirms.

**Criterion 5:** `sed`-extracted "## PROVEN … ## Limitations" and the
"⚠️ Disclosed upstream-defect workaround" section are byte-identical between
`008ca7c` and the working tree (diff empty). Old "workaround is the only
path" framing: grep "only path/唯一路径" → 0 hits. Nothing strengthened,
nothing deleted.

**Criterion 6:** all 17 linked paths exist on disk (pristine-failure
run1/run2, root-cause.md, round-a-loads1-3, e2e-run1/2 .txt+.json,
default-semantics, diff-audit, task-8 verdict report, upstream-issue capture,
finetune-smoke txt/json, evidence/README.md).

**Criterion 7 (spot translation):** EN "execution-only smoke … no quality
claims" ↔ CN 「仅执行冒烟验证…无质量结论」; EN three fixed clauses ↔ CN
「端到端执行已验证/已作为 PR #373 提交（OPEN）/仍需按文档使用临时变通方案」;
comparison-row validation chain items map 1:1 (原始失败双重复现 → pristine
double reproduction; 最小修复受控隔离 → minimal-fix controlled isolation;
3 次针对性加载验证 → 3× targeted loader validations; 修复分支两次独立 E2E →
two independent E2E runs from the fix branch; 默认 flash_attention_2 语义保持 →
preserved default semantics; 独立链路校验 PASS → independent chain-verifier
PASS); same 8 evidence links and same gitignored-clone workaround disclosure
in both languages.

## 9. Regression tests

`.venv/bin/python -m pytest -m 'not gpu' -q` run twice at HEAD `6f509aa`:
`267 passed, 38 deselected` both times, exit 0 — matches the expected
267/38 exactly; no failures, no skips beyond the marker deselection.

## 10. Problems found

None blocking. Three non-blocking observations:

1. The bold fixed-wording sentence in `docs/finetuning-rocm.md:103-105`
   wraps across three source lines and omits the backticks the design-spec
   template places around `qwen-tts==0.1.1`. Rendered output is identical to
   the mandated sentence with `#373` substituted; the acceptance wording
   quotes it without backticks. Cosmetic only.
2. First `git ls-remote` attempt failed with the host's known GnuTLS -110
   flake before the retry succeeded — environment, not repository.
3. Forbidden-phrase strings do appear verbatim inside
   `docs/superpowers/specs/` and `docs/superpowers/plans/` — as quoted
   *prohibitions* in the program rulebook, pre-existing and untouched by
   commit `6f509aa`. Not claims; no action needed.

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence I re-derived or re-executed
myself rather than trusting the implementer: the live GitHub PR state was
fetched (`gh pr view 373` → OPEN, matching every doc statement), the remote
branch was proven to hold the exact reviewed SHA (`ls-remote` → `6f509aa…`),
the CPU suite was run twice to the exact expected count (267 passed / 38
deselected / exit 0), the review package's diff stat was reproduced from real
git history, the PROVEN/NOT-PROVEN and disclosed-workaround sections were
byte-compared as unchanged, every newly-linked evidence artifact was proven
to exist (and the two claims-about-evidence — the PR appendix and the
chain-verifier PASS — were opened and confirmed), the mandated wording was
string-compared (verbatim modulo the intended PR-number substitution and line
wrap), and the CN text was independently translated back and shown to carry
the identical substance, scope guards, and links. The claims audit found no
statement stronger than "submitted" anywhere in the repo: all nine
merge-mentions in the changed files are disclaimers, the merged-upstream path
is explicitly declared NOT run, and the only forbidden-phrase hits are the
program's own rulebook quoting its prohibitions. Nothing was modified except
this report file.
