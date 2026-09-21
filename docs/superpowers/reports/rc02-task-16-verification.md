# Task 16 independent verification — claims audit wave (brief 17)

Verifier role: independent release verifier (falsification pass). All
acceptance criteria were re-executed against the live repo, the GitHub API,
and the CI record. Nothing was modified except this report file.

## 1. Verdict

**PASS.**

## 2. Exact commit / working-tree SHA reviewed

- Commit under review: `89d001544e11dd8b9b646be07a498cb9a1afc324`
  ("docs: claims audit wave — every public claim evidence-accurate for
  v0.2.0"), parent `9a91900`.
- Working tree at review start: `89d0015`, `git status --porcelain` empty
  (clean). Remote `main` (GitHub API `branches/main.commit.sha`) =
  `89d001544e11dd8b9b646be07a498cb9a1afc324` — the commit is pushed.

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Audit table covers vocabulary grep; ~778 hits; 10 diverse hits spot-checked | PASS | Reproduced **exactly**: 779 post-fix matching lines / 70 files (case-sensitive); pre-fix = 756 tracked (`git grep` @ 9a91900) + 22 untracked `egg-info/PKG-INFO` = 778, post-fix = 757 + 22 = 779. Per-file counts match the audit table (README 67, README_CN 32, CHANGELOG 50, vllm-omni-rocm 34, p0-parity 34, evidence/README 24, pyproject 4, PKG-INFO 22, src 36 over 9 files, tests 59 over 20 files, 2026-08-27-* 34). 12 diverse hits spot-checked (below), all dispositions accurate. |
| 2 | 313 CPU + 38 GPU = 351; READMEs trued; no live stale 305 | PASS | `.venv/bin/python -m pytest --co -q` → "351 tests collected" (exit 0). README.md and README_CN.md both say 351/351 = 313 CPU + 38 GPU (table row, CI paragraph, test-suite bullet, re-run comment). **Zero** `305` occurrences remain in README.md or README_CN.md; remaining `305`s live only in dated contexts: CHANGELOG truing bullet (with explicit supersession pointer), frozen ground-truth row description, dated program records. |
| 3 | Version agreement; CHANGELOG 0.2.0 unreleased | PASS | pyproject `version = "0.2.0"`; `__init__.py` `__version__ = "0.2.0"`; `tests/test_scaffold.py` pins `== "0.2.0"` (passes in the suite run); CHANGELOG heading `## [0.2.0] - Unreleased (opened 2026-09-20; latest tagged release: v0.1.0)`; `git tag -l` → `v0.1.0` only; READMEs contain no `0.2.0` mention at all; other `0.2.0` strings are synthetic test fixtures (`test_check_upstream_drift.py:207`, `test_loader.py:309`). |
| 4 | PR #373 OPEN; docs say submitted-not-merged | PASS | `gh pr view 373 --repo QwenLM/Qwen3-TTS --json state,mergedAt,headRefOid` → `{"state":"OPEN","mergedAt":null,"headRefOid":"48b8644…"}`. README capability row + comparison table: "submitted as Qwen3-TTS PR #373 (OPEN)"; finetuning-rocm.md: "OPEN, not merged" + workaround still required. No strengthened claim. |
| 5 | Deferred minors (a)–(d) | PASS | (a) root-cause.md now says "3 of the 4 `examples/*.py` — `test_tokenizer_12hz.py` contains no such literal" and cites `demo.py:606` (mapping) with the 608–613 pass noted; Q4(b) reads "351-test suite — 305 at experiment time, trued by the 2026-09-21 claims audit". (b) finetuning-rocm.md:103 is one unwrapped line with `` `qwen-tts==0.1.1` `` backticked. (c) erratum block in evidence/README.md ("Errata — stale figure inside `vllm-vs-qwen-tts-2026-09-21.json` `fairness_notes`"), JSON not among the 9 files the commit touches (byte-unedited); erratum figures independently recomputed from the JSON's 72 `rows`: max audio 33.52 s (vLLM p4), qwen max 32.56 s, ≈402 tokens @12 Hz — all match. (d) vllm-omni-rocm.md §6: "pooled-median 7.72 s render (qwen-side median 8.28 s…)"; recomputed: pooled median 7.72, qwen median 8.28 — exact. |
| 6 | jiwer CI fix + run 35597375251 green | PASS | tests/test_quality_eval.py has the try/except `import jiwer` → `_HAVE_JIWER` + `requires_jiwer` skipif with `pip install -e '.[quality]'` reason (same convention as the adjacent resemblyzer block) and exactly 12 `@requires_jiwer` decorators. `gh run view 35597375251 --json` → status `completed`, conclusion `success`, headSha `89d0015…`, all 4 jobs (build, test 3.10/3.11/3.12) `success`. |
| 7 | No claim weakened/strengthened; matrix states & versions untouched | PASS | `git show 89d0015 --stat`: exactly the 9 stated files. No changed line in the diff contains a capability-matrix state (✅/🟡/⬜/🚫). No version number changed; the only version-adjacent edit is the CHANGELOG heading dated → "Unreleased" (strictly more honest). Count/wording edits only; the finetuning-rocm.md change is a re-wrap + backticks with wording unchanged; the ground-truth row gained a supersession annotation. |
| 8 | 89d0015 pushed; tree clean; CPU suite green | PASS | Remote main = 89d0015 (GitHub API); porcelain empty at start; `.venv/bin/python -m pytest -m 'not gpu' -q` → **313 passed, 38 deselected in 18.22 s**, exit 0. |

## 4. Files reviewed

- Inputs: `task-16-brief.md`, `task-16-report.md`,
  `review-9a91900..89d0015.diff`,
  `evidence/claims-audit-2026-09-21.md` (all read in full).
- Current tree: README.md, README_CN.md, CHANGELOG.md,
  docs/finetuning-rocm.md, docs/vllm-omni-rocm.md, docs/benchmarks.md,
  docs/p0-parity-report.md (header), evidence/README.md,
  evidence/upstream-372-root-cause.md, evidence/vllm-vs-qwen-tts-2026-09-21.json
  (recomputed), tests/test_quality_eval.py, tests/test_scaffold.py,
  src/qwen3_tts_rocm/{__init__,loader,env}.py, pyproject.toml; plus the
  per-file grep breakdown across all 70 in-scope files.

## 5. Exact commands executed

```
git rev-parse HEAD; git status --porcelain; git log --oneline -3; git status -sb
grep -rniE 'validated|supported|all models|all Radeon|zero patch|zero patches|fine-tuning|finetuning|vLLM|streaming|305|0\.2\.0|gfx1151|gfx1100' \
  README.md README_CN.md CHANGELOG.md docs evidence/README.md src tests pyproject.toml | wc -l
grep -rE  '<same pattern>' <same set> | wc -l        # case-sensitive
git grep -E '<same pattern>' 9a91900 -- README.md README_CN.md CHANGELOG.md docs evidence/README.md src tests pyproject.toml | wc -l
git grep -lE '<same pattern>' 9a91900 -- <same set> | wc -l
grep -rcE '<same pattern>' <per-class dirs: reports, plans, specs, development/2026-08-27-*, src, tests>
grep -rn 'all models|all Radeon|gfx1100' <scope>     # overclaim check
grep -n '305' README.md; grep -n '305|351' README_CN.md; grep -n '351|313 CPU' README.md
head -8 CHANGELOG.md; grep -n '## \[' CHANGELOG.md
grep -n '7.72|8.28|8–9' docs/vllm-omni-rocm.md
grep -n 'Errata|33.52|402|351|313' evidence/README.md
grep -n 'validated' src/qwen3_tts_rocm/loader.py; grep -n 'version' pyproject.toml
grep -n '__version__' src/qwen3_tts_rocm/__init__.py; grep -rn '0\.2\.0' tests/test_scaffold.py
grep -n 'requires_jiwer|_HAVE_JIWER|import jiwer' tests/test_quality_eval.py; grep -c '@requires_jiwer' tests/test_quality_eval.py
head -20 docs/p0-parity-report.md
grep -n '3 of the 4|demo.py:606|608-613|351-test' evidence/upstream-372-root-cause.md; sed -n '100,110p' docs/finetuning-rocm.md
sed -n '278,295p' CHANGELOG.md
git show 89d0015 --stat; git show 89d0015 --unified=0 | grep -E '^[+-]' | grep -E '0\.[0-9]+\.[0-9]|version|✅|🟡|⬜|🚫'
grep -rn '0\.2\.0' <live surfaces>
.venv/bin/python -m pytest --co -q | tail -3
.venv/bin/python -m pytest -m 'not gpu' -q | tail -3
.venv/bin/python - <<'PY'  (recompute vllm JSON medians/max from rows)
gh pr view 373 --repo QwenLM/Qwen3-TTS --json state,mergedAt,headRefOid,title
gh run view 35597375251 --repo AIwork4me/Qwen3-TTS-ROCm --json status,conclusion,headSha,displayTitle,jobs
gh api repos/AIwork4me/Qwen3-TTS-ROCm/branches/main --jq '.commit.sha'
git tag -l; git check-ignore -v .superpowers/...; git ls-files src/qwen3_tts_rocm.egg-info/ | wc -l
```

## 6. Exit codes

- All git/grep/sed commands: 0.
- `.venv/bin/python -m pytest --co -q`: 0.
- `.venv/bin/python -m pytest -m 'not gpu' -q`: 0 (313 passed, 38 deselected).
- JSON recompute script: 0.
- `gh pr view 373` (first attempt used invalid field `merged`): 1 (my error,
  field list returned); corrected invocation: 0. `baseRefOid` likewise
  invalid → dropped; final: 0.
- `gh run view 35597375251`: 0. `gh api branches/main`: 0.
- `git fetch --dry-run`: TLS error (environment network flake) — remote state
  confirmed instead via `gh api branches/main` (exit 0).

## 7. Runtime evidence inspected

- Live pytest collection (351) and full CPU suite run (313 passed / 38
  deselected, 18.22 s) on this host.
- GitHub API, live: PR #373 `state=OPEN`, `mergedAt=null`, head `48b8644`;
  CI run 35597375251 `completed/success` on headSha `89d0015` with all four
  jobs green; remote `main` at `89d0015`.
- Machine JSON `vllm-vs-qwen-tts-2026-09-21.json` re-parsed: 72 rows;
  pooled median audio 7.72 s; qwen-side median 8.28 s; max 33.52 s (vLLM),
  32.56 s (qwen) — every figure quoted in the audit/erratum/doc reproduced.
- git: tags (`v0.1.0` only), remote SHA equality, per-commit stat (9 files,
  +370/−37), pre-fix grep at `9a91900` via `git grep`.

## 8. Regression tests

- `.venv/bin/python -m pytest -m 'not gpu' -q` → **313 passed, 38
  deselected**, exit 0 — matches the implementer's claim and the trued
  README figure (CPU half). No failures, no new skips on this host.
- `tests/test_scaffold.py` version pin (`== "0.2.0"`) inside that green run.
- Guard behavior in a plain env is proven by CI run 35597375251 (success on
  3.10/3.11/3.12 where jiwer is absent — the visible-skip path).

## 9. Claims audit (spot-checked dispositions, 12 hits)

1. README.md "Automated tests" row → 351/351 (313+38) — matches live count.
2. README.md CI paragraph → "collects the same 313 CPU tests… 15 further
   visible skips" — matches CI run 35597375251 reality.
3. README_CN.md row/bullets → CN lockstep 351/313, no 305 anywhere.
4. CHANGELOG 0.2.0 heading → "Unreleased (opened 2026-09-20…)" — correct,
   only tag is v0.1.0.
5. docs/vllm-omni-rocm.md §6 → 7.72/8.28 s medians — recomputed exact.
6. evidence/README.md erratum → 33.52 s / 402 tokens / qwen 32.56 s —
   recomputed exact; JSON untouched by the commit.
7. evidence/upstream-372-root-cause.md Q2/Q4 → 3-of-4 examples,
   demo.py:606 (+608–613), 351-suite annotation — present as claimed.
8. src/qwen3_tts_rocm/loader.py → "validated AMD ROCm wheel stack ships no
   flash-attn build" — scoped, no overclaim.
9. README streaming row → "🚫 not exposed upstream — measured 2026-09-21" —
   evidence-linked, not overclaimed.
10. docs/benchmarks.md / src/env.py vocabulary hits → factual gfx1151
    records ("none" disposition confirmed).
11. docs/p0-parity-report.md → header declares dated `8815238` scope, README
    authoritative — correctly left unchanged.
12. "all models" / "all Radeon" / "gfx1100" → zero in-scope hits outside
    vocabulary lists — audit F.4 confirmed.

## 10. Problems found

Non-blocking (neither falsifies an acceptance criterion):

1. Audit §F.2 class tallies have small bookkeeping errors:
   "`docs/superpowers/reports/**` — 19 files, 162 hits" sums to 159 hits
   (file count correct), and "plans/** + specs/** — 4 files" is actually
   5 files (hit count 179 exact). Totals (779) and every live-surface
   per-file count reproduce exactly; no disposition is affected.
2. The audit's headline "778/779 hits" is a case-sensitive *matching-line*
   count that includes 22 hits from the untracked build artifact
   `src/qwen3_tts_rocm.egg-info/PKG-INFO` (disclosed in §F.3). The 778
   pre-fix figure only reconciles with that untracked file included. The
   method paragraph does not spell out "lines, case-sensitive, incl.
   untracked egg-info", but the count reproduces exactly under that
   reading and the artifact is disclosed as out-of-scope for claims.
3. `gh pr view --json merged` / `baseRefOid` are not valid fields on this
   gh version (`mergedAt` is) — an implementer-report phrasing nit, not a
   repo defect; the underlying PR facts re-verified live.

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence I re-derived myself rather
than took from the implementer: the headline grep count reproduces to the
exact integer (779 post-fix / 778 pre-fix, 70 files, per-file counts
matching on every live surface); the 351 = 313 + 38 count is confirmed by a
fresh collection and a fresh green CPU run (313 passed, 38 deselected,
exit 0); version agreement is confirmed across pyproject, `__version__`,
the test pin, CHANGELOG (now explicitly Unreleased) and the tag list; PR
#373 is OPEN with `mergedAt` null on the live API while every doc surface
still says "submitted / OPEN, not merged"; all four deferred minors are
verified in the current files, with the vLLM figures (7.72 / 8.28 / 33.52 /
32.56 s, 402 tokens) recomputed from the machine JSON's own 72 rows and the
JSON itself untouched by the commit; the jiwer visible-skip guard exists
exactly as described (12 decorated tests, resemblyzer-convention) and CI
run 35597375251 is completed/success on `89d0015` with all four jobs green;
the diff touches only the nine stated files with no capability-matrix state
or version-number changes; and `89d0015` is the live remote main with a
clean tree. The only findings are two immaterial tally/bookkeeping nits
inside the audit's §F.2 class table and a method-paragraph opacity about
hit counting — none of which weakens any disposition or claim.
