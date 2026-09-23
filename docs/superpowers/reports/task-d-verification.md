# Task D verification — GPU CI status flip

Date: 2026-09-21 (verifier run). Commit under review: `d6f9d44`
(`a3a8a75..d6f9d44`, docs-only). All checks performed by live
execution/grep against the working tree at HEAD = `d6f9d44` and the
GitHub API. No files modified except this report.

## Verdict

**VERIFIED (PASS)** — every criterion holds against live state; three
minor, non-blocking observations listed under Problems.

## Criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Honest flip: LIVE anchored to run 35857806038, powered/online-at-window qualifier, no overclaim, full-weekly-never-run disclosed | PASS | README.md:181-189 LIVE paragraph with run link + "runs when the validation host is powered/online at the window — a missed window is not a regression signal"; README_CN.md:162-170 same qualifier in Chinese ("仅当验证主机在该时刻开机在线时才会运行 —— 错过窗口并非回归信号"); runbook STATUS header (gpu-ci-runbook.md:3-5) LIVE with the qualifier. No "always runs nightly" wording anywhere (grep). full-weekly disclosure: runbook jobs table (line 35, workflow_dispatch-only, no cron) + go-live checklist item 5 "☐ … still pending" (line 284) and item 6 "full-weekly (step 5) still unrun" (line 288). Note: the README paragraphs attribute the green run to `gpu-short` specifically and never claim full-weekly ran; the explicit never-executed statement lives in the linked runbook (see Problems #2). |
| 2 | Runbook corrections all present and accurate | PASS | Verified by grep in the live file: `--method POST` for registration-token (gpu-ci-runbook.md:159); release-asset API `digest` field, no `.sha256` sidecar (163-165); user-level systemd unit + `loginctl enable-linger` as as-built (144-149, 178) with `sudo ./svc.sh` labeled "the upstream default, **not** what was done" (182-186); "Security posture (binding constraints, as actually deployed)" personal-repo section incl. runner-group endpoints N/A + allowed_actions=all residual (193-215); cron `0 18 * * *` (34, 227) — matches the live workflow file (`.github/workflows/gpu-nightly.yml:61`); tracked-file-write caveat — `git clean -qfdx` does not revert tracked modifications (248-255); `benchmark-nightly.json` ephemerality — no actions, not a run-page artifact, next run's clean deletes it (256-264). |
| 3 | GPU badge next to an EXISTING CI badge, correct URL | PASS | README.md:28 `[![GPU CI](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/gpu-nightly.yml/badge.svg?branch=main)](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/gpu-nightly.yml)` immediately after the existing `ci.yml` badge (line 27, same `?branch=main` style); README_CN.md:22 after its CI badge (line 21). Remote workflow filename confirmed via API: `.github/workflows/` on main contains `gpu-nightly.yml` — URL pattern correct for AIwork4me/Qwen3-TTS-ROCm. |
| 4 | evidence/README.md row + evidence file matches live run | PASS | Row present (evidence/README.md:82). File line 3 `status=completed conclusion=success event=workflow_dispatch headSha=a3a8a75971b5d7ae35bbb5a8309935f75b4951b5` — matches live `gh run view` byte-for-byte. All 7 step names + completed/success conclusions match live `gh run view … --json jobs` output exactly. See Problems #1 for a cosmetic duplication. |
| 5 | CHANGELOG [Unreleased], conventions, no version bump | PASS | CHANGELOG.md:343-345 `## [Unreleased]` + `### Changed` — heading style matches the file's existing `### Added/Verified/Changed/Fixed/Performance/Security` convention (lines 10/171/266/325/381/414/421). No new version heading; `[Unreleased]` retained. |
| 6 | Residual "BLOCKED ON RUNNER INFRASTRUCTURE" hits all justified | PASS | Full `grep -rn` (excluding .git) returns hits in exactly 8 files; every one is a dated historical record or describes one, and the implementer's disposition table matches my grep exactly. Hit-by-hit judgment below. |
| 7 | Closure doc §10 addendum, record otherwise untouched | PASS | `git show d6f9d44 -- docs/radeon-reference-closure-v0.2.md` = one hunk, +4 lines only (addendum at :298-300). §10 body (:283-296) and §17 item 2 (:466) verbatim. |
| 8 | d6f9d44 == origin/main, tree clean | PASS | `gh api repos/AIwork4me/Qwen3-TTS-ROCm/commits/main --jq .sha` → `d6f9d44f9ef2ab71ab629d668f90525e854c7fac` (first try, no TLS retry needed); `git rev-parse HEAD` = same; `git status --porcelain` empty (exit 0). |
| 9 | CPU suite green | PASS | `.venv/bin/python -m pytest -m 'not gpu' -q` → `313 passed, 38 deselected, 2 warnings in 16.75s`, exit 0 — exactly the expected 313/38. |

### Criterion 6 — every remaining hit and judgment

| Hit | Judgment |
|---|---|
| docs/radeon-reference-closure-v0.2.md:285 (§10 body), :466 (§17 item 2) | Allowed — dated v0.2.0 release record left verbatim; §10 now ends with the 2026-09-23 addendum (:298) pointing to runbook + evidence, exactly the "addendum instead of rewrite" mechanism. |
| evidence/claims-audit-2026-09-21.md:192 | Allowed — historical claims-audit record, task instruction says verbatim. |
| evidence/gpu-ci-prep-validation.txt:5,140 | Allowed — append-only machine transcript of the dated 2026-09-21 prepare-only validation; its STATUS header is historically true of that file. |
| evidence/README.md:76 | Allowed (judgment call) — the hit is inside the index row *describing* the prep-validation file's own STATUS header; still an accurate description of that dated artifact, not a current-state claim; the new LIVE row sits directly below (:82). Disclosed by the implementer. |
| docs/superpowers/specs/2026-09-21-radeon-reference-closure-v0.2-design.md:34,89 | Allowed — dated design spec (prepare-only decision record). |
| docs/superpowers/plans/2026-09-21-radeon-reference-closure-v0.2.md:504,553,555,561,841 | Allowed — dated implementation plan. |
| docs/superpowers/reports/rc02-task-19-final-verdict.md:45 | Allowed — dated verification report (2026-09-21 state). |
| docs/superpowers/reports/rc02-task-11-verification.md:15,77,145,162 | Allowed — dated verification report + quoted command transcript. |

No live-state document (README, README_CN, runbook, docs/development
index) retains the phrase.

## Commands + exit codes

| Command | Result |
|---|---|
| `gh run view 35857806038 --repo AIwork4me/Qwen3-TTS-ROCm --json status,conclusion,event,headSha` | exit 0 → completed / success / workflow_dispatch / a3a8a75971b5d7ae35bbb5a8309935f75b4951b5 |
| `gh run view 35857806038 … --json jobs` (steps) | exit 0 → 7 steps, all completed/success, names match evidence file verbatim |
| `gh api repos/AIwork4me/Qwen3-TTS-ROCm/commits/main --jq .sha` | exit 0 (try 1) → d6f9d44f9ef2ab71ab629d668f90525e854c7fac |
| `gh api repos/AIwork4me/Qwen3-TTS-ROCm/contents/.github/workflows --jq '.[].name'` | exit 0 → ci.yml, gpu-nightly.yml, upstream-drift.yml (badge target exists) |
| `git show d6f9d44 --stat` / full diff | 8 files, +241/-75; per-file hunks as described in the implementer report |
| `git status --porcelain` | exit 0, empty — tree clean |
| `grep -n cron .github/workflows/gpu-nightly.yml` | `0 18 * * *` (line 61) — matches runbook/README/CHANGELOG claims |
| `grep -rn "BLOCKED ON RUNNER INFRASTRUCTURE"` (excl. .git) | 8 files / 17 lines — all dispositioned above |
| `.venv/bin/python -m pytest -m 'not gpu' -q` | exit 0 → 313 passed, 38 deselected |

## Problems

All minor; none block the verdict.

1. **Cosmetic duplication in `evidence/gpu-ci-first-green-2026-09-23.txt`** — the two log lines after the `=== VERIFICATION ===` pointer (the "22 passed … 248.76s" and "4 passed … 29.25s" lines) duplicate the tail of the KEY LOG LINES block verbatim. Values are correct against the live run; looks like a paste artifact in the transcript. Cosmetic only.
2. **README/README_CN do not spell out "full-weekly has never executed" in the paragraph itself** — the LIVE paragraphs attribute the green run to `gpu-short` specifically (so no overclaim) and link the runbook, where checklist item 5 ("still pending") and item 6 ("still unrun") carry the explicit disclosure. Judged adequate under criterion 1's "no overclaim + disclosed where the workflow is described"; noting it since a stricter reading would want the fact in the README paragraph too.
3. **evidence/README.md:76 retains the phrase** — borderline by the letter of "allowed ONLY in dated historical records"; judged acceptable because the row describes the dated prep-validation file's own header (accurate as a description of that artifact), not current GPU-CI state, and the implementer disclosed it explicitly.

## Justification

Every factual anchor in the flip was re-derived live and matches: the run
(35857806038, completed/success/workflow_dispatch/a3a8a75971b5…), the
cron (`0 18 * * *` in the actual workflow file), origin/main == HEAD ==
d6f9d44 with a clean tree, and the CPU suite at exactly 313/38. The
LIVE wording carries the powered-at-the-window qualifier in all three
places with no unconditional nightly overclaim; all seven runbook
corrections are present and accurate in the live file; the badge URL is
correct and sits beside the pre-existing CI badge in both READMEs; the
evidence file matches the live run field-for-field and step-for-step;
the CHANGELOG entry follows convention with no version bump; and every
one of the 17 residual "BLOCKED ON RUNNER INFRASTRUCTURE" hits is a
dated historical record (or an accurate description of one), matching
the implementer's disposition table exactly. The three problems found
are cosmetic or judgment-call disclosures, none rising to a blocking
issue.
