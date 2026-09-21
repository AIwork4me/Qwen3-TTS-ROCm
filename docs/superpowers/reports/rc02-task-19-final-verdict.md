# RC v0.2 Task 18 Step 18.2 / Task 19 pre-gate — Final Independent Release Verdict

Role: FINAL independent release verifier for Radeon Reference Closure v0.2.
No participation in any implementation; every prior agent's PASS treated as a
claim to falsify; live checks re-executed wherever possible; nothing in the
repository modified except this report file.

Single question answered:

> Is Qwen3-TTS-ROCm v0.2 ready to be described as a Radeon reference without
> misleading users?

Date of verification: 2026-09-21 (evening, after commit `400f2b7`).

---

## 1. Verdict

**PASS** — v0.2 may be described as a Radeon reference without misleading
users. Every public claim I attacked is backed by an evidence artifact I
re-read, a number I recomputed from raw machine JSON, a live GitHub state I
re-fetched, or a test I re-ran myself on the validation host. Three minor,
non-blocking observations (Section 10); none misleads any user.

## 2. Exact commit / working-tree SHA reviewed

| Item | Value (verified by me) |
|---|---|
| Downstream HEAD | `400f2b7db62e359f4eba56992d674c50690f7027` (`400f2b7`, branch `main`) |
| Working tree | `git status --porcelain` → empty (clean) |
| CPU-CI-verified SHA | `89d001544e11dd8b9b646be07a498cb9a1afc324` (`89d0015`) — run 35597375251 |
| HEAD − CI gap | exactly 4 commits, all docs/evidence-only (`git diff --stat 89d0015..400f2b7`: closure doc + 2 verifier reports + evidence README + 2 evidence files; zero source/test/config changes) — CI green at `89d0015` therefore still describes the code under review |
| Pristine upstream clone | `.upstream/Qwen3-TTS` @ `022e286b98fbec7e1e916cb940cdf532cd9f488e`, porcelain-clean (0 lines) |
| FIX worktree | `.upstream/Qwen3-TTS-fix` @ `48b8644aac8512e6fdc0f4442788bf399c425fdb` on `fix/finetuning-attn-implementation`, porcelain-clean (0 lines) |
| Live upstream main (gh API) | `022e286b98fbec7e1e916cb940cdf532cd9f488e` — unchanged during the program window |

## 3. Acceptance criteria checklist (the 12 mandated inspections)

| # | Criterion | Verdict | Independent evidence (mine, this session) |
|---|---|---|---|
| 1 | Upstream #372 work + PR live + chain verdict report | **PASS** | `gh pr view 373 --repo QwenLM/Qwen3-TTS` → `state: OPEN`, `mergedAt: null`, title "fix(finetuning): make attention implementation configurable", head `48b8644` = local FIX HEAD exactly. `rc02-task-8-upstream-372-verdict.md` read in full (challenges A–H + 12/12 preconditions, exact commands, exit codes). Spot-checked 3 load-bearing evidence files verbatim: `upstream-372-pristine-failure-run1.txt` (ImportError at :386, `sft-as-is-exit=1` at :387, pristine checks before/after), `upstream-372-e2e-run1.txt` (direct patched invocation at :362, `sft-direct-exit=0` at :386, `RELOAD-AND-SYNTHESIS-OK` at :559), `upstream-372-default-semantics.txt` (parser-default FA2 load fails as expected, unittest-exit=0). FIX diff re-derived by me: 2 files, `+115/−5`; vendor-pattern grep (`amd|hip|rocm|radeon|gfx|cuda`) over the whole diff → 0 hits. |
| 2 | Capability matrix: every ✅ row's evidence exists & plausibly supports claim; 🟡/🚫/⬜ honest | **PASS** | Every evidence filename linked from the README matrix exists on disk (checked all, incl. `gen-customvoice.txt`, `gen-voicedesign.txt`, `gen-voiceclone.txt`, `gpu-suite-2026-09-20.txt`, `voice-workflow-2026-09-20.{txt,json}`, `tokenizer-codec.txt`, `multilingual-matrix.{txt,json}`, `finetune-smoke-2026-09-20.{txt,json}`, the 9-file #372 chain). Multilingual JSON summary recomputed: 24/24 matrix cells passed (10 CV + 10 VD + 4 cross-lingual clone) — matches the README's "10+10+4". vLLM row is 🟡 scoped to offline feasibility + A/B (no serving/quality claims); streaming 🚫 with measured evidence; instruct-control 🚫 pinned by tests. Fine-tuning row is ✅ *scoped* execution-only with the exact PR-OPEN wording (verified in both README and README_CN). |
| 3 | gfx1151 evidence base: index vs disk coherence (sample 10) | **PASS** | Programmatic cross-check of `evidence/README.md` vs directory: 61 index-referenced files, **all 61 exist**; 63 files on disk, only `claims-audit-2026-09-21.md` unindexed in `evidence/README.md` (it is linked from the main README and CHANGELOG — see Problems). Deep-read 10+: the three #372 transcripts above, `vllm-vs-qwen-tts-2026-09-21.json` (72 rows re-parsed), `quality-benchmark-2026-09-21.json` (rows re-parsed), `multilingual-matrix.json`, `benchmark.json` + `benchmark-2026-09-21.json` + `benchmark-06b-2026-09-21.json` (medians recomputed), `final-regression-2026-09-21.txt` (in-transcript 313/38, 38/38, ruff exits, 0.2.0 build). |
| 4 | Second-GPU gap disclosure | **PASS** | Closure doc §6/§8: "deferred — no second-architecture hardware available"; README compatibility table keeps everything non-gfx1151 at 🧪 not-yet-validated. Repo grep for `gfx1100|W7900|7900` over public docs: hits only inside internal plan/spec vocabulary lists and the CHANGELOG's description of the audit methodology — zero validation or parity claims for any second GPU. |
| 5 | CI status | **PASS** | `gh run view 35597375251 --repo AIwork4me/Qwen3-TTS-ROCm` → `completed` / `success`, headSha `89d001544e11dd8b9b646be07a498cb9a1afc324` (exactly the expected SHA). GPU CI: `gh run list --workflow=gpu-nightly.yml` → **empty** (never executed); both READMEs carry only the CPU `ci.yml` badge, no GPU badge; "BLOCKED ON RUNNER INFRASTRUCTURE" wording present in README, README_CN (via scoped row), closure §10, runbook. HEAD-vs-CI gap is 4 docs/evidence-only commits (verified by diff-stat) — no code drift. |
| 6 | vLLM comparison conclusions match evidence JSON | **PASS** | Recomputed from `evidence/vllm-vs-qwen-tts-2026-09-21.json`'s 72 rows: median RTF qwen **1.3033** (n=48) vs vLLM **1.1818** (n=24) → doc's 1.303/1.182 and "~9% faster" (1−1.1818/1.3033=9.3%) exact. Load: qwen 4.772/4.903 s (A1/A2) vs vLLM 69.063 s → "4.8–4.9 s", "69.1 s", "14×" all hold. Memory (sysfs measured-phase peaks): qwen 7119.7/7220.2 MiB (=7.0/7.1 GiB) vs vLLM 26907.0 MiB (=26.3 GiB); ratio 26907.0/7119.7 = **3.779** → "≈3.8×, 3.78:1 vs A1" exact. Secondary figures also reproduce: pooled median audio 7.720 s, qwen median 8.280 s, longest output 33.520 s = 402 tokens @12 Hz, block drift 1.2907→1.3236. Conclusions section carries explicit unproven list + fairness notes; no quality/serving claims. |
| 7 | Quality benchmark scope | **PASS** | Closure §12 + JSON disclose foundation scope (fixed 5-case manifest, whisper-tiny, smoke-grade, "not a quality verdict"); JSON rows re-parsed and match closure §7 digit-for-digit (CER 0.0/0.2 zh, WER 0.0/0.0 en, clone CER 2.8, speaker-sim 0.6728); pesq REJECTED with reason in `omitted_metrics` and the verbatim ITU-T P.862 IPR notice (BT/Psytechnics, KPN/OPTICOM) archived in `evidence/quality-benchmark-2026-09-21.txt` (sdist listing + license text present); STOI scope-limit stated. MOS/naturalness grep over public docs: only "no MOS-style claims" contexts — no overclaim. |
| 8 | Zero-patch wording + upstream clones pristine | **PASS** | Distinction sentence present and correct in closure §14 ("zero committed patches … does **not** mean every official workflow runs unchanged"), operationalized in the README fine-tuning row (0 patches in the verified table + published package "still requires the documented temporary workaround") and `docs/finetuning-rocm.md` §"upstream-fix-status" (same fixed wording). Both clones re-checked by me: pristine `022e286` porcelain-clean; FIX `48b8644` porcelain-clean; FIX branch = pinned base + exactly 2 commits; diff = 2 files (+115/−5), vendor-grep 0 hits. |
| 9 | Version consistency | **PASS** | `pyproject.toml` `version = "0.2.0"`; `src/qwen3_tts_rocm/__init__.py` `__version__ = "0.2.0"`; CHANGELOG heading `## [0.2.0] - Unreleased (opened 2026-09-20; latest tagged release: v0.1.0)`; README agrees (0.2.0-era counts, no version contradictions); `dist/` contains exactly the two fresh 2026-09-21 20:43 artifacts `qwen3_tts_rocm-0.2.0-py3-none-any.whl` + `qwen3_tts_rocm-0.2.0.tar.gz` — no stale 0.1.0 pair. |
| 10 | Final test results re-run by me | **PASS** | CPU: `.venv/bin/python -m pytest -m 'not gpu' -q` → **313 passed, 38 deselected** in 17.07 s, exit 0 (exact expected counts). GPU slice: `.venv/bin/python -m pytest -m gpu -q -k "custom_voice or tokenizer"` → **11 passed, 340 deselected** in 56.34 s (real gfx1151 execution, model loads + synthesis), exit 0. |
| 11 | Release notes absent; closure factual summary accurate; prohibitions satisfiable | **PASS** | No release-notes file anywhere; no `v0.2.0` tag; `gh release list` shows only v0.1.0 — Task 19 correctly not started. Closure doc's summary claims each verified against evidence (§5 rows → files; §7 numbers → benchmark/quality JSONs recomputed; §10 → live run list empty; §11 → recomputed above; §12 → JSON; §15 → my re-runs + live CI). Prohibitions: (a) no "fixed upstream" anywhere on public surfaces while PR OPEN — grep verified, PR live-verified OPEN/unmerged; (b) no GPU-CI-live claim — never-run verified live, no badge; (c) no second-GPU parity — grep verified. All three are satisfiable by the release notes as the docs now stand. |
| 12 | Closure doc structure: 18 sections; known gaps honest | **PASS** (with one observation) | All 18 sections present in order (North Star → … → v0.2.0 release URL pending). §17 lists 8 gaps incl. the deferred minors **ruff format not enforced** (item 6, matches `ruff-format-check-exit=1` in the regression transcript) and **editable-metadata lag** (item 8), plus second-arch, GPU CI, streaming, fine-tuning-merge, quality breadth, and the resolved dist/ cleanup. Observation: the claims-audit §F.2 tally nits are disclosed in `rc02-task-16-verification.md` but not repeated as a §17 item (Problem 1) — immaterial, see below. |

## 4. Files reviewed

- `README.md` (full), `README_CN.md` (fine-tuning/streaming rows, badges), `CHANGELOG.md` (head/Unreleased + claims-audit entry), `pyproject.toml`, `src/qwen3_tts_rocm/__init__.py`
- `docs/radeon-reference-closure-v0.2.md` (full, 495 lines — all 18 sections)
- `docs/vllm-omni-rocm.md` (full), `docs/finetuning-rocm.md` (status sections), `docs/benchmarks.md` (via README cross-checks)
- `docs/superpowers/reports/rc02-task-8-upstream-372-verdict.md` (full), `rc02-task-16-verification.md` (findings §), report index on disk
- `evidence/README.md` (index, programmatic cross-check), `evidence/upstream-372-pristine-failure-run1.txt`, `upstream-372-e2e-run1.txt`, `upstream-372-default-semantics.txt` (verbatim spot-checks), `vllm-vs-qwen-tts-2026-09-21.json` (72 rows re-parsed), `quality-benchmark-2026-09-21.json` (+ pesq section of the .txt), `multilingual-matrix.json`, `benchmark.json`, `benchmark-2026-09-21.json`, `benchmark-06b-2026-09-21.json` (medians recomputed), `final-regression-2026-09-21.txt` (key claims), `claims-audit-2026-09-21.md` (§F)
- `.upstream/Qwen3-TTS` and `.upstream/Qwen3-TTS-fix` (status, HEAD, log, diff)
- `.github/workflows/` (ci.yml, gpu-nightly.yml, upstream-drift.yml present), `dist/` (listing)
- Live GitHub state: PR 373, upstream main SHA, run 35597375251, gpu-nightly run list (empty), release list (v0.1.0 only)

## 5. Exact commands executed (by this verifier)

```
git log --oneline -5; git status --short; git rev-parse HEAD; git diff --stat 89d0015..400f2b7
gh pr view 373 --repo QwenLM/Qwen3-TTS --json state,url,title,headRefName,createdAt
gh pr view 373 --repo QwenLM/Qwen3-TTS --json state,mergedAt,baseRefName,headRefOid
gh api repos/QwenLM/Qwen3-TTS/commits/main --jq '.sha'
gh run view 35597375251 --repo AIwork4me/Qwen3-TTS-ROCm --json status,conclusion,headSha,name,workflowName,event
gh run list --repo AIwork4me/Qwen3-TTS-ROCm --workflow=gpu-nightly.yml --limit 3
gh release list --repo AIwork4me/Qwen3-TTS-ROCm --limit 5; git tag
git -C .upstream/Qwen3-TTS status --porcelain | wc -l; rev-parse HEAD; log --oneline -2
git -C .upstream/Qwen3-TTS-fix status --porcelain | wc -l; rev-parse HEAD; log --oneline -3
git -C .upstream/Qwen3-TTS-fix diff 022e286..HEAD --stat
git -C .upstream/Qwen3-TTS-fix diff 022e286..HEAD | grep -icE 'amd|hip|rocm|radeon|gfx|cuda'
grep -rn "fixed upstream|fixes upstream|merged upstream|landed upstream" README* docs/ CHANGELOG.md   (public surfaces)
grep -rni "gfx1100|W7900|7900" README* docs/ CHANGELOG.md                                            (public surfaces)
grep -n "MOS|naturalness" README.md docs/*.md evidence/README.md
grep -rn "runs unchanged|distinction" README* docs/ CONTRIBUTING.md NOTICE
grep -n "373|0.1.1" README_CN.md; grep -n "zero|patch|373|372" docs/finetuning-rocm.md
python3 <evidence/README.md↔disk cross-check: 61 refs vs 63 files>
python3 <recompute: vLLM 72-row medians/per-block summaries/side_summary; multilingual summary; quality rows; benchmark.json + 09-21 + 06b-09-21 per-alias cell-median ranges>
grep -n "ImportError|sft-as-is-exit|PRISTINE" evidence/upstream-372-pristine-failure-run1.txt
grep -n "sft-direct-exit|RELOAD-AND-SYNTHESIS-OK" evidence/upstream-372-e2e-run1.txt
grep -n "DEFAULT|exit|ImportError" evidence/upstream-372-default-semantics.txt
grep -n "313 passed|38 passed|7/7|ruff|0.2.0" evidence/final-regression-2026-09-21.txt
grep -n "F.2|tally" docs/radeon-reference-closure-v0.2.md evidence/claims-audit-2026-09-21.md; sed -n '140,177p' rc02-task-16-verification.md
.venv/bin/python -m pytest -m 'not gpu' -q
.venv/bin/python -m pytest -m gpu -q -k "custom_voice or tokenizer"          (--co first: 11/351, then full run)
git ls-remote https://github.com/QwenLM/Qwen3-TTS.git HEAD                    (host TLS-flaky → replaced by gh api, same fact)
find … -iname "*release*"; ls dist/ .github/workflows/; grep -n version pyproject.toml; grep -n __version__ src/qwen3_tts_rocm/__init__.py; head -40 CHANGELOG.md
```

## 6. Exit codes (verifier-observed)

- `gh pr view 373` (both forms): exit 0 — OPEN, `mergedAt: null`, head `48b8644` = local FIX HEAD.
- `gh api …/commits/main`: exit 0 — `022e286b…` (upstream unchanged; matches closure §3).
- `gh run view 35597375251`: exit 0 — completed/success at `89d001544e11dd8b9b646be07a498cb9a1afc324`.
- `gh run list --workflow=gpu-nightly.yml`: exit 0, empty output — GPU CI never ran (desired).
- CPU pytest: exit 0 — `313 passed, 38 deselected in 17.07s`.
- GPU-slice pytest: exit 0 — `11 passed, 340 deselected in 56.34s`.
- "fixed upstream" grep over public surfaces: only a no-merged-upstream disclaimer in `docs/finetuning-rocm.md` + the prohibition texts inside internal plan/spec docs — no violating claim.
- gfx1100/W7900 grep: hits only in internal plan/spec vocabulary lists + CHANGELOG audit-methodology description — no parity claims.
- Vendor-pattern grep over FIX diff: 0 (desired). Both clones' porcelain: 0 lines (desired).
- `git ls-remote` direct: failed (host's known github.com:443 TLS flakiness, documented in `scripts/check_upstream_drift.py`'s retry rationale) — fact obtained via `gh api` instead; not a repo defect.

## 7. Runtime evidence inspected

- **Live GitHub state (this session):** PR #373 OPEN/unmerged with head SHA byte-equal to the local FIX worktree; upstream main still `022e286` (no drift since the program pinned it); CPU CI run 35597375251 green at `89d0015`; zero gpu-nightly runs; only release v0.1.0 exists.
- **Re-executed on the validation host (gfx1151):** full CPU suite and a real-GPU slice, both green with exact counts (§6).
- **Re-parsed machine JSON:** vLLM A/B (72 rows → medians 1.3033/1.1818; loads 4.77–4.90/69.06 s; sysfs peaks 7119.7/7220.2 vs 26907.0 MiB = 3.78:1; pooled/qwen audio medians 7.720/8.280 s; longest 33.52 s ≈ 402 tokens; block drift 1.2907→1.3236), quality benchmark (CER/WER/sim rows match closure §7 digit-for-digit; pesq IPR rejection + archived notice), multilingual (24/24 cells), three benchmark JSONs (README RTF ranges 1.31–1.51 / 1.27–1.62 / 1.71–1.88 and 09-21 replication base 1.27–1.40 reproduce exactly).
- **Verbatim transcript spot-checks:** pristine failure (ImportError + exit 1 + before/after pristine invariants), E2E run 1 (direct patched invocation, all phase exits 0, reload-OK), default-semantics (default FA2 fails as pristine; unit leg exit 0), final regression (313/38, 38/38, ruff check exit 0 / format exit 1 informational, 0.2.0 build).

## 8. Regression tests

- `.venv/bin/python -m pytest -m 'not gpu' -q` → **313 passed, 38 deselected**, exit 0 (re-executed by me; exact expected counts, matching `evidence/final-regression-2026-09-21.txt` and the Task 17 verifier).
- `.venv/bin/python -m pytest -m gpu -q -k "custom_voice or tokenizer"` → **11 passed, 340 deselected**, exit 0 (my chosen real-GPU slice: 1.7B+0.6B CustomVoice generation surface + tokenizer codec tests; includes real model loads and synthesis on gfx1151).
- FIX-branch integrity re-verified statically: diff vs pinned base = the two described commits, 2 files, +115/−5, zero vendor-specific patterns.

## 9. Claims audit (hunting for anything that could mislead a user)

- **Every ✅ in the capability matrix traces to an existing, plausible artifact** — I read or re-parsed each class (generation transcripts, GPU suites, multilingual matrix, tokenizer roundtrip, voice-workflow, fine-tuning chain) and found no row claiming more than its file shows; the fine-tuning ✅ is explicitly *scoped* execution-only with the mandated PR-OPEN wording in both languages.
- **Every number I recomputed reproduced exactly** — vLLM headline + secondary figures, quality metrics, benchmark RTF ranges, multilingual cell counts, test counts. Zero numeric drift found between docs and raw JSON.
- **Honest-negative claims verified, not just asserted:** streaming 🚫 is backed by a measured probe; instruct 🚫 pinned by tests; vLLM 🟡 scoped; second-GPU 🧪 deferred; GPU CI never-run confirmed live; pesq rejection archived.
- **Version/artifact state unambiguous:** 0.2.0 everywhere, Unreleased-marked CHANGELOG, dist/ clean of stale artifacts, no premature tag/release/notes.
- **Zero-patch posture holds:** clones pristine, published-package dependency only, parity test present, and the "zero committed patches ≠ every workflow runs unchanged" distinction is stated where users will read it.

## 10. Problems found

None blocking. Three minor, non-blocking observations:

1. **Claims-audit §F.2 tally nits are not repeated in the closure doc's §17 known-gaps list.** `evidence/claims-audit-2026-09-21.md` §F.2 still carries the two immaterial bookkeeping errors the Task 16 verifier found (the "19 files, 162 hits" row sums to 159; "plans/** + specs/** — 4 files" is 5 files) — kept verbatim per the repo's dated-record convention, and fully disclosed in `rc02-task-16-verification.md` §Findings, which closure §16 indexes. But unlike the Task 7/12 rows ("1 minor prose nit", "4 minor, non-blocking"), the Task 16 row carries no qualifier, and §17 (which does list ruff-format and editable-metadata) omits this third deferred minor. Disclosure-completeness nit on an internal record; no public claim is affected; recommend a one-line §17 item or a qualifier on the §16 Task 16 row at next touch.
2. **`evidence/claims-audit-2026-09-21.md` is not indexed in `evidence/README.md`** (63 files on disk, 61 indexed). It is linked from the main README and CHANGELOG, so nothing is hidden — but the evidence index is otherwise exhaustive; add the row for completeness at next touch.
3. **Capture-environment quirk (no repo defect):** direct `git ls-remote` to github.com currently fails on this host with TLS errors — the same flakiness `scripts/check_upstream_drift.py` documents and retries. All live facts were obtained through `gh` (API/CLI), which worked; upstream-unchanged (`022e286`) was verified that way.

## 11. Why PASS is justified

I verified this release as a falsifier, live and hands-on: I re-fetched PR #373 (OPEN, unmerged, head identical to the local FIX branch), re-fetched upstream main (still the pinned `022e286`), re-fetched the CPU CI run (success at exactly `89d0015`, with the 4 commits since proven docs/evidence-only by diff-stat), and proved the GPU workflow has never run (empty run list) while the docs say exactly that. I re-ran the full CPU suite (313/38, exit 0) and a real-GPU slice (11 passed, exit 0) on the validation host myself. I recomputed every headline number in the vLLM comparison, the quality benchmark, the multilingual matrix, and all three benchmark RTF tables from the raw machine JSONs — every one reproduced to the digit, including the ~9% RTF gap, 14× load ratio, 3.78:1 memory ratio, 402-token fairness figure, and the README's quoted RTF ranges. I read the #372 chain's load-bearing transcripts verbatim and re-derived the FIX diff from git (2 files, +115/−5, zero vendor patterns), with both upstream clones porcelain-clean at the expected SHAs. I grepped the public surfaces for each of the three release prohibitions and for MOS/naturalness/second-GPU overclaims and found none — the only hits are disclaimers, dated internal records, or the prohibition texts themselves. Versions, CHANGELOG Unreleased marking, dist/ contents, and the absent release notes are all in the correct pre-release state. The three findings in Section 10 are completeness nits on internal records and a host quirk — none of them could mislead a user about what v0.2 does or does not prove. The repository describes exactly one validated configuration (gfx1151), labels every unproven thing as unproven, and links every claim to evidence that holds up. That is what "Radeon reference without misleading users" requires, and it is met.
