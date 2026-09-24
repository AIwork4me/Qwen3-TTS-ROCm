# TASK 0 — BASELINE FREEZE — Independent Verification Report

**FINAL VERDICT (after fix round 1): PASS** — the two concerns from the initial verification round were corrected in commit b7f38f2 by a one-line edit to the evidence/README.md index row only (transcript left byte-identical, verified); see the "Fix round 1" section below. Initial-round verdict was PASS_WITH_CONCERNS; every substantive requirement of Task 0 was independently reproduced and holds (commit touches only the three claimed paths, no claims changed, baseline HEAD verified as parent, README extract verbatim, all test counts and environment facts reproduce live, upstream GitHub states identical at re-query).

## Fix round 1 — re-verification of commit b7f38f2 (2026-09-24)

Scope per the fix notice: one-line change to the evidence/README.md index row only; the transcript evidence/gfx1151-production-closure-baseline-2026-09-24.txt left byte-for-byte untouched per the repo's append-only transcript policy (the index row is the correction venue, matching the existing Errata convention).

**Fix check 1 — commit scope.** `git show --stat b7f38f2` reports `evidence/README.md | 2 +-` / `1 file changed, 1 insertion(+), 1 deletion(-)`; `git diff 9282229 b7f38f2 --name-only` lists only `evidence/README.md`. The diff replaces exactly one row of the index table. PASS.

**Fix check 2 — transcript immutability.** `diff <(git show 9282229:evidence/gfx1151-production-closure-baseline-2026-09-24.txt) <(git show b7f38f2:evidence/gfx1151-production-closure-baseline-2026-09-24.txt)` produced zero differences — the transcript is byte-identical between the two commits, honoring the append-only policy. PASS.

**Fix check 3 — timings now quote the transcript verbatim.** The new row reads "live CPU suite **313 passed / 38 deselected in 18.07 s**, GPU collection **38 nodes / 313 deselected in 4.30 s**"; transcript line 70 is `313 passed, 38 deselected, 2 warnings in 18.07s` and line 74 is `38/351 tests collected (313 deselected) in 4.30s` — the numeric figures 18.07 and 4.30 now match the indexed artifact exactly (Concern 1 resolved; the previously misquoted 18.48 / 4.19 figures no longer appear anywhere). PASS.

**Fix check 4 — porcelain phrasing now accurate.** The row now says "git HEAD `cd46a81…` with porcelain listing only Task 0's own two untracked deliverables (no tracked-file modifications)", which is precisely what the transcript shows (the two `??` entries for the plan doc and the transcript itself) and is consistent with the additions-only commit diff (Concern 2 resolved). PASS.

**Fix check 5 — no other claims touched.** Every other field of the row is unchanged from 9282229 (HEAD, version, CHANGELOG note, matrix description "13 rows + per-checkpoint table", counts, gfx1151/32 GiB, torch/HIP/GPU/Python, qwen-tts 0.1.1, six model dirs, upstream #372 OPEN / PR #373 OPEN-unmerged head `48b8644…`, vllm-omni `3d43571…`, "No claims changed"); README.md, README_CN.md, CHANGELOG.md, src/, and tests/ are untouched by b7f38f2. The edit corrects a descriptive inaccuracy; it does not add, remove, or strengthen any capability claim. PASS.

**Fix round 1 conclusion:** both discrepancies from the initial round are resolved in the correct venue, with zero collateral change. FINAL VERDICT: **PASS**.

Verification performed by an independent verifier subagent on 2026-09-24, read-only with respect to all tracked files. Repository: /home/amd/Desktop/Qwen3-TTS-ROCm, branch main, verifier working tree at HEAD `9282229ef58dc0d009ff2dd7f1b6d0b55d9f611d` with clean porcelain (only this report file is new and untracked).

## Check 1 — Commit 9282229 touches ONLY the three claimed paths

Command: `git show --stat 9282229`

```
commit 9282229ef58dc0d009ff2dd7f1b6d0b55d9f611d
 ...2026-09-24-gfx1151-production-closure-v0.2.1.md | 993 +++++++++++++++++++++
 evidence/README.md                                 |   1 +
 ...1151-production-closure-baseline-2026-09-24.txt | 130 +++
 3 files changed, 1124 insertions(+)
```

Cross-checked with `git diff --name-status cd46a81 9282229`:

```
A	docs/superpowers/plans/2026-09-24-gfx1151-production-closure-v0.2.1.md
M	evidence/README.md
A	evidence/gfx1151-production-closure-baseline-2026-09-24.txt
```

Result: PASS. Exactly three paths — the program text (new), the evidence transcript (new), and one added row in the evidence index. README.md, README_CN.md, CHANGELOG.md, src/, and tests/ are untouched; the commit is additions-only (1124 insertions, 0 deletions), so no claim text was changed anywhere. This satisfies the Task 0 requirement "change NO claims" mechanically, not just on the implementer's say-so.

## Check 2 — Evidence file internal consistency

Command: `git show 9282229:evidence/gfx1151-production-closure-baseline-2026-09-24.txt`

The file's header line is `=== GFX1151 PRODUCTION CLOSURE v0.2.1 — BASELINE FREEZE — 2026-09-24 ===` (correct date), the GIT section records `$ git rev-parse HEAD` → `cd46a8152a56e0d475512ba8e86c80b46ae0dbe2` (correct baseline HEAD), and every record category required by the program's Task 0 list is present: git HEAD, git status, version (pyproject 0.2.0 plus a CHANGELOG note), README capability matrix extract, CPU test count (line 70: `313 passed, 38 deselected, 2 warnings in 18.07s`), GPU test count (line 74: `38/351 tests collected (313 deselected) in 4.30s`), rocm-smi (gfx1151, 34359738368 B = 32 GiB VRAM), torch/HIP/Python (2.12.0+rocm7.14.0 / 7.14.60850 / 3.12.3), GPU name and gcnArchName (AMD Radeon 8060S Graphics / gfx1151), qwen-tts version (0.1.1 with install location), six model directories, upstream issue #372 and PR #373 JSON, and the vllm-omni main SHA. All commands the program's Task 0 section specifies to run appear verbatim in the transcript. Result: PASS, with one wording observation recorded under Concern 2 below (the porcelain note).

## Check 3 — Recorded baseline HEAD is truly the parent of 9282229

Command: `git rev-parse 9282229^ && git rev-parse 9282229`, verbatim output:

```
cd46a8152a56e0d475512ba8e86c80b46ae0dbe2
9282229ef58dc0d009ff2dd7f1b6d0b55d9f611d
```

(first line is the parent, second is the commit itself). The parent of 9282229 is exactly `cd46a8152a56e0d475512ba8e86c80b46ae0dbe2`, matching both the evidence file and the commit message. Result: PASS.

## Check 4 — README capability-matrix extract is a verbatim copy of README.md lines 49-92 at the baseline commit

Commands: `git show cd46a81:README.md | sed -n '49,92p' > /tmp/readme_baseline_49_92.txt` then extracted the evidence file's matrix section between its `=== README CAPABILITY MATRIX` and `=== CPU TEST SUITE ===` headers and diffed.

First diff result: `44a45 > ` (one extra trailing blank line in the evidence extract). That blank line is the transcript's section separator before the next `===` header, not README content. After stripping trailing blank lines from the extract, `diff` reported zero differences and `wc -l` confirmed 44 lines both sides. Result: PASS — the extract is byte-identical to README.md lines 49-92 at cd46a81 (the full 13-row capability table plus the per-checkpoint table and boundary notes), plus one blank section-separator line.

## Check 5 — Live test counts reproduced NOW (CPU run + GPU collection only; GPU suite not run)

Command: `.venv/bin/python -m pytest -m "not gpu" -q 2>&1 | tail -1`

```
313 passed, 38 deselected, 2 warnings in 18.29s
```

Command: `.venv/bin/python -m pytest -m gpu --co -q 2>&1 | tail -1`

```
38/351 tests collected (313 deselected) in 4.32s
```

Result: PASS. Both counts (313 passed / 38 deselected CPU; 38 of 351 GPU nodes collected) reproduce exactly at current HEAD; wall times differ only in the expected run-to-run way (evidence 18.07s / 4.30s vs verifier 18.29s / 4.32s).

## Check 6 — Environment facts reproduced NOW

Command: `.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); p=torch.cuda.get_device_properties(0); print(p.name, getattr(p,'gcnArchName',None))"` plus `.venv/bin/python --version`

```
2.12.0+rocm7.14.0 7.14.60850 True
AMD Radeon 8060S Graphics gfx1151
Python 3.12.3
```

`.venv/bin/pip show qwen-tts | grep -E "^(Name|Version|Location)"`:

```
Name: qwen-tts
Version: 0.1.1
Location: /home/amd/Desktop/Qwen3-TTS-ROCm/.venv/lib/python3.12/site-packages
```

`ls models/`:

```
Qwen3-TTS-12Hz-0.6B-Base
Qwen3-TTS-12Hz-0.6B-CustomVoice
Qwen3-TTS-12Hz-1.7B-Base
Qwen3-TTS-12Hz-1.7B-CustomVoice
Qwen3-TTS-12Hz-1.7B-VoiceDesign
Qwen3-TTS-12Hz-Tokenizer-12Hz
```

`grep -m1 '^version' pyproject.toml` → `version = "0.2.0"`; `head -8 CHANGELOG.md` confirms the top released section is `## [0.2.0] - 2026-09-22 — Radeon Reference Closure`, matching the evidence file's CHANGELOG note. Live `rocm-smi --showproductname --showmeminfo vram` still reports `VRAM Total Memory (B): 34359738368` (32 GiB), `Card Series: AMD Radeon Graphics`, `Card Model: 0x1586`, `GFX Version: gfx1151` — identical to the transcript. Result: PASS on every field, zero discrepancies.

## Check 7 — Upstream GitHub queries reproduced NOW (2026-09-24)

`gh api repos/QwenLM/Qwen3-TTS/issues/372 --jq '{state,title}'`:

```
{"state":"open","title":"finetuning/sft_12hz.py hard-codes attn_implementation=\"flash_attention_2\" — official fine-tuning fails out-of-the-box on ROCm (no flash-attn build)"}
```

`gh api repos/QwenLM/Qwen3-TTS/pulls/373 --jq '{state,merged,head:.head.sha}'`:

```
{"head":"48b8644aac8512e6fdc0f4442788bf399c425fdb","merged":false,"state":"open"}
```

`gh api repos/vllm-project/vllm-omni/commits/main --jq '.sha'`:

```
3d43571b94f1683412023c45bd886f9ce30f76bd
```

Result: PASS. Issue #372 state open (title verbatim-identical), PR #373 state open / merged false / head `48b8644aac8512e6fdc0f4442788bf399c425fdb`, and vllm-omni main SHA `3d43571b94f1683412023c45bd886f9ce30f76bd` — all three byte-identical to the evidence file's records. No upstream drift since the snapshot (no merge of #373, no close of #372, no vllm-omni main movement), so no program-relevant observation is needed on that front.

## Check 8 — The new evidence/README.md index row does not strengthen any claim

The single added row (verified via the commit diff) is purely descriptive of the snapshot contents: HEAD `cd46a81…`, version 0.2.0, the matrix extract, 313/38 counts, rocm-smi facts, torch/HIP/GPU/Python/qwen-tts facts, six model dirs, upstream states, vllm-omni SHA, and the sentence "No claims changed". It adds no capability, no validation level, no performance or quality statement; the capability vocabulary it uses (e.g. the matrix's existing status labels) is quoted from the already-frozen extract. Its "13 rows + per-checkpoint table" description is accurate — the extract's capability table has exactly 13 data rows. Result: PASS.

## Check 9 — Index row vs evidence file mutual consistency

Result: ONE minor contradiction found (Concern 1 below). Every other field in the row matches the transcript exactly: HEAD, version, CHANGELOG note, matrix description, counts (313/38, 38 nodes/313 deselected), gfx1151/32 GiB, torch 2.12.0+rocm7.14.0, HIP 7.14.60850, GPU name, Python 3.12.3, qwen-tts 0.1.1, six dirs, #372 open, PR #373 open/unmerged head `48b8644…`, vllm-omni `3d43571…`, queried 2026-09-24.

## Discrepancies and concerns

**Concern 1 (minor, factual mismatch between index row and artifact):** the index row in evidence/README.md (line 83) states the live CPU suite ran "313 passed / 38 deselected in 18.48 s", but the transcript it indexes records `313 passed, 38 deselected, 2 warnings in 18.07s` (line 70). Neither `18.48` nor the GPU-collection figure `4.19` (quoted in the implementer's summary) appears anywhere in the committed transcript — its GPU line says `4.30s`. The counts, which are the substantive frozen facts, match everywhere; the wall-clock numbers are descriptive run-to-run noise and no claim depends on them (my own re-run today produced 18.29s / 4.32s). The likely cause is that the index row and summary quoted an earlier or later run of the same commands than the one captured in the transcript. Recommendation: correct the index row's `18.48 s` to `18.07 s` (one-word edit) so the index quotes its own artifact; alternatively record that the figure came from a different run of the same command. Not claim-affecting.

**Concern 2 (minor, wording):** the index row says "clean porcelain" and the transcript's porcelain section carries the parenthetical "(exit=0 — empty output above means clean tree)", yet the output directly above it is not empty — it lists two untracked entries, `?? docs/superpowers/plans/2026-09-24-gfx1151-production-closure-v0.2.1.md` and `?? evidence/gfx1151-production-closure-baseline-2026-09-24.txt`, which are exactly the two files Task 0 itself was creating. The transcript is honest (it shows the entries rather than hiding them) and the substantive freeze requirement holds — zero modifications to tracked files at cd46a81, confirmed mechanically by the additions-only commit diff — but the "clean" phrasing glosses over the two expected untracked artifacts and the parenthetical reads as if the output had been empty when it was not. Recommendation: none required for the freeze; future freezes should phrase this as "no tracked changes; only this task's new files untracked".

No other discrepancies were found in any check.

## Observations

- The Task 0 command list in the program text (docs/superpowers/plans/2026-09-24-gfx1151-production-closure-v0.2.1.md, section "TASK 0 — BASELINE FREEZE" at line 112) is fully covered by the transcript: git status/rev-parse, both pytest invocations, rocm-smi with exactly the specified flags, the torch probe printing torch/HIP/cuda_available/GPU/arch, the version/marker checks, model dirs, upstream queries, and the required evidence filename with the date filled in as 2026-09-24.
- Working tree at verification time: clean porcelain apart from this report file; HEAD is 9282229 with cd46a81 as its only parent, so the verifier reproduced counts at the exact post-freeze commit with no intervening changes.
- Upstream states re-queried live today still match the frozen snapshot; if PR #373 merges or issue #372 closes later in the program, that will be a task-relevant event for whichever task tracks upstream, not a defect of this freeze.

## Verdict rationale

Task 0's purpose is an untampered, reproducible snapshot: no claims changed, exact baseline recorded, counts and environment true. All of that verified independently and exactly. The two concerns are presentation-level (an index-row timing figure that disagrees with its own artifact, and a "clean porcelain" phrasing that glosses two expected untracked files); they do not weaken the freeze, but the index-row timing mismatch is a literal inconsistency between two committed files, so the verdict is PASS_WITH_CONCERNS rather than PASS. Fixing the single timing figure in evidence/README.md would clear the path to PASS on re-verification.
