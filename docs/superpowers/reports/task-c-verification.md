# Task C — Independent Verification: First Green Self-Hosted GPU CI Run

- Verifier: independent verification agent (no subagents, no repo modifications except this file, no new runs triggered)
- Date of verification: 2026-09-23 ~12:22 UTC (local clock and GitHub agree; the session context date "2026-09-21" was stale)
- Claim under test: GitHub Actions run 35857806038 (workflow gpu-nightly, job gpu-short, event workflow_dispatch, 2026-09-23) is a REAL, GREEN, GPU-executing run on the self-hosted radeon-gfx1151 runner, executing commit a3a8a75.

## Verdict

**VERIFIED (PASS).** The claim holds against live GitHub state on every substantive criterion. One non-blocking problem found: the evidence file on disk is defective (gh CLI error dump in place of run metadata; two key log lines missing) and should be regenerated. The run itself is fully corroborated.

## Criteria checklist

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Live run state: completed / success / workflow_dispatch / headSha a3a8a759… | PASS | `gh run view` live JSON: status=completed, conclusion=success, event=workflow_dispatch, headSha=a3a8a75971b5d7ae35bbb5a8309935f75b4951b5, workflowName=gpu-nightly, run #3, url https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/runs/35857806038, created 2026-09-23T12:00:16Z, updated 12:18:19Z |
| 2 | Every gpu-short step success | PASS | All 7 steps conclusion=success (list below). Sibling job full-weekly = skipped (workflow gating, not a failure; steps=[]) |
| 3 | Ran on self-hosted runner, not ubuntu | PASS | Jobs API: runner_name=amd-HP-ZBook-Ultra, labels=[self-hosted, radeon-gfx1151], group=Default. Log header: "Runner name: 'amd-HP-ZBook-Ultra'" / "Machine name: 'amd-HP-ZBook-Ultra'" / runner version 2.337.0. Workspace paths /home/amd/actions-runner/_work/Qwen3-TTS-ROCm/Qwen3-TTS-ROCm (PYTHONPATH in every pytest step) and host venv /home/amd/Desktop/Qwen3-TTS-ROCm/.venv (env HOST_REPO). GitHub-hosted runners are named "GitHub Actions *", group "GitHub Actions", with paths /home/runner/work/... — none of that appears |
| 4 | GPU execution real (torch/cuda/device + SDPA ROCm warnings + pass counts) | PASS | Diagnostics: rocm-smi shows Card Model 0x1586, SKU STRXLGEN, GFX Version gfx1151, 32 GiB VRAM; torch line "2.12.0+rocm7.14.0 True AMD Radeon 8060S Graphics". All three pytest slices emit "Flash Efficient attention on Current AMD GPU is still experimental… sdp_utils.cpp:323" and "Mem Efficient attention… sdp_utils.cpp:383" (AMD ROCm-only warnings). Pass counts: "6 passed, 345 deselected, 5 warnings in 13.93s" + "22 passed, 329 deselected, 4 warnings in 248.76s (0:04:08)" + "4 passed, 347 deselected, 4 warnings in 29.25s" = 32 nodes; deselected 345+329+347 = 1021, and 6+345 = 22+329 = 4+347 = 351 collected per slice — internally consistent |
| 5 | Provenance: code-under-test = checkout of a3a8a75, not host tree | PASS | Log: `/home/amd/actions-runner/_work/Qwen3-TTS-ROCm/Qwen3-TTS-ROCm/src/qwen3_tts_rocm/__init__.py`; diagnostics step has a hard guard exiting non-zero unless '/_work/' is in the module path, and the step succeeded. Remote commit a3a8a75971b5… exists ("ci: harden gpu-nightly for real self-hosted execution…", committer date 2026-09-23T11:46:53Z) and is the current tip of main |
| 6 | SYNCED_TO a3a8a759… present; retry hardening worked (~10 min) | PASS | Log: "SYNCED_TO a3a8a75971b5d7ae35bbb5a8309935f75b4951b5" == run headSha. Retry timestamps: attempt 1 failed 12:02:19Z (GnuTLS -110), attempt 2 failed 12:04:49Z (GnuTLS -110), attempt 3 failed 12:08:03Z (connect timeout after 133946 ms), fetch succeeded 12:11:02Z. Step began 12:00:49Z → ~10 min 13 s from step start to SYNCED_TO; the 15-attempt/60 s-sleep loop demonstrably recovered the run |
| 7 | Evidence file with run metadata, step table, key log lines | PARTIAL | /home/amd/Desktop/Qwen3-TTS-ROCm/evidence/gpu-ci-first-green-2026-09-23.txt exists; step table matches live values exactly; key lines present: SYNCED_TO, provenance path, SDPA warnings, slice-1 and slice-2 pass counts. DEFECTS: (a) the "run metadata" section is a captured gh CLI error — `Unknown JSON field: "run_started_at"` plus the available-fields dump — so status/conclusion/headSha/timestamps were never written; (b) the diagnostics command echo is captured but its output line (torch 2.12.0+rocm7.14.0 True AMD Radeon 8060S Graphics) is missing; (c) slice-3 pass-count line ("4 passed, 347 deselected") and slice-3 "Mem Efficient" warning are missing. What metadata IS present (run id, workflow, job, event) matches live values |
| 8 | Runner online/idle again | PASS | `gh api …/actions/runners`: single runner amd-HP-ZBook-Ultra, status=online, busy=false, labels=[self-hosted, Linux, X64, radeon-gfx1151] |

## gpu-short steps (live, jobs API job id 107170418907)

| # | Step | Status | Conclusion |
|---|------|--------|-----------|
| 1 | Set up job | completed | success |
| 2 | Sync repository (retrying fetch — host github.com is TLS-flaky) | completed | success |
| 3 | Environment diagnostics (fail fast if host prerequisites drifted) | completed | success |
| 4 | Upstream parity + tokenizer (GPU) | completed | success |
| 5 | Core generation suites (CustomVoice 1.7B/0.6B, VoiceDesign, Base clone 1.7B/0.6B incl. reusable prompt) | completed | success |
| 6 | Voice Studio voice workflow (GPU) | completed | success |
| 7 | Complete job | completed | success |

## Exact commands + exit codes

1. `gh run view 35857806038 --repo AIwork4me/Qwen3-TTS-ROCm --json status,conclusion,event,headSha,workflowName,createdAt,updatedAt,number,displayTitle,headBranch,url` — exit 0.
   (First attempt using the brief's field names `runNumber`/`runStartedAt` exited 1 with "Unknown JSON field"; `runnedById` from the brief also does not exist in gh's field set. Corrected to real fields.)
2. `gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/runs/35857806038/jobs --jq '{id,name,status,conclusion,runner_name,labels,steps[]…}'` — exit 0.
3. `gh run view 35857806038 --repo AIwork4me/Qwen3-TTS-ROCm --log > /tmp/run-35857806038.log` — exit 0; 179 lines, inspected in full.
4. `gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/runners --jq '.runners[] | {id,name,status,busy,labels}'` — exit 0.
5. `gh api repos/AIwork4me/Qwen3-TTS-ROCm/commits/a3a8a75971b5d7ae35bbb5a8309935f75b4951b5 --jq '{sha,date,msg}'` — exit 0. `gh api repos/AIwork4me/Qwen3-TTS-ROCm/commits/main --jq '.sha'` — exit 0 (same SHA).
6. `date -u` — exit 0 (2026-09-23 12:22 UTC; confirms the run date is today, not in the future).
7. Read `/home/amd/Desktop/Qwen3-TTS-ROCm/evidence/gpu-ci-first-green-2026-09-23.txt` (Read tool) — success.

No sudo used. No workflow runs triggered. No repo files modified except this report. Log download written only to /tmp.

## Problems found

1. **Evidence file defects (non-blocking, fix recommended):** `evidence/gpu-ci-first-green-2026-09-23.txt` embeds a gh CLI failure (`Unknown JSON field: "run_started_at"` + available-fields list) where the run metadata (status/conclusion/headSha/timestamps) should be; the torch diagnostics OUTPUT line and the slice-3 pass-count line ("4 passed, 347 deselected") are missing, as is slice-3's "Mem Efficient" SDPA warning. The file should be regenerated with corrected field names (`createdAt`/`updatedAt`, not `run_started_at`) and a grep that includes the diagnostics output and all three pass-count lines. All missing content is confirmed present in the live log, so this is an artifact-quality issue, not a truth issue.
2. **Minor log-timestamp anomaly in slice 1 (observation, explained):** pytest self-reports "6 passed … in 13.93s", but log timestamps jump from the progress line at 12:11:18Z to the warnings/summary lines at 12:13:29Z (~2 min 11 s gap). Slices 2 and 3 wall-clock durations match their pytest self-reports almost exactly (248.76 s vs ~249 s; 29.25 s vs ~30 s), and every slice reports an identical 351 collected nodes. The gap is consistent with a delayed log-chunk upload on the documented TLS-flaky host (the same host showed GnuTLS -110 and connect-timeout failures minutes earlier), not with fabricated execution.
3. **Stale context date (observation):** the verification session's context date (2026-09-21) was two days behind both the local clock and GitHub (2026-09-23). No impact; noted to preempt "run dated in the future" confusion.

## Justification

Every substantive element of the claim reproduces from live GitHub APIs and the live run log: the run is completed/success via workflow_dispatch at headSha a3a8a75971b5d7ae35bbb5a8309935f75b4951b5; all 7 gpu-short steps succeeded; the job ran on self-hosted runner amd-HP-ZBook-Ultra (labels self-hosted/radeon-gfx1151) with workspace under /home/amd/actions-runner/_work and the host's .venv — impossible on a GitHub-hosted runner; GPU execution is evidenced by rocm-smi gfx1151 output, torch "2.12.0+rocm7.14.0 True AMD Radeon 8060S Graphics", AMD-only SDPA warnings in all three slices, and 6+22+4=32 passing nodes with consistent deselection counts; the code-under-test provably resolved from the runner workspace checkout of a3a8a75; the fetch-retry hardening survived ~10 minutes of real TLS failures and synced to exactly the claimed SHA; and the runner is online and idle again. The only defect found is in the on-disk evidence artifact (incomplete metadata section and two missing key lines), which should be regenerated but does not affect the verdict.
