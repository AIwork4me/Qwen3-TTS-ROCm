# Task A verification — self-hosted GPU runner installation

Verifier: independent verification agent (execution-based, no subagents, no sudo,
no modifications except this report file)
Date: 2026-09-23 (local CST, UTC+8) — program date 2026-09-21
Input: `.superpowers/sdd/gpu-ci-enablement/task-a-report.md`
(implementer claimed DONE_WITH_CONCERNS)

## Verdict: **PASS** — all 8 criteria confirmed by execution.

---

## Criteria checklist

### 1. Runner online + labeled — PASS

```
$ gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/runners        (exit 0)
{"total_count":1,"runners":[{"id":21,"name":"amd-HP-ZBook-Ultra","os":"Linux",
 "status":"online","busy":false,"version":"2.337.0",
 "labels":[self-hosted(ro), Linux(ro), X64(ro), radeon-gfx1151(custom)]}]}
```

- Name `amd-HP-ZBook-Ultra` ✓; `status:"online"` and `busy:false` (Idle — even
  stronger than the criterion's online-or-busy) ✓; custom label
  `radeon-gfx1151` present with the three read-only defaults ✓; `os:"Linux"` ✓
  (X64 comes via the `X64` read-only label; the API has no separate arch field).
- Repo scope: the runner is returned by the **repo-level** endpoint
  `/repos/AIwork4me/Qwen3-TTS-ROCm/actions/runners`, and
  `/orgs/AIwork4me/actions/runner-groups` → HTTP 404 (re-run this session:
  `{"message":"Not Found",...,"status":404}`) — the expected no-org grouping
  for a personal-account repo. Registration is repo-scoped. ✓

### 2. Service active + enabled + unit file matches — PASS

```
$ export XDG_RUNTIME_DIR=/run/user/$(id -u)
$ systemctl --user is-active github-runner.service   → active    (exit 0)
$ systemctl --user is-enabled github-runner.service  → enabled   (exit 0)
```

Unit file read directly at `/home/amd/.config/systemd/user/github-runner.service`
— byte-for-byte matches the report's §3 block: `Type=simple`,
`WorkingDirectory=/home/amd/actions-runner`,
`ExecStart=/home/amd/actions-runner/run.sh`, `Restart=on-failure`,
`RestartSec=10`, `KillSignal=SIGINT`, `WantedBy=default.target`,
`After=network-online.target`. ✓

Bonus health evidence: `systemctl --user show github-runner.service
--property=NRestarts,ActiveState,SubState` → `NRestarts=0`, `active`,
`running`, up since `Wed 2026-09-23 19:30:13 CST` — zero restarts since start,
so no crash loop behind the "active" answer.

### 3. Linger — PASS

```
$ loginctl show-user amd --property=Linger   (exit 0)
Linger=yes
```

### 4. Tarball integrity — PASS

```
$ sha256sum ~/actions-runner/actions-runner-linux-x64-2.337.0.tar.gz   (exit 0)
70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
$ stat -c %s → 226430031 bytes

$ gh api repos/actions/runner/releases/tags/v2.337.0 \
    --jq '.assets[] | select(.name | contains("linux-x64")) | .digest'   (exit 0)
sha256:70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
```

Local digest == official digest (after stripping the `sha256:` prefix);
size 226,430,031 == the official asset size. ✓

### 5. Runs as user amd, not root — PASS

`systemctl --user show github-runner.service --property=User,UID` returned
`User=` / `UID=[not set]` — expected systemd semantics for a **user** unit (it
runs as the owning user manager's user by construction; there is no User field
to report). The criterion's fallback settles it:

```
$ ps aux | grep run.sh
amd  128501  /bin/bash /home/amd/actions-runner/run.sh
amd  128510  /home/amd/actions-runner/bin/Runner.Listener run
```

Both the wrapper and the listener are owned by **amd** (started 19:30, no root
process involved). The service's MainPID (128501) matches the amd-owned run.sh. ✓

### 6. Security posture claims — PASS

```
$ gh api repos/AIwork4me/Qwen3-TTS-ROCm/actions/permissions   (exit 0)
{"enabled":true,"allowed_actions":"all","sha_pinning_required":false}
```

Matches the report exactly — including the honestly-disclosed
`allowed_actions: all` residual risk.

Workflow file read directly at
`/home/amd/Desktop/Qwen3-TTS-ROCm/.github/workflows/gpu-nightly.yml`:

- `on:` (lines 44–52): **only** `schedule` (cron `0 2 * * *`) and
  `workflow_dispatch` (with the `suite` input). No `pull_request`, no `push`,
  no `pull_request_target`. ✓
- `permissions:` (lines 54–55): `contents: read` at workflow top level. ✓
- Bonus: `actions/checkout` is pinned to a full commit SHA
  (`3d3c42e5aac5ba805825da76410c181273ba90b1`), so despite
  `sha_pinning_required:false` at repo level the workflow itself is SHA-pinned.

### 7. No repo files modified by Task A — PASS

```
$ git -C /home/amd/Desktop/Qwen3-TTS-ROCm status --porcelain   (exit 0)
(no output)
```

Working tree clean — zero modified, staged, or untracked paths.

### 8. Report quality — PASS

The implementer report records all four required items:

- **Truncated-tarball retry**: §1 — found file at 27,417,877 bytes (12% of
  official), `gzip -t` failure, resume-loop re-download, final digest+size
  verification. (Independently corroborated by my criterion-4 check: the
  on-disk tarball now matches the official digest exactly.)
- **Sudo pivot rationale**: §3 + §8.1 — svc.sh requires root; replaced by
  user-level systemd unit + linger per the ruled pivot; explicitly states no
  sudo was used.
- **POST-method registration-token correction**: §2 + §8.3 — GET 404s, correct
  command `gh api --method POST .../registration-token --jq .token`, flagged
  for the Task D runbook patch.
- **Honest residual posture**: §6 + §9 — `allowed_actions: all` +
  `sha_pinning_required:false` called out as the widest-open posture with its
  mitigations and future-exposure conditions; plus runner-group
  not-applicable-for-personal-repo analysis and the `After=network-online.target`
  weak-ordering caveat.

---

## Context (not a criterion): run 35831353081

Independently confirmed the failure attribution. `gh run view 35831353081`:

- `gpu-short` failed in 7m8s at step **"Check out repository"** ("Set up job ✓").
- Annotations: `git` exit code 128;
  `Failed to connect to github.com port 443 after 132873 ms: Couldn't connect`;
  `GnuTLS recv error (-110): The TLS connection was non-properly terminated`.

i.e. the runner accepted the job, set it up, and the *checkout* died on the
host's known github.com TLS/connect flakiness — a network/Task-B-scope defect,
not a runner installation defect. Corroborating runner health after the failure:
journal shows `Listening for Jobs` at 11:30:40Z → `Running job: gpu-short` at
11:30:44Z → `Job gpu-short completed with result: Failed` at 11:37:52Z, and the
runner is now back to `online` + `busy:false` (idle), `NRestarts=0` throughout.

## Problems found

None blocking. Three minor observations:

1. `systemctl --user show --property=User,UID` is vacuous for user units
   ("not set") — inherent systemd behavior, not a defect; process ownership
   (amd, PIDs 128501/128510) is the operative evidence and is correct.
2. The runner was `busy:true` during the implementer's API snapshot and
   `busy:false` (idle) during mine — both within the criterion; the transition
   is fully explained by the scheduled run starting and failing.
3. Pre-existing staleness (NOT a Task A failure — Task A correctly touched
   nothing): the runbook header and the workflow file's header comment still
   say "no self-hosted runner is registered" / "BLOCKED ON RUNNER
   INFRASTRUCTURE". Updating those belongs to the go-live checklist (item 6,
   after a green run) per the runbook's own instructions.
