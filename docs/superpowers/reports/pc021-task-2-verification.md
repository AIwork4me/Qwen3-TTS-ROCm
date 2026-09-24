# v0.2.1 Task 2 — Docker real-GPU E2E closure — INDEPENDENT VERIFICATION

Verdict: **PASS_WITH_CONCERNS** (one cosmetic numeric erratum: the curated docs say "17/17 checks" but the probe records 16 checks; everything substantive — image identity, embedded-prove provenance, full GPU E2E reproduction, WAV/JSON artifact integrity, RTF erratum arithmetic, transcript/log consistency, attempt-1 honesty, docs scope, pytest slice — verified or independently reproduced, all green).

Verifier: independent subagent, 2026-09-24 ~16:35–17:00 +08. Repo at HEAD `1bb0684` (clean tree; only untracked `.work-docker/`). No tracked files modified; writes only this report and `/tmp/task2-verify/`.

## Check 1 — Image identity + embedded probe provenance: PASS

`docker images qwen3-tts-rocm:gfx1151-e2e` → `sha256:3d31b1399fbef848e16df59e2aaa969fe8f0a4dd669230df969d1ed95b907cd3` (id prefix `3d31b1399fbe` as claimed). `docker inspect` → Id same, `Created 2026-09-24T16:03:15.209712301+08:00` — matches transcript line 4 "built 2026-09-24 16:03:15 +0800" exactly.

In-image probe sha256 vs git:

```
git show 488a028:scripts/docker_gpu_e2e.py | sha256sum   → d404b387a7cac7380a38905681e02255b02f85bcef82931295ecb9f6abc35451
git show 96bc051:scripts/docker_gpu_e2e.py | sha256sum   → 81da1e47c6d950f1c20fa57ba3b3287dc6246f00e43769310af8ec63c45fc137
git show 5ca613e:scripts/docker_gpu_e2e.py | sha256sum   → c3711f5d7cb5dc34e6197200aa5ad01fbe56e863ec950a16e9eb463aa25dc29a
docker run … sha256sum /workspace/scripts/docker_gpu_e2e.py → d404b387a7cac7380a38905681e02255b02f85bcef82931295ecb9f6abc35451
```

Image embeds the **488a028** version byte-for-byte. The numpy fix is present in-image (`148: wav_t = torch.from_numpy(wav.astype("float32"))`); `grep -c "Repo-binding definition"` in-image = **0** (the 96bc051 comment/fix is absent), proving the image predates `96bc051` exactly as disclosed.

## Check 2 — Reproduced the container GPU diagnostics: PASS

Reran the image's own embedded probe (`/workspace/.venv/bin/python /workspace/scripts/docker_gpu_e2e.py`) with `/dev/kfd` + `/dev/dri` + `--group-add 44` (video) + `--group-add 992` (render) + mounted `models/`, outputs to `/tmp/task2-verify/`. Exit code **0**. Verbatim check lines:

```
[e2e] OK   torch_is_rocm_build — 2.12.0+rocm7.14.0
[e2e] OK   hip_version — 7.14.60850
[e2e] OK   cuda_available
[e2e] OK   gpu_name — AMD Radeon 8060S Graphics
[e2e] OK   gfx_arch — gfx1151
[e2e] OK   bf16_matmul_finite — shape=(512, 512) min=-102.500 max=110.000
[e2e] OK   sdpa_finite — shape=(2, 4, 64, 32)
[e2e] OK   torchaudio_import — 2.11.0+rocm7.14.0
[e2e] OK   loader_load — alias=custom-voice-0.6b load_seconds=4.29
[e2e] OK   speakers_surface — n=9 first=aiden
[e2e] OK   one_waveform — n=1
[e2e] OK   sample_rate — 24000 (expected 24000)
[e2e] OK   waveform_finite
[e2e] OK   non_silent — rms=0.1122 (>= 1e-3)
[e2e] OK   duration_sane — 3.84s in [0.5, 60]
[e2e] OK   wav_written — /workspace/out/v.wav 184364 bytes via soundfile
DOCKER-GPU-E2E-OK
```

My fresh JSON (`/tmp/task2-verify/v.json`): 16 checks, all ok; generation `{"load_seconds": 4.29, "generate_seconds": 10.1, "audio_seconds": 3.84, "rtf": 0.38, "peak_alloc_gb": 2.28, "peak_reserved_gb": 2.33, "wav_bytes": 184364, "wav_writer": "soundfile"}`; meta torch/hip/gpu/gfx1151 identical to archived; `when_utc 2026-09-24T08:41:58Z`. Numbers differ from the archived run only where stochastic/timing-dependent is expected; every check green. Note my run also produced 3.84 s / 184364 bytes (deterministic greedy decode of the pinned sentence), and `rtf 0.38` — the reciprocal definition, as expected from the 488a028-era probe.

## Check 3 — Archived WAV independently inspected: PASS

`evidence/docker-gpu-e2e-gen-2026-09-24.wav` via host `.venv` soundfile: **samplerate 24000**, mono, **92160 frames = 3.84 s**, PCM_16 WAV, **all finite**, **RMS 0.0909** (≥ 1e-3), peak 0.5859, **184364 bytes**. `sha256sum` = `745d9d5512085bef3503e25bb5b525baf79a4acc3bd8d6c9f574264976788b07` — matches the claim. JSON sha256 = `71308c22718a15d4b3be159c94333b7f7067c59eca10bf3d6977fd64e10b4e63` — matches. `.work-docker/gen.wav` is byte-identical to the archived WAV; `.work-docker/docker-gpu-e2e.json` diff-identical to the archived JSON.

## Check 4 — Archived JSON internal numbers + RTF erratum arithmetic: PASS

JSON `generation`: `load_seconds 6.37`, `generate_seconds 10.47`, `audio_seconds 3.84`, stored `rtf 0.37`. Recomputed: wall/audio = 10.47/3.84 = **2.7266 → 2.73** (repo convention); audio/generate = 3.84/10.47 = **0.3668 → 0.37** = the stored value. Source at 488a028 confirms `rtf = round(dur / gen["generate_seconds"], 2)` — the reciprocal. Erratum statement is arithmetically exact. HEAD's probe (96bc051) computes `generate_seconds / dur` — fix present. (Wording nuance: the erratum calls 0.37 "audio/wall"; precisely it is audio/generate-seconds — audio/probe-wall 18.89 s would be 0.20. The intended arithmetic — reciprocal of 2.73 — is correct.)

## Check 5 — Transcript vs `.work-docker/` logs: PASS

All scratch files present: `build2.log` (1875 lines), `runtime2.log` (609), `pytest.log` (12), `runtime-attempt1-failed.log` (610), `build.log` (1953), `runtime.log` (610), `build-head.txt`, `checksums.txt`, `docker-gpu-e2e.json`, `gen.wav`. Cross-checks: `runtime2.log` contains the exact 16 `[e2e] OK` lines + `wrote …json` + `DOCKER-GPU-E2E-OK` shown in the curated transcript; `pytest.log` ends `4 passed, 2 warnings in 38.33s`; `build2.log` contains the transcript's DONE lines (`#5 DONE 3.3s / #6 DONE 482.0s / #7 DONE 0.1s / #8 DONE 95.3s / #9 DONE 265.0s / #10 DONE 1525.7s`) and `BUILD2_RC=0` (last line); `build.log` ends `BUILD_RC=0` (attempt-1 build also green). `build-head.txt` = `488a02837e5df22cbc81cb48ece13a1daa82d38c` = transcript line 3. `checksums.txt` lists both evidence sha256s, verified above. `--no-cache` consistency: the only `CACHED` in build2.log is `#4` = `FROM docker.io/library/ubuntu:24.04` (local base-image store, not layer cache); repo stages #5–#10 all executed with real durations; 200 `Downloading` lines (transcript: "200 wheel downloads" — 199 of them `.whl`; accurate as an approximation). Timeline reconciles: stage #9 ends ≈16:03:15 = BuildKit's image `Created` (config commit before the 1525.7 s export), export ends 16:28:40 = build2.log mtime, runtime2 16:29:31 = JSON `when_utc 08:29:31Z`, pytest 16:30:28, evidence archived 16:31 — internally consistent, no gaps that suggest hidden reruns.

## Check 6 — Attempt-1 honesty: PASS

`runtime-attempt1-failed.log` shows, in order: 11 green checks (`torch_is_rocm_build … torchaudio_import / loader_load 6.15 s / speakers_surface / one_waveform` — exactly the set the transcript cites), then:

```
  File "/workspace/scripts/docker_gpu_e2e.py", line 143, in main
    rms = float(wav.pow(2).mean().sqrt())
AttributeError: 'numpy.ndarray' object has no attribute 'pow'
```

i.e. the crash is AFTER a successful render at the RMS computation, exactly as disclosed. (`runtime-attempt1-failed.log` is a byte-identical preserved copy of `runtime.log` — fine for gitignored scratch.)

## Check 7 — Docs audit: PASS except the 17/17 count (see Discrepancies)

- `README.md` Docker section (one hunk, @@ -563,8 +563,16 @@): numbers 3.84 s @ 24 kHz, non-silent, WAV written, 4-node GPU pytest slice — all match artifacts; wording upgraded to GPU-runtime-validated. No overclaim (no claim that other models were Docker-validated; scope is the 0.6B CustomVoice chain + listed checks).
- `README_CN.md` (one hunk): same numbers (3.84 秒 @ 24 kHz, 4 项 GPU pytest 切片). Matches.
- `docker/README.md` +31 lines "GPU runtime validation (state: E2E validated 2026-09-24…)": torch 2.12.0+rocm7.14.0, HIP 7.14.60850, gfx1151, 3.84 s @ 24 kHz, RMS 0.0909, soundfile, 4 passed in 38.33 s, erratum 0.37/2.73, correct rerun command with GIDs. All match.
- `evidence/README.md` +3 rows (txt/json/wav): numbers 6.37 s load, 3.84 s @ 24000 Hz, RMS 0.0909, 184364 bytes, 2.28 GiB peak, 4 passed 38.33 s, wall 18.89 s, image `3d31b1399fbe`, HEAD `488a028`, GIDs 44/992, erratum 10.47/3.84=2.73 — all match artifacts. **But both text rows say "17/17 in-container checks green" / "17 check records" — the JSON has 16 (see D1).**
- `CHANGELOG.md`: new entry under `## [Unreleased] → ### Added`; all numbers match artifacts; **also says "17/17 checks" (D1)**.
- All three evidence files are git-tracked (WAV force-added, 184364 bytes binary in the 1bb0684 diffstat, per the stated precedent).

## Check 8 — Capability-matrix scope: PASS

`git diff 5ca613e~1..1bb0684 --stat` touches exactly: `CHANGELOG.md`, `README.md` (12 ±), `README_CN.md` (10 ±), `docker/README.md` (+31), `evidence/README.md` (+3), the three evidence files, `scripts/docker_gpu_e2e.py` (+210). The `README.md` diff is a single hunk in the Docker section (line ~563); the capability matrix (lines ~51–94+) is untouched. No capability rows changed.

## Check 9 — Pytest slice reproduced in-container: PASS

Same docker flags, `cd /workspace && .venv/bin/python -m pytest -m gpu tests/test_generate_custom_voice_06b.py -q` → `4 passed, 2 warnings in 37.17s`, rc 0 (archived run: 38.33 s; timing jitter only, node count identical).

## Discrepancies

- **D1 (the concern behind PASS_WITH_CONCERNS): check-count off-by-one — "17/17" should be "16/16".** The probe records 16 checks (`checks` array length 16 in both the archived and my fresh JSON; 16 `[e2e] OK` lines). "17/17" appears in `evidence/docker-gpu-e2e-gfx1151-2026-09-24.txt` line 44, `evidence/README.md` (two rows: "17/17 in-container checks green", "17 check records"), `CHANGELOG.md` [Unreleased] entry, and the `1bb0684` commit message. Likely origin: counting the 17 lines that start with `[e2e]` (16 OK + 1 `[e2e] wrote … .json`). All 16 real checks passed, so no capability is overstated — but a number in curated evidence docs does not match the artifact it describes. Suggested remediation (repo policy forbids hand-editing archived artifacts): a one-line erratum appended via a future commit to `evidence/README.md` + `CHANGELOG.md` noting the count is 16 (16/16 green); the commit message is immutable.
- **D2 (negligible):** transcript says "200 wheel downloads"; build2.log has 200 `Downloading` lines of which 199 are `.whl`. Effectively accurate.
- **D3 (wording, arithmetic unaffected):** erratum calls the stored 0.37 "audio/wall"; precisely it is audio/generate-seconds (0.37 = 3.84/10.47). Reading "wall" as the 18.89 s probe wall would give 0.20; the code at 488a028 settles the intended meaning.
- **D4 (observation):** `runtime.log` and `runtime-attempt1-failed.log` are identical copies (preservation copy; gitignored scratch).
- **D5 (observation, resolved):** image `Created` 16:03:15 vs build2.log completion 16:28:40 is not an anomaly — BuildKit stamps `Created` when the final config commits (start of the 1525.7 s export phase); the full timeline (build → runtime2 16:29:31 → pytest 16:30:28 → evidence 16:31) is internally consistent.

## Observations

The verification-first design of this task is unusually strong: committing the probe before `docker build` makes the image self-proving (my rerun executed the byte-identical probe, sha256-verified against `git show 488a028`), the honest-iteration log and the no-hand-edit erratum policy both held up under inspection, and the archived WAV/JSON are byte-identical to the scratch outputs of the archived run. The single real defect is the 17-vs-16 count — cosmetic, correctable in living docs, and it does not inflate what was proven: a fresh `--no-cache` image did run the full real-GPU chain (ROCm/HIP/GPU/arch, finite bf16 + SDPA, torchaudio, repository-loader 0.6B CustomVoice load, real synthesis at 24 kHz / 3.84 s / non-silent, WAV write, plus a 4-node GPU pytest slice), reproduced end-to-end by this verifier.

## Fix round 1 re-verification

Independent re-verifier (fresh subagent, adversarial: instructed to assume the implementer is wrong), 2026-09-24 ~17:15 +08. Scope: ONLY commit `68ca633` (parent `1bb0684`, HEAD at audit time), which claims to remediate D1 (17/17 miscount) and D3 (rtf erratum wording). No tracked files modified; the only write is this appended section. Repo state confirmed: `git log --oneline -4` → HEAD `68ca633` ← `1bb0684` ← `96bc051` ← `488a028`; `git status --porcelain` clean except untracked `.work-docker/` and this report.

**R1 — Commit footprint: PASS.** `git show --stat 68ca633` → exactly the three claimed files and nothing else: `CHANGELOG.md` (+4/−1), `evidence/README.md` (+2/−2), `evidence/docker-gpu-e2e-gfx1151-2026-09-24.txt` (+3/−0). Total 9 insertions, 3 deletions.

**R2 — Transcript change is a pure append: PASS.** The diff hunk for the .txt is `@@ -58,3 +58,6 @@` — three context lines, three added lines at EOF, zero deletions in that file. Belt-and-suspenders: `git show 68ca633^:evidence/docker-gpu-e2e-gfx1151-2026-09-24.txt | head -60 | sha256sum` and `head -60 <worktree file> | sha256sum` are byte-identical (`d0676418…`), and the file grew 60 → 63 lines. Lines 1–60 (including the original "17/17 checks ok" capture at line 44 and the `=== VERIFICATION ===` block) are untouched; the no-hand-edit-of-archived-capture policy held.

**R3 — Corrected numbers match the artifact: PASS.** `.venv/bin/python -c "…print(len(d['checks']), all(c['ok'] for c in d['checks']))"` on `evidence/docker-gpu-e2e-gfx1151-2026-09-24.json` → `(16, True)`. `grep -c '\[e2e\] OK' .work-docker/runtime2.log` → **16**, and the next (17th) `[e2e]`-prefixed line is `[e2e] wrote /workspace/out/docker-gpu-e2e.json` (line 608 of 609) — exactly the miscount-origin mechanism the appended erratum states (16 OK lines + 1 wrote-notice = 17 `[e2e]` lines).

**R4 — Erratum arithmetic exact: PASS.** JSON `generation` re-read: `generate_seconds 10.47`, `audio_seconds 3.84`, `rtf 0.37`; `meta.wall_seconds 18.89`. Recomputed: 10.47/3.84 = 2.7266 → **2.73** ✓ (erratum's repo-convention RTF); 3.84/10.47 = 0.3668 → **0.37** ✓ (stored value, now correctly labeled audio-seconds/generate-seconds — the D3 sharpening); and the erratum's parenthetical "not audio/probe-wall, which would be 0.20" also checks out: 3.84/18.89 = 0.2033 → 0.20 ✓.

**R5 — Index rows and CHANGELOG corrections accurate; no other claims touched: PASS.** Full-diff read: the `evidence/README.md` change modifies exactly the two docker-gpu-e2e rows — the .txt row now reads "**16/16 in-container checks green**" with an inline note attributing the "17/17" phrasing to the transcript's original capture and `1bb0684`'s message (both true, both disclosed), and the .json row now says "16 check records" (true) with the D3-sharpened erratum wording "audio-seconds/generate-seconds throughput, NOT the repo-binding wall/audio RTF — correct RTF = 10.47/3.84 = 2.73" (exact per R4). The CHANGELOG hunk adds only the parenthetical miscount correction inside the [Unreleased] entry; no other numbers or claims in any of the three files were altered. "17/17" at HEAD survives only in: the transcript's original line 44 (above the erratum, disclosed), the erratum/correction texts themselves quoting it (transcript line 63, README row, CHANGELOG parenthetical), CHANGELOG line 350's original phrasing (immediately corrected in place by the parenthetical at 354 — annotate-in-place rather than rewrite; disclosed, not misleading), this verifier's own two reports (historical; `rc02-task-10-verification.md`'s "17/17 paths exist" is an unrelated different 17), and the immutable `1bb0684` commit message (`git log --format=%B -1 1bb0684` confirmed unchanged). No occurrence anywhere asserts 17 as the current true count without an adjacent disclosure.

**Fix-round verdict: PASS — D1 and D3 fully remediated.** All five re-checks green; the remediation style (append-only transcript erratum, trued index rows, in-place CHANGELOG annotation) matches the repo's no-hand-edit policy, and every number in the corrections was independently re-derived from the artifacts. One residual observation (not a defect): the CHANGELOG keeps its original "17/17 checks:" sentence with a following correction rather than rewriting it, while the evidence index rows were rewritten outright — a stylistic inconsistency between the two living docs, but the disclosure is immediate and unambiguous, so no reader can be misled.

**Task 2 overall (fix round 1): PASS.** The original PASS_WITH_CONCERNS hinged solely on D1; with the count corrected everywhere mutable and disclosed everywhere immutable, and D3's wording now exact, no open concern remains. D2/D4/D5 were already assessed negligible/resolved in round 1 and were out of scope here.
