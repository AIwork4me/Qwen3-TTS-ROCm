# Task 9 Independent Verification — vLLM-Omni Online Serving + True Streaming (pc021)

Verdict: **PASS_WITH_CONCERNS**

- Every substantive claim reproduced, including a fresh live server rerun with my own client-side chunk timestamps confirming INCREMENTAL HTTP PCM streaming (warm rerun near-identical to the archived 9D record: 22 chunks, 88320 B, gap max 1.9145 s vs 1.9146 s).
- The concerns are four minor documentation-consistency defects (below), none of which inflates a claim: a wrong character count (464 vs the machine record's 453) propagated into the public matrix, a 2.03 s nonstream number that contradicts the archived 7.832 s machine record, a README_CN table row merged onto one line, and stale "serving/streaming rungs remain unrun" roadmap wording now contradicted by the new ✅ rows. All are underclaims or typos, not overclaims; core evidence discipline (client-side timestamps, verbatim logs, no .venv contamination) holds.

Verifier: independent, assumed implementer wrong, reproduced the checks at HEAD 11a6796 on 2026-09-25 (00:46–00:53 local). No tracked files modified; scratch in /tmp only; live server killed and GPU freed afterward (`rocm-smi`: "No KFD PIDs currently running").

## Check 1 — JSON internal consistency + probe-script client-side timestamping: PASS

Arithmetic recomputed from the archived JSONs (`evidence/vllm-streaming-9d-stream.json`, `evidence/vllm-streaming-9f-long.json`, `-nonstream.json`):

- 9D: 88320 B / 48000 = 1.84 s == `decoded_audio_seconds` ✓; RTF 2.166/1.84 = 1.177 ✓; ttfa/wall 0.249/2.166 = 0.115 ✓; gaps 21 == chunk_count−1 ✓; ttfa+gap_max (0.249+1.9146=2.164) ≤ wall 2.166 ✓; size-sum bounds [2560×22, 4096×22] contain 88320 ✓; `playback_sim` present (`{"underruns": 1, "min_buffer_bytes": 0.0}`) ✓.
- 9F: 1555200/48000 = 32.4 s ✓; RTF 41.289/32.4 = 1.2744 → 1.274 ✓; ttfa/wall 0.222/41.289 = 0.0054 → 0.005 ✓; gaps 388 == 389−1 ✓; bounds [1792×389, 4096×389] contain 1555200 ✓; `playback_sim` present (2 underruns) ✓.
- Evidence JSONs are byte-identical to the raw probe outputs: `diff .work-vllm/probe-9d-stream.json evidence/vllm-streaming-9d-stream.json` (and -nonstream, -9f) all clean.

Probe script inspection (`scripts/vllm_streaming_probe.py`): genuinely CLIENT-SIDE — `requests.post(..., stream=True)` + `r.iter_content(chunk_size=4096)` with `chunks.append((time.monotonic(), len(chunk)))` per chunk (lines 92–107); TTFA, gaps, wall, RTF, playback simulation all derived from those client timestamps; the script never opens or parses any server log (verified by reading the whole 187-line file; only `requests`, `statistics`, `json`, `time` imported). Verdict logic is data-driven: `INCREMENTAL if ttfa/wall < 0.8` (line 181).

Minor quality nit (non-gating): `playback_sim.min_buffer_bytes` is structurally always 0.0 — the buffer is clamped to 0 before the min is taken, so that sub-metric is uninformative; underrun counting ignores dips shallower than 4096 B.

## Check 2 — WAV verification (soundfile, repo .venv): PASS

`evidence/vllm-online-wavs/9a-customvoice.wav` sr=24000, frames=40320, dur=1.6800 s, rms=0.0980, mono PCM_16 — matches claim (1.68 s / 0.0980). `9b-voicedesign.wav` dur=1.0400 s, rms=0.1303 — matches (1.04 / 0.1303). `9c-base-clone.wav` dur=6.4800 s, rms=0.1529 — matches (6.48 / 0.1529). All three exact.

## Check 3 — LIVE RERUN (the core gate): PASS — INCREMENTAL reproduced with my own client-side timestamps

Started one CustomVoice server exactly as the implementer did, isolated stack, port 8092 to avoid clashes:

`HF_HOME=.work-vllm/hf-home HF_HUB_OFFLINE=1 .work-vllm/venv30/bin/vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --deploy-config .work-vllm/venv30/lib/python3.12/site-packages/vllm_omni/deploy/qwen3_tts.yaml --omni --port 8092`

(log `/tmp/task9-verify-server.log`; "vLLM server version 0.30.0, serving model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Delegating entrypoint handling to vllm-omni"; Application startup complete at 09-25 00:48:04 — ready in ~110 s, inside the claimed 60–135 s band.)

(a) Streaming probe, cold first request (`--mode stream --input "Hello, how are you?" --voice vivian --language English --json-out /tmp/task9-verify-stream.json`): TTFA 1.382 s, ttfa/wall 0.178, 20 chunks, 80640 B, audio 1.68 s, wall 7.774 s, RTF 4.627, gap max 6.3794 s, 1 sim underrun → STREAM-VERDICT: INCREMENTAL. (Cold-start JIT cost; TTFA still far under half of wall.)

(a') Streaming probe, WARM second request (`/tmp/task9-verify-stream2.json`) — near-exact reproduction of the archived 9D record: TTFA 0.206 s (theirs 0.249), ttfa/wall 0.097 (theirs 0.115), 22 chunks (theirs 22), 88320 B (theirs 88320 — byte-identical audio), chunk sizes min 2560 / median 4096 / max 4096 (theirs identical), gap max 1.9145 s (theirs 1.9146), wall 2.123 s (theirs 2.166), audio 1.84 s (theirs 1.84), RTF 1.154 (theirs 1.177), 1 sim underrun (theirs 1) → STREAM-VERDICT: INCREMENTAL.

Server-log corroboration of my client numbers (not used as the source, only cross-check): cold `stream=true status=ok total_ms=7771.28 first_chunk_ms=1379.15`; warm `total_ms=2120.41 first_chunk_ms=203.84` vs my client TTFA 0.206 s — client-side and server-side agree to ~2 ms.

(b) Nonstream contrast POST (`/tmp/task9-verify-nonstream.json`): 200, wall 2.166 s, 88364 B, content-type audio/wav (88364 = 88320 PCM + 44-byte WAV header) — single response at completion, matching the implementer's warm-contrast shape (see D2 below re their numbers).

(c) `/v1/audio/voices`: HTTP 200, `{"voices":["aiden","default","dylan","eric","ono_anna","ryan","serena","sohee","uncle_fu","vivian"],"uploaded_voices":[]}` — 10 speakers incl. vivian, exactly as claimed.

Teardown: killed the vllm pid (kill, then kill -9), `pkill -9 -f StageEngineCoreProc`; `ps` count of vllm/StageEngine procs = 0, port 8092 released, `rocm-smi --showpidgpus` → "No KFD PIDs currently running".

## Check 4 — 9C failure modes from artifacts: PASS

- `.work-vllm/server-9c2.log`: `[SpeechE2E] ... status=bad_request ... error=ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)` — verbatim as claimed.
- `.work-vllm/server-9c3.log`: `error=Cannot load local files without --allowed-local-media-path.` (+ `RuntimeError:` repeats) — verbatim.
- `.work-vllm/task9-9bc.log` records the honest iteration chain: inline b64 → `/usr/bin/curl: Argument list too long` (ARG_MAX, request never sent); rerun → http 400 size 163; rerun2 (file:// URI) → 400 size 160; rerun3 (flag) → `http_code=200 time_total=88.819054s size=311084`.
- `.work-vllm/server-9c4.log` line 20: non-default args include `'allowed_local_media_path': '/home/amd/Desktop/Qwen3-TTS-ROCm/src'` — the vLLM core flag was actually set; and `[SpeechE2E] ... stream=false status=ok total_ms=88825.28 response_bytes=311084` + `POST /v1/audio/speech 200 OK` — matches 88.82 s / 311084 B.
- Working request body `.work-vllm/task9-9c-body.json` is inline `task_type=Base` with `ref_audio: file:///…/ref_en.wav` + `ref_text`. Precomputed voice not tested — disclosed as a scope stop in the evidence txt (line 11). Honest.

## Check 5 — 9E drift evidence: PASS

- `.work-vllm/task9-9e.log` contains all four attempts verbatim: (1) default URL → `ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8000)`; (2) `ws://127.0.0.1:8091/v1/audio/speech/stream` → `ConnectionClosedError: no close frame received or sent`; (3) `/v1/realtime` → upstream client dies at `streaming_speech_client.py`, line 270 `print(f"  ERROR: {msg['message']}")` → `KeyError: 'message'`; (4) `/v1/duplex` → `ConnectionClosedOK: received 1000 (OK); then sent 1000 (OK)`.
- `.work-vllm/server-9e2.log` shows the server's actual WS routes: `Route: /v1/realtime, Endpoint: realtime_websocket` and `Route: /v1/duplex, Endpoint: duplex_websocket` — the client's `/v1/audio/speech/stream` path is absent (only an HTTP-level `[accepted]` handshake appears before the drop).
- Upstream-verbatim claim independently checked: `gh api repos/vllm-project/vllm-omni/contents/examples/online_serving/text_to_speech/qwen3_tts/streaming_speech_client.py?ref=7e5897b8482d67800b561d6a7a5736a67c6485a1` sha256 `50218dd367bceb72bb75739d99ef2bdc0364b55a1e7864e2e8d2a618c1bf8bf0` == local `.work-vllm/omni-current-7e5897b/streaming_speech_client.py` — byte-identical, no upstream code modified. The file itself confirms the wrong default port (`ws://localhost:8000/v1/audio/speech/stream`, line 328) and that line 270 is the `msg['message']` printer. Not a gfx1151 failure; not claimed working — correctly recorded.

## Check 6 — Docs audit at 11a6796: PASS with defects D1–D4

- `git diff 145992c..11a6796` shows the intended split: the old 🟡 "vLLM-Omni serving … no serving/streaming/perf claims yet" row replaced by (i) ✅ serving row scoped to the three proven families and (ii) ✅ streaming row citing client-side timestamps with the WS non-working disclosure. The qwen-tts "True streaming inference" 🚫 row is untouched in both READMEs (context-only in the diff) — Rule-3 wording preserved; the streaming-2026-09-21 finding row unchanged.
- `evidence/README.md` index rows (lines 98–99): numbers match the JSONs — TTFA 0.249/0.115/22 chunks/RTF 1.18 (JSON 1.177 → 1.18 ✓), 0.222/0.005/389/32.4 s/RTF 1.27 (1.274 → 1.27 ✓)/gap 12.0 (12.035 ✓)/2 underruns ✓; WS-not-working disclosed ✓.
- CHANGELOG `[Unreleased]` Task 9 entry accurate in substance (serving 3 families, client-side TTFA 0.249/0.222, WS not working, new probe, Rule-3 preserved) — but repeats "464 chars" (D1).
- Roadmap: no WS overclaim anywhere (WS is not claimed in the roadmap at all). But see D4 (stale underclaim).

## Check 7 — official .venv untouched: PASS

`.venv/bin/pip list | grep -i vllm|omni` → empty; no vllm/vllm-omni dirs in `.venv/lib/python*/site-packages`; `.venv/bin/python -c "import vllm"` → ModuleNotFoundError. The isolated `.work-vllm/venv30` stack carried the entire rerun. `git status` after all verification: 0 tracked files modified.

## Discrepancies (all minor; none changes a verdict or overclaims)

- **D1 — 9F input length: 453 vs "464".** Machine record `evidence/vllm-streaming-9f-long.json` (and the raw `.work-vllm/probe-9f-stream.json`) says `input_chars: 453`; the hand-written summary `evidence/vllm-streaming-gfx1151-2026-09-24.json` says `input_chars: 464`, and "464" propagated into the streaming txt, README.md line 74, README_CN line 66, evidence/README.md line 99, and CHANGELOG. The source of truth says 453. Cosmetic for the conclusions (TTFA ratio unchanged), but a wrong number sits in the public capability matrix.
- **D2 — nonstream contrast: 2.03 s vs 7.832 s.** `evidence/vllm-streaming-gfx1151-2026-09-24.txt` line 8 says "Non-stream contrast: 2.03s wall single response" while its own referenced machine records (`evidence/vllm-streaming-9d-nonstream.json`, the summary JSON, and the companion `vllm-online-serving-…txt` line 16) all say 7.832 s (cold first request on a fresh server). My warm rerun measured 2.166 s, so ~2 s is the right warm magnitude — but 2.03 s is an unarchived number and the two companion txts contradict each other.
- **D3 — README_CN line 66 table defect.** The new vLLM-Omni streaming row and the qwen-tts 🚫 streaming row were merged onto ONE markdown table line separated by `||` (missing newline; English README has them on separate lines 74/75). The qwen-tts wording is still present verbatim, but the row does not render as its own table row in README_CN.
- **D4 — stale roadmap wording (both READMEs).** README.md ~line 414 "(for vLLM-Omni, evidence so far is offline-scope…)" and ~line 442 "Still offline-scope only; serving/streaming rungs remain unrun" (README_CN 365–366/389 equivalent) were true after Task 8 but contradict the ✅ serving/streaming matrix rows added by this same commit. It is an underclaim, not an overclaim, and no WS claim is made; still, the section intro is now factually stale.
- **D5 — probe nit.** `playback_sim.min_buffer_bytes` always reports 0.0 by construction (see Check 1); underrun counts ignore sub-4096 B dips. Does not affect the underrun counts reported (1/2) or any verdict.

## Observations

- The warm rerun reproduced the implementer's 9D record to the byte (88320 B, 22 chunks, identical size distribution, gap max within 0.1 ms), which also shows generation is near-deterministic for the same input on this build — the archived numbers are representative, not cherry-picked.
- Cold first request costs ~1.1 s extra TTFA and ~5.6 s extra wall (JIT warmup): cold TTFA 1.382 s / RTF 4.63 vs warm 0.206 s / 1.15. The implementer's archived nonstream (7.832 s) and 9A speech (23.25 s) were cold-first-request numbers; their archived 9D stream (2.166 s wall) was warm. Not deceptive — server logs confirm the sequence — but worth knowing when quoting numbers.
- Server-side `first_chunk_ms` in the vLLM log independently matches the client-measured TTFA to ~2 ms, so the "never inferred from server logs" discipline is doubly satisfied: the client timestamps are real AND they agree with the server's own view.

## Fix round 1 re-verification

Re-verified at HEAD f84d517 (fix commit for D1–D5) on 2026-09-25 by an independent re-verification subagent; implementer assumed wrong; no tracked files modified (git status clean apart from this untracked report); no GPU work. Verdict for the fix round: **PASS** — all five findings verified fixed as claimed; one residual non-gating note on D5 recorded below.

- Check 1 — commit scope: PASS. `git show --stat f84d517` lists exactly the 9 claimed files (evidence/vllm-streaming-gfx1151-2026-09-24.{txt,json}, evidence/vllm-online-serving-gfx1151-2026-09-24.{txt,json}, README.md, README_CN.md, evidence/README.md, CHANGELOG.md, scripts/vllm_streaming_probe.py); I read every hunk of all nine diffs and nothing outside the five findings' scope was touched.
- Check 2 — D1 (453 vs 464): PASS. `grep -rn "464"` over the six named prose files returns zero hits; 453 is present in each place 464 used to be (streaming txt line 19, online-serving txt line 22, README.md line 74, README_CN line 66, evidence/README.md line 99, CHANGELOG line 355). Both summary JSONs now carry 9F `input_chars: 453`, matching the machine record `evidence/vllm-streaming-9f-long.json` `input_chars: 453` (extracted programmatically from all three files). A repo-wide sweep for char-count contexts ("464 char", "464-char", "464 字符", `"input_chars": 464`) finds the only remaining occurrence inside my own round-1 report text above (the historical finding itself — correct to leave); every other "464" hit in the repo is an unrelated substring (sha256/tree fragments, byte counters, MIOpen grid descriptors).
- Check 3 — D3 (CN table row split): PASS. README_CN line 66 is the vLLM-Omni streaming row, line 67 is a separate line starting with `| 真流式推理` (the 🚫 qwen-tts row); no merged `||` row remains. Pipe-count uniformity: table rows 53–67 each have exactly 5 pipes (4-column rows), line 68 is the table-terminating blank line — the table renders as two rows, matching the English README's structure.
- Check 4 — D4 (stale roadmap wording): PASS. `grep -n "remain unrun|尚未运行|offline-scope|离线范围|unrun"` over both READMEs returns zero hits. English roadmap parenthetical now reads "(for vLLM-Omni: offline, online serving, and HTTP-PCM streaming are now evidenced — see the capability matrix rows and the links below; WebSocket streaming remains upstream-broken on the current build)" and item-1's tail now reads "Since then (v0.2.1 Task 9, same night): online serving for all three task families and HTTP-PCM true streaming are evidenced (matrix rows above); the WebSocket rung remains upstream-broken"; README_CN has the exact semantic equivalents (lines 366–368 and 390–391). WebSocket is noted upstream-broken in both languages, no WS overclaim introduced.
- Check 5 — D2 (nonstream contrast): PASS. The streaming txt line 8 now reads "Non-stream contrast (archived machine record, cold first request): 7.832 s wall, single response — audio only at completion (verifier's warm contrast: 2.166 s)"; I confirmed `evidence/vllm-streaming-9d-nonstream.json` `total_wall_seconds` is indeed 7.832 (with response_bytes 80684 matching the companion online-serving txt line 16, unchanged from round 1 and already consistent). The appended "verifier's warm contrast: 2.166 s" matches my own round-1 measurement recorded in this report (check 3(b): warm nonstream POST wall 2.166 s). The streaming summary JSON gained `nonstream_contrast_note` on the 9D-stream entry quoting both numbers; the unarchived "2.03 s" is gone everywhere.
- Check 6 — D5 (probe min_buffer): PASS as claimed, with one residual note. The diff shows `min_buffer = None` initialization plus a four-line comment stating that min_buffer_bytes tracks the shallowest pre-refill buffer level and that the archived v0.2.1 JSONs recorded a by-construction 0.0 which is kept unedited; both archived machine records (`vllm-streaming-9d-stream.json`, `vllm-streaming-9f-long.json`) still carry `min_buffer_bytes: 0.0` unedited as claimed. `.venv/bin/python -m ruff check scripts/vllm_streaming_probe.py` → "All checks passed!" and `py_compile` OK; repo-wide ruff also passes. Residual non-gating note: the metric remains 0.0 by construction for FUTURE runs too — the first chunk's arrival equals `t_play` exactly, so the first pre-refill level is exactly 0.0, and every subsequent level is clamped to ≥0 before the min is taken; and the comment's parenthetical "0.0 == never dipped below the previous chunk's credit" cannot actually distinguish a dipped run from a non-dipped one (dips clamp to 0.0 as well). Strictly a documentation-precision nit on an uninformative sub-metric that was already ruled non-gating in round 1 (underrun counts 1/2 and all verdicts are computed independently of it and are untouched); it does not inflate or deflate any claim.
- Check 7 — no collateral edits to round-1-verified values: PASS. In every diff hunk the metric strings are byte-identical old→new except the intended changes: TTFA 0.249 s / 0.222 s, ttfa/wall 0.115 / 0.005, 22/389 chunks, RTF 1.177→quoted 1.18 / 1.274→1.27, playback-sim underruns 1–2, gap max 1.9146 s / 12.035 s, and the 9E WebSocket client/server drift text in both evidence txts are unchanged; the JSONs' `verdict` fields (INCREMENTAL ×2, UPSTREAM CLIENT/SERVER DRIFT — NOT WORKING) are untouched; only `input_chars` 464→453 and the added `nonstream_contrast_note` differ.

Fix round verdict: **PASS** (all five findings D1–D5 verified fixed as claimed, commit scoped exactly as claimed, lint clean). Overall Task 9 verdict upgraded from PASS_WITH_CONCERNS to **PASS** — D1–D4 fully resolved, D5 resolved to the extent of its non-gating scope with the residual sub-metric informativeness nit recorded above (no impact on any reported number or verdict).
