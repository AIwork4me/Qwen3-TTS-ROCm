# RC v0.2 Task 6 — Independent release verification (Rounds B+C, two independent full fine-tuning E2E runs)

Verifier role: independent release verifier. Assumed the implementer might be wrong;
attempted to falsify every acceptance criterion against source, git state, on-disk
artifacts, raw evidence, and the CPU suite (run by the verifier, not trusted from the
report). Nothing outside this report file was modified.

## 1. Verdict

**PASS.**

## 2. Exact commit / working-tree SHA reviewed

- Downstream repo `/home/amd/Desktop/Qwen3-TTS-ROCm`: branch `main`, HEAD
  `16c92cc` ("test(upstream): Rounds B+C — two independent ROCm fine-tuning E2E runs
  pass with sdpa"), the single commit in the review range `69a9b4c..16c92cc`.
  `git status --porcelain` at review time: **empty** (clean tree; evidence files on
  disk are byte-identical to the committed blobs).
- Transcripts record the runs themselves against downstream HEAD `69a9b4c8b46599acd…`
  (the pre-commit state — expected: the evidence is what the commit adds).
- FIX worktree under test: `.upstream/Qwen3-TTS-fix`, branch
  `fix/finetuning-attn-implementation`, HEAD `48b8644aac8512e6fdc0f4442788bf399c425fdb`.
- Pristine upstream clone: `.upstream/Qwen3-TTS`, HEAD `022e286` ("fix finetuning bug").

## 3. Acceptance criteria checklist

| # | Criterion | Result | Independent evidence (verifier-executed, not copied from the report) |
|---|---|---|---|
| 1 | Genuine independence: two clean trees, dataset re-prepared fresh in each, separate processes, no reused checkpoint/trainer state | **MET** | Transcripts show `ls` non-existence → `rm -rf && mkdir` for `e2e-r1/` (created 14:43:55+08:00) and `e2e-r2/` (created 14:46:40+08:00), each with a full 12-utterance render pass ([render] lines with per-run timings, per-run file mtimes 14:44:xx vs 14:47:xx) and its own `prepare_data.py` run writing into its own tree. Verifier recomputed all dataset hashes on disk NOW: `train_raw.jsonl` `e8adc00c…` (r1) vs `9d4a6c1d…` (r2) — distinct jsonl renders; `train_with_codes.jsonl` `2985c706…` vs `e898e205…` — distinct; concatenated wav-bytes `c76541c3…` BOTH runs — identical, consistent with the documented per-render `torch.manual_seed(1234)` determinism (identical bytes are the *expected* signature of two fresh deterministic renders, not of a copy: a copy could not produce different jsonl hashes with in-tree paths, and the r2 render phase visibly executed for ~66 s with its own per-utterance timings). `grep -c e2e-r2` in r1's jsonl files = 0 and `grep -c e2e-r1` in r2's = 0 — zero cross-tree references. Separate processes: r1 driver log mtime 14:46:09.4 vs r2 driver header 14:46:38+08:00 (r2 started ~29 s after r1's driver completed); direct-sft tee logs 14:45:24 vs 14:48:10. No reused trainer state: `sft_12hz.py` constructs a fresh `AdamW` and loads `--init_model_path models/Qwen3-TTS-12Hz-1.7B-Base` every invocation (source lines 65–72); nothing resumes a checkpoint. Four trainings → four distinct epoch-1 `model.safetensors` sha256 (recomputed by verifier, table in §7). |
| 2 | Patched official script itself ran: DIRECT `python finetuning/sft_12hz.py … --attn_implementation sdpa` in BOTH transcripts, exit 0 | **MET** | Transcribed verbatim in both transcripts (run1 txt line 362, run2 line 362): `(cd .upstream/Qwen3-TTS-fix && .venv/bin/python finetuning/sft_12hz.py --attn_implementation sdpa --init_model_path … --output_model_path <run>/runs --train_jsonl <run>/data/train_with_codes.jsonl --batch_size 2 --lr 2e-5 --num_epochs 2 --speaker_name e2e_rN_speaker) 2>&1 | tee .work-finetune/e2e-rN.log` with `sft-direct-exit=0 … (PIPESTATUS[0], NOT of the tee wrapper)` in both (line 386 each). The tee targets `.work-finetune/e2e-r1.log`/`e2e-r2.log` still exist on disk and contain the same stdout incl. the loss lines. The launcher appears only as the disclosed phase-D peak-mem supplement, which the brief explicitly authorizes. |
| 3 | Full chain per run: prepare exit 0 → real optimizer steps → checkpoints on disk NOW → official-API reload → synthesis finite/non-silent/24 kHz/plausible | **MET** | `prepare-exit=0` both; training loss start→end recorded verbatim (r1 14.6621→10.0124, r2 14.1067→11.9973) — real forward/backward/step work, and the step arithmetic is source-backed: `optimizer.step()` fires on every microbatch (only `clip_grad_norm_` is gated on `sync_gradients`), so 12 samples / batch 2 × 2 epochs = 12 optimizer steps. Checkpoints exist NOW (verifier `ls` + `sha256sum`): 6 dirs, each `model.safetensors` 3,833,402,520 bytes (plausible for 1.7B bf16 + voice components), mtimes 14:45 (r1) / 14:48 (r2), all four epoch-1 hashes distinct and equal to the claimed values. `config.json` on disk NOW: `tts_model_type=custom_voice`, `spk_id={'e2e_r1_speaker': 3000}` / `{'e2e_r2_speaker': 3000}` (verifier-parsed). Reload: `reload-exit=0` + `RELOAD-AND-SYNTHESIS-OK` both; verifier re-parsed both wav headers from disk NOW: PCM fmt=1, mono, 24000 Hz, 16-bit, 76,800 frames → 3.2000 s (r1) and 82,560 frames → 3.4400 s (r2); peak 0.3320/0.1914, rms 0.05800/0.04274, nonzero-sample fractions 0.9894/0.9264 — finite, non-silent, 24 kHz, plausible durations, all matching the transcript claims to 4–5 decimals. The reload path is the official API: `qwen3_tts_rocm.loader.load` imports `Qwen3TTSModel` from `qwen_tts` and calls `Qwen3TTSModel.from_pretrained` (bf16, HIP default attn sdpa) — verifier read `src/qwen3_tts_rocm/loader.py`. |
| 4 | Metrics recorded per run (steps, losses, batch, precision, wall, peak-mem, ckpt path, audio metadata) | **MET** | Both JSONs carry: optimizer_steps=12 with arithmetic note, verbatim loss lines, batch_size=2, precision bf16 (source-backed: `Accelerator(mixed_precision="bf16")` + `torch_dtype=torch.bfloat16`), wall 22.8/23.2 s, peak-mem 18.013/18.477 GiB (r1) and 18.026/18.496 GiB (r2) from in-process `torch.cuda.max_memory_allocated/reserved`, checkpoint paths, and full audio metadata. Peak mem is read from a real training process (the launcher resets peak stats at start and runs `train()` verbatim with identical args — verifier read `e2e_sft_launcher.py`); direct checkpoints remain the reload source, untouched (launcher writes to `runs-mem/`). |
| 5 | Only upstream difference vs pristine is the FIX branch diff | **MET** | Verifier ran: `git -C .upstream/Qwen3-TTS-fix status --porcelain` → empty; `git -C .upstream/Qwen3-TTS-fix log --oneline -3` → `48b8644` → `0be0026` → `022e286` exactly; pristine `.upstream/Qwen3-TTS` clean at `022e286`; `git -C .upstream/Qwen3-TTS-fix diff 022e286..HEAD --stat` → only `finetuning/sft_12hz.py` (15 lines) + new `tests/test_sft_attn_implementation.py` (105 lines). Verifier read the full sft_12hz.py diff: adds `--attn_implementation` (default `flash_attention_2`, upstream default preserved) via a `build_arg_parser()` refactor and passes `args.attn_implementation` to `from_pretrained` — minimal and behavior-preserving otherwise. `diff` of `prepare_data.py` pristine-vs-FIX prints IDENTICAL (in-transcript and consistent with the diff stat). |
| 6 | rocm-smi snapshots around GPU phases; headers complete | **MET** | Both transcripts wrap all five GPU phases (A dataset-gen, B prepare, C direct train, D launcher, E reload+synthesis) with `rocm-smi --showproductname --showuse --showmeminfo vram` BEFORE and AFTER — 10 snapshots per transcript, all inside the tee'd capture. Headers carry: GPU (AMD Radeon 8060S via torch + rocm-smi), gfx1151, hip 7.14.60850, torch 2.12.0+rocm7.14.0, Python 3.12.3, qwen-tts 0.1.1 (+transformers/accelerate/safetensors), kernel 6.17.0-1032-oem, downstream SHA, FIX branch+HEAD, pinned upstream SHA `022e286b98fbec…` (diff-stat command + JSON `code_under_test`), exact echoed commands (`$ …` lines from the driver), and per-phase exit codes. |
| 7 | Transcripts never hand-edited (tee flow visible); disclosed deviations recorded and non-weakening | **MET** | Driver script `.work-finetune/e2e-driver.sh` (mtime 14:43:43, i.e. written before run 1) generates every line of the transcripts; its header documents the outer `bash … 2>&1 | tee …-driver.log`. Verifier `cmp`'d the archived evidence against the tee captures: `evidence/upstream-372-e2e-run1.txt` ≡ `.work-finetune/e2e-r1-driver.log` and run2 ≡ `e2e-r2-driver.log` (both byte-identical). Transcript sha256 recomputed: `68156fbb…` / `6877ad19…` — equal to the JSON-claimed values. Deviations: (a) absolute paths for model/jsonl args (forced by cwd = FIX worktree; scale/args otherwise identical to the 2026-09-20 smoke — verifier cross-checked the smoke transcript: 1.7B Base, batch 2, lr 2e-5, 2 epochs, 12 samples, 6 steps/epoch, loss-only-at-step-0); (b) per-run speaker names — label only, strengthens independence; both recorded in report + JSONs, plus the launcher supplement (~19 s extra training per run) disclosed in-transcript and in both JSONs. None weakens the proof. |
| 8 | CPU suite green (expected 267 passed, 38 deselected) | **MET** | Verifier ran `.venv/bin/python -m pytest -m 'not gpu' -q` itself: **267 passed, 38 deselected in 16.23s** — exact match. |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-6-brief.md` (brief 6B+6C)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-6-report.md` (implementer report — treated as claims, not evidence)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-69a9b4c..16c92cc.diff` (5 files, +1469: 4 evidence files + 4 README rows)
- `evidence/upstream-372-e2e-run1.txt` (598 lines), `…run1.json`, `…run2.txt`, `…run2.json`
- `evidence/finetune-smoke-2026-09-20.txt` (validated scale/args reference)
- `evidence/README.md` (new rows for the four files)
- Source: `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py` (+ its `022e286..HEAD` diff), `.upstream/Qwen3-TTS-fix/finetuning/prepare_data.py` vs pristine, `src/qwen3_tts_rocm/loader.py`
- Run machinery: `.work-finetune/e2e-driver.sh`, `.work-finetune/e2e_sft_launcher.py`, `.work-finetune/e2e_reload_check.py`, `.work-finetune/e2e-r{1,2}.log`, `.work-finetune/e2e-r{1,2}-driver.log`
- On-disk artifacts under `.work-finetune/e2e-r1/` and `e2e-r2/` (data, wavs, checkpoints)

## 5. Exact commands executed (by the verifier)

```
git log --oneline -8 ; git status --porcelain ; git diff --stat 69a9b4c..16c92cc
sha256sum evidence/upstream-372-e2e-run1.txt evidence/upstream-372-e2e-run2.txt
sha256sum <4× checkpoint-epoch-1/model.safetensors>
.venv/bin/python <verifier script: RIFF/WAV header parse + duration/peak/rms/nonzero for both reload wavs>
cmp evidence/upstream-372-e2e-run1.txt .work-finetune/e2e-r1-driver.log
cmp evidence/upstream-372-e2e-run2.txt .work-finetune/e2e-r2-driver.log
git -C .upstream/Qwen3-TTS-fix status --porcelain | wc -l ; git -C .upstream/Qwen3-TTS-fix log --oneline -5
git -C .upstream/Qwen3-TTS status --porcelain | wc -l ; git -C .upstream/Qwen3-TTS log --oneline -2
git -C .upstream/Qwen3-TTS-fix diff 022e286..HEAD --stat ; … -- finetuning/sft_12hz.py (full diff)
sha256sum .work-finetune/e2e-r{1,2}/data/train_raw.jsonl .work-finetune/e2e-r{1,2}/data/train_with_codes.jsonl
cat .work-finetune/e2e-r{1,2}/data/utt_*.wav …/ref_speaker.wav | sha256sum   (per run)
sha256sum .work-finetune/e2e_sft_launcher.py .work-finetune/e2e_reload_check.py
python3 -c 'json-parse both checkpoint config.json (tts_model_type, spk_id)'
grep -c e2e-r2 <r1 jsonl files> ; grep -c e2e-r1 <r2 jsonl files>
stat -c '%y %n' <driver logs, tee logs, driver scripts>
.venv/bin/python -m pytest -m 'not gpu' -q
```

## 6. Exit codes

- `pytest -m 'not gpu' -q` → exit 0 (267 passed, 38 deselected).
- Both `cmp` (archived evidence vs tee captures) → exit 0 (identical).
- All verifier git commands → exit 0; `status --porcelain` outputs empty for BOTH upstream worktrees.
- Recorded in-transcript (trusted only where independently corroborated): generator/prepare/sft-direct (PIPESTATUS[0])/launcher/reload exits = 0 for both runs — corroborated for the direct-train and reload phases by on-disk artifacts the verifier hashed/parsed itself, and for training by the loss lines in the still-present tee logs.

## 7. Runtime evidence inspected

- Checkpoints on disk NOW (all verified by the verifier):
  - `.work-finetune/e2e-r1/runs/checkpoint-epoch-{0,1}/` — `model.safetensors` 3,833,402,520 B each; epoch-1 sha256 `bf87ab9def9b…8734d8`
  - `.work-finetune/e2e-r1/runs-mem/checkpoint-epoch-1/` — `464fd883b899…dc8eb`
  - `.work-finetune/e2e-r2/runs/checkpoint-epoch-{0,1}/` — epoch-1 `4bd5f6924feb…730d`
  - `.work-finetune/e2e-r2/runs-mem/checkpoint-epoch-1/` — `308d5c6026b7…92e864`
  - Four-for-four distinct, every hash equal to the value claimed in the transcript/JSON; mtimes inside the respective run windows.
- Synthesis wavs parsed by the verifier from disk NOW (RIFF/WAVE, not the transcript's word):
  r1 `reload_synth_en01.wav`: PCM, mono, 24,000 Hz, 16-bit, 76,800 frames, **3.2000 s**, peak **0.3320**, rms **0.05800**, nonzero fraction **0.9894**;
  r2: PCM, mono, 24,000 Hz, 82,560 frames, **3.4400 s**, peak **0.1914**, rms **0.04274**, nonzero **0.9264**. All match claims.
- Dataset hashes recomputed (§3 row 1). Driver/tee logs byte-compared (§3 row 7).
- Environment facts re-confirmed live: pytest ran under the same `.venv`; upstream git states unchanged since the runs.

## 8. Regression tests

`.venv/bin/python -m pytest -m 'not gpu' -q` in the repo root, executed by the verifier:
**267 passed, 38 deselected in 16.23s** — exactly the expected counts. (GPU suite not
re-run by this verifier; GPU execution evidence comes from the two transcripts, whose
artifacts were independently re-verified on disk.)

## 9. Claims audit (implementer report vs independently established facts)

| Claim | Audit result |
|---|---|
| Both runs' five phase exits 0 | Consistent with transcripts; corroborated for train (tee logs + checkpoints + loss lines), prepare (jsonl on disk), reload (wavs on disk). |
| 12 optimizer steps/run; loss only at step 0 | Verified against `sft_12hz.py` source: per-microbatch `optimizer.step()`, `step % 10 == 0` logging. |
| Four distinct trained-weight hashes | Recomputed; 4/4 distinct, match claims. |
| Dataset regenerated fresh in each tree (not copied) | Supported: distinct jsonl hashes with in-tree paths, distinct render phases/timings/mtimes, zero cross-tree references; identical wav-byte hash is the deterministic-reseed signature (reseed 1234 documented in both dataset reports and consistent with the smoke flow). |
| Peak-mem 18.013/18.477 and 18.026/18.496 GiB from a real training process | Launcher source verified (`reset_peak_memory_stats` at start, `train()` verbatim, reads allocator stats in-process); values plausible vs the 34.36 GB GPU and consistent with the 2026-09-20 smoke's scale. |
| Only-arg-addition `--attn_implementation sdpa`; smoke-identical scale | Verified against the smoke transcript (1.7B Base / batch 2 / lr 2e-5 / 2 epochs) and the fix diff. |
| FIX worktree pristine before/after; HEAD unchanged | Re-verified NOW: porcelain empty, HEAD `48b8644`; log order `48b8644 → 0be0026 → 022e286` exact. |
| Transcripts machine-generated, never hand-edited | Supported: byte-identity with the tee captures, driver script generates every observed line, script mtimes precede run 1. |
| Evidence committed per brief Step 6.5 | Verified: commit `16c92cc` contains exactly the 4 evidence files + README rows; clean tree. |

## 10. Problems found

None that affect the verdict. Minor observations (all non-blocking, none weakening the proof):

1. The JSON `commands_and_exits` blocks render commands in repo-relative form while the
   transcripts show absolute paths; the transcripts (and driver script) are the
   authoritative, verbatim record — cosmetic inconsistency in the JSON summary only.
2. The brief's Step 6.1 template used `echo "EXIT=$?"` after the pipeline (which would
   report tee's status — the exact errata documented for the 2026-09-20 smoke). The
   implementer's `PIPESTATUS[0]` is strictly stronger; a beneficial, disclosed-in-place
   deviation.
3. `sox: not found` banner appears in prepare/sft stdout — benign (present also in the
   validated 2026-09-20 smoke), exits remain 0.
4. Non-weight checkpoint files carry Aug-27 mtimes and `.ok` markers — explained by the
   script's own `shutil.copytree(MODEL_PATH, …)` save path; the trained
   `model.safetensors` and `config.json` carry run-time mtimes.
5. The launcher adds a second ~19 s training per run for the memory reading — disclosed
   in-transcript and in both JSONs, authorized by the brief, and its outputs are
   quarantined in `runs-mem/` so the direct run's artifacts are untouched.

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence the verifier produced or re-derived
itself, not by trusting the implementer's prose: the committed transcripts hash to the
claimed digests and are byte-identical to the driver+tee captures; the exact direct
`python finetuning/sft_12hz.py … --attn_implementation sdpa` invocation with exit 0
(PIPESTATUS[0]) appears in both transcripts and its tee logs, loss lines, checkpoints
(distinct sha256 ×4, plausible 3.83 GB sizes) and synthesis wavs (24 kHz PCM, 3.20 s /
3.44 s, peaks 0.33/0.19, rms 0.058/0.043, ≥93 % nonzero samples — parsed from disk by
the verifier) all exist NOW and match every claimed number; independence is proven by
non-existent-then-fresh trees, distinct jsonl hashes with zero cross-tree references,
distinct render phases, sequential process timestamps, and source-level absence of any
checkpoint/trainer-state reuse; the only upstream delta is the minimal 15-line FIX diff
(default preserved) plus its test, with both upstream worktrees clean and the required
`48b8644 → 0be0026 → 022e286` history; the step/loss/precision arithmetic checks out
against the patched script's source; and the CPU suite passes with the exact expected
267/38 counts when run by the verifier. Falsification attempts (hand-edited transcripts,
copied datasets, reused trainer state, launcher substituting for the direct proof,
undisclosed upstream drift, silent/short/clipped audio) each failed to find support.
