# rc02 Task 5 — Independent Release Verification (Round A loader validation)

Verifier role: independent falsifier. Every acceptance criterion was tested against
runtime/executable evidence gathered in this session, not against the implementer's prose.

## 1. Verdict

**PASS**

## 2. Exact commit / working-tree SHA reviewed

- Reviewed commit: `47d83fedb59dd75b7aa5a1b49f65a1a2b4ab14db` (`47d83fe`) on `main` (branch
  verified via `rev-parse --abbrev-ref HEAD`).
- Working tree matches the commit exactly: `git diff 47d83fe -- <evidence files + README>`
  is empty; `git status --porcelain` is empty.
- Pre-amend (superseded) commit `3cb1aa18c19e343d647c87fa72e2b5ab98652e7c` verified to
  exist, same parent (`5bfd44834e51e8d431a03add3c638c606bfb9b7a`) and same message as
  `47d83fe` (a true `--amend`), not an ancestor of HEAD, contained in no branch.
- FIX worktree input: `48b8644aac8512e6fdc0f4442788bf399c425fdb` on
  `fix/finetuning-attn-implementation`, `status --porcelain` empty (re-verified live).

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | 3 distinct fresh-process invocations for BOTH verbatim and supplement sections | PASS | Verbatim starts 14:21:38 / 14:21:50 / 14:21:58; supplement starts 14:26:27 / 14:26:41 / 14:26:51 — six distinct timestamps, each run labeled "RUN N OF 3" with its own env block and its own rocm-smi BEFORE/AFTER pair whose VRAM values chain run-to-run (e.g. run1-after 1406259200 = run2-before; run2-supp-after 1407172608 = run3-supp-before). Each python invocation prints the qwen_tts import-time banner exactly once (fresh-process indicator). No PIDs are recorded in the transcripts, but the criterion's timestamp/snapshot alternatives are satisfied. |
| 2 | Verbatim per run: parsed=sdpa, config._attn_implementation=sdpa, cuda_available=True, real device, ROUND_A_LOAD_OK, exit 0 | PASS | grep counts per file: `parsed attn_implementation=sdpa` 1 (print) each, `config._attn_implementation=sdpa` 1 print each (+1 supplement-header prose), `cuda_available=True` 1 each, `device=AMD Radeon 8060S Graphics` 1 each, `ROUND_A_LOAD_OK` 1 print each (+2 prose), `python-exit=0` 1 real exit line each (+1 prose), closeout `RUN_N_EXIT_0` 1 each. All in files 1, 2, 3. |
| 3 | Supplement per run: config=sdpa, param_device=cuda:0, mem_alloc_GiB>0 (3.91), ROUND_A_LOAD_OK, exit 0, placement via device_map | PASS | grep counts per file: `placement_method=device_map` 1 each (fallback never taken), `param_device=cuda:0` 1 print each, `mem_alloc_GiB=3.91` 1 each (>0), `ROUND_A_LOAD_OK` print, `python-exit=0`, closeout `RUN_N_SUPPLEMENT_EXIT_0` 1 each. |
| 4 | No FlashAttention ImportError in any run | PASS | Only "ImportError" strings in all three files are the wrapper's own header prose (lines 8 and 123 per file: "Expected: no FlashAttention ImportError"). The per-process "Warning: flash-attn is not installed. Will only run the manual PyTorch version" banner (lines 72 and 181 per file) is the benign qwen-tts import warning; every one of the 6 processes continued past it to `ROUND_A_LOAD_OK` with `python-exit=0`. Grep for `traceback|error|fail|abort|exception` (case-insensitive, banner-filtered) matches nothing outside the same prose lines. |
| 5 | Verbatim sections not retro-edited (hash chain + internal consistency) | PASS | `head -111` of each current file hashes EXACTLY to the pre-append blob hashes cited in the report/README: d902f7a9…, e6f520b1…, 48e8f2fb… (independently confirmed against the blobs stored in pre-amend commit 3cb1aa1). Current full-file hashes 5ab46949… / 7d1f0050… / 5e3e9cfc… match both the README rows and the working tree. `git diff --numstat 3cb1aa1 47d83fe`: 111 added / 0 deleted per evidence file — pure append. Internal consistency: supplement closeout timestamps equal the evidence files' filesystem mtimes (14:26:34/48/58); driver/script mtimes precede their run starts; VRAM chain across runs is coherent. |
| 6 | Fix-round bookkeeping: amend recorded; 47d83fe on main; tree clean | PASS | Amend choice recorded in task-5-report.md ("choice recorded: amend, permitted because nothing was pushed"); fix-round supplement ruling recorded in progress.md line 38. `git log --oneline -3` shows 47d83fe (Round A) on main over 5bfd448; `git status --porcelain` empty; 3cb1aa1 confirmed unreachable (superseded by amend). |
| 7 | Downstream CPU suite green | PASS | Ran myself this session: `.venv/bin/python -m pytest -m 'not gpu' -q` → **267 passed, 38 deselected in 15.88s**, exit code 0. Exactly the expected 267/38. `gpu` marker is registered in pyproject.toml `[tool.pytest.ini_options]`. |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-5-brief.md` (brief 6A)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-5-report.md`
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-5bfd448..47d83fe.diff` (full 725-line read)
- `evidence/upstream-372-round-a-loads1.txt`, `-loads2.txt`, `-loads3.txt` (222 lines each; full content read via the diff and spot-read on disk)
- `evidence/README.md` (rows 65–67, both hash generations)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/progress.md` (rulings, lines 34, 36–39)
- `.work-finetune/round_a_load_check.py` (gitignored driver, read in full)
- `.work-finetune/round_a_load_check_gpu.py` (gitignored supplement driver, read in full)
- `.work-finetune/round_a_driver.sh`, `.work-finetune/round_a_driver_gpu.sh` (wrapper shells, read in full)
- `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py` (patched module: `--attn_implementation` argparse arg, `from_pretrained(..., attn_implementation=args.attn_implementation)`)
- `pyproject.toml` (pytest markers)

## 5. Exact commands executed (by me, this session)

1. `sha256sum evidence/upstream-372-round-a-loads{1,2,3}.txt` (+ `wc -l`)
2. `ls -la` + `sha256sum .work-finetune/round_a_load_check.py .work-finetune/round_a_load_check_gpu.py`
3. `git -C /home/amd/Desktop/Qwen3-TTS-ROCm log --oneline -3`
4. `git -C /home/amd/Desktop/Qwen3-TTS-ROCm status --porcelain`; `git rev-parse HEAD`; `git rev-parse --abbrev-ref HEAD`
5. `git cat-file -t 3cb1aa1…`; `git diff --numstat 3cb1aa1 47d83fe`
6. `git log -1 --format='%H%n%P%n%s' 3cb1aa1` and `47d83fe`; `git merge-base --is-ancestor 3cb1aa1 HEAD`; `git branch -a --contains 3cb1aa1`
7. `grep -n '2026-09-21T14:2'` + run-header greps over the three evidence files
8. Marker-count greps (11 patterns × 3 files) for parsed/config/cuda_available/device/ROUND_A_LOAD_OK/python-exit=0/placement_method/param_device/mem_alloc_GiB=3.91/RUN_N_EXIT_0/RUN_N_SUPPLEMENT_EXIT_0
9. `grep -n 'ImportError'` and `grep -n -i 'traceback|error|fail|abort|exception'` over the three evidence files
10. `git show 3cb1aa1:evidence/…loadsN.txt | sha256sum` (×3) and `git diff 47d83fe -- <4 files> | wc -l`
11. `head -111 evidence/…loadsN.txt | sha256sum` (×3)
12. `git -C .upstream/Qwen3-TTS-fix rev-parse HEAD`; `status --porcelain | wc -l`; `grep -n attn_implementation …/sft_12hz.py`
13. `stat -c '%y %n'` on evidence files and drivers
14. `.venv/bin/python -m pytest -m 'not gpu' -q` (in /home/amd/Desktop/Qwen3-TTS-ROCm)

## 6. Exit codes

- pytest: 0 (267 passed, 38 deselected)
- `git status --porcelain`: exit 0, empty output (tree clean)
- all `git` inspection commands: 0
- all `sha256sum`/`grep`/`stat`/`ls` commands: 0 (greps that legitimately found zero non-prose matches: ImportError prose-only, error-scan prose-only)
- recorded in-transcript exits (not mine): `python-exit=0` ×6, closeouts `RUN_{1,2,3}_EXIT_0` and `RUN_{1,2,3}_SUPPLEMENT_EXIT_0`

## 7. Runtime evidence inspected

- Six python invocations across three transcripts (3 verbatim CPU-resident + 3 supplement GPU-resident), each with env pinning (downstream HEAD, FIX worktree HEAD 48b8644 + pristine count 0, driver sha256, model path) and rocm-smi BEFORE/AFTER.
- Verbatim runs print `mem_alloc_GiB=0.00` — CPU-resident by construction (no `device_map` in the brief's verbatim call), exactly as disclosed; the binding supplement ruling covers GPU residency.
- Supplement runs print `placement_method=device_map` (wrapper accepted `device_map="cuda:0"`; `.to()` fallback never taken), `param_device=cuda:0`, `mem_alloc_GiB=3.91` (bf16 1.7B weights on cuda:0 via the caching allocator).
- Driver integrity: on-disk `round_a_load_check.py` sha256 `fa87b054…` and `round_a_load_check_gpu.py` sha256 `a6cd725a…` match the hashes recorded inside every transcript header.
- VRAM values chain coherently across consecutive runs (run-after ≈ next-run-before), and evidence-file mtimes equal the recorded supplement closeout times — strong authenticity signals against fabrication/retro-editing.

## 8. Regression tests

- `.venv/bin/python -m pytest -m 'not gpu' -q` → 267 passed, 38 deselected, exit 0 (expected 267/38 — exact match).
- Marker sanity: `gpu` marker registered in `pyproject.toml`; the 38 deselected are the GPU suite, not silent collection loss.

## 9. Claims audit (implementer report vs. verified fact)

| Claim | Verified |
|---|---|
| Final commit 47d83fe on main, parent 5bfd448, tree clean, not pushed | Yes (git log/status/rev-parse; no remote ref consulted beyond local) |
| 3cb1aa1 created then amended (--amend --no-edit), now unreachable | Yes (same parent + message; not ancestor; in no branch) |
| Verbatim transcripts are byte-untouched prefixes of the supplemented files | Yes (head -111 sha256 == pre-append blob sha256 in 3cb1aa1; numstat 111/0) |
| README rows carry both hash generations | Yes (rows 65–67; both sets re-hashed) |
| Driver sha256s recorded in transcripts match on-disk drivers | Yes (fa87b054…, a6cd725a…) |
| Drivers differ only by device_map (+fallback) and param_device print | Yes (both files read in full; statement accurate) |
| Parse line deviates from brief only to supply required dummy args | Yes (`--train_jsonl` is `required=True` at sft_12hz.py:35; Task 3 ruling in progress.md line 34 mandates carrying this fact to Task 4/5) |
| placement_method=device_map in all 3 supplement runs; fallback unused | Yes (1 occurrence each; no "device_map rejected"/"to_after_load" strings) |
| Only ImportError strings are wrapper prose; flash banner is benign | Yes (lines 8/123 per file are prose; banner once per fresh process; all runs reached OK + exit 0) |
| Supplement ran at downstream HEAD 3cb1aa1 (pre-amend) | Yes (recorded in each supplement section; chronologically consistent with amend afterwards) |
| CPU suite 267 passed / 38 deselected | Yes (re-run independently, exact match) |
| FIX worktree 48b8644 pristine | Yes (re-verified live: 0 porcelain lines; configurable attn_implementation present) |

## 10. Problems found

None that affect the verdict. Non-blocking observations:

1. **No PIDs in transcripts.** Criterion 1 lists "timestamps/PIDs/rocm-smi snapshots" as alternatives; only timestamps and snapshots are present. Distinctness is nevertheless proven (six distinct timestamps, per-run rocm-smi pairs, import banner once per process, chained VRAM), so the criterion's intent is met.
2. **Verbatim parse line deviates from the brief's literal text** (dummy `--init_model_path/--train_jsonl/--output_model_path`). Necessary — upstream's parser makes the brief's literal line exit 2 — disclosed in the report and header prose, and covered by the recorded Task 3 ruling. Not a defect of the implementation.
3. **Verbatim `mem_alloc_GiB=0.00`.** Inherent to the plan's own verbatim driver (no `device_map`); disclosed, and GPU residency is satisfied by the ruled supplement. The verifier's criterion 2 correctly does not gate verbatim on mem>0.
4. Supplement AFTER rocm-smi snapshots show no 4 GiB VRAM spike — correctly explained (python process has exited before the AFTER snapshot); residency is proven in-process via `param_device=cuda:0` + `mem_alloc_GiB=3.91`, which is what criterion 3 requires.

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence I independently executed or re-derived, not by trusting the implementer: (1) six distinct invocations demonstrated via distinct timestamps, chained rocm-smi snapshots and per-process import banners; (2)–(3) all per-run success markers grep-verified with exact expected counts in all six runs, including `placement_method=device_map`, `param_device=cuda:0` and `mem_alloc_GiB=3.91` in every supplement run; (4) zero genuine ImportError/Traceback occurrences — only wrapper prose and the benign banner, with all processes exiting 0; (5) non-retro-editing proven cryptographically (head-111 of each file equals the pre-amend blob hash in 3cb1aa1; numstat 111/0; README carries both generations; mtimes corroborate); (6) amend bookkeeping verified in git topology (same parent/message, 3cb1aa1 unreachable) and recorded in the report and progress.md; (7) the CPU suite re-run by me reproduces exactly 267 passed / 38 deselected / exit 0. No falsification attempt succeeded.
