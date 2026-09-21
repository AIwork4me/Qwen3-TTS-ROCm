# Task 1 Independent Release Verification — Pristine Double Reproduction

Program: Radeon Reference Closure v0.2, Task 1 (brief 1)
Verifier role: independent; assumes the implementer may be wrong; every criterion
re-derived from executable/runtime evidence gathered by the verifier, not accepted
from the implementer report.

## 1. Verdict

**PASS**

## 2. Exact commit / working-tree SHA reviewed

- Commit under review: `eaedaae81be4d53ea8b8fe32a6f62ece0f5d51bd`
  (`chore(evidence): reproduce Qwen fine-tuning attention failure twice from pristine upstream`)
  - Parent: `b0952a42d75f2bc34be6e9ea17e8256673052116` (verified via `git show --format='%H %P'`)
  - Commit time: 2026-09-21 13:24:39 +0800 (after both run logs' last-write times — see §7)
- Working tree at review time: `eaedaae81be4d53ea8b8fe32a6f62ece0f5d51bd`, `git status --porcelain` → 0 bytes (clean); reviewed content == committed content.
- Upstream clone pin: `git -C .upstream/Qwen3-TTS rev-parse HEAD` → `022e286b98fbec7e1e916cb940cdf532cd9f488e` (matches the pinned SHA required by the brief and Task 0).

## 3. Acceptance criteria checklist

| # | Criterion | Verdict | Independent evidence (all gathered by verifier) |
|---|---|---|---|
| 1 | Upstream pristine NOW (final state) | PASS | `git -C .upstream/Qwen3-TTS status --porcelain` → empty (0 bytes, exit 0); `rev-parse HEAD` → `022e286b…`; `grep -n flash_attention_2 …/sft_12hz.py` → line 51 `attn_implementation="flash_attention_2",` |
| 2 | Pristine recorded BEFORE and AFTER each run, in-transcript | PASS | Both transcripts embed before/after `status --porcelain \| wc -l` → `0` and `rev-parse HEAD` → `022e286b…` (run1 before: lines 28–33; run1 after: lines 457–461; run2 before: 28–33 of run2; run2 after: 887–891 per review-diff numbering / verified in files) |
| 3 | Two genuinely fresh processes with fresh output dirs | PASS | (a) In-transcript `ls .work-finetune/repro-rN` → "No such file or directory" before phase A in both; (b) two separate driver scripts on disk (`repro-r1-driver.sh`, `repro-r2-driver.sh`; r2 driver contains zero references to repro-r1/sft-run1 — grep exit 1); (c) per-run non-deterministic noise differs (MIOpen workspace pointers `0x7f51…` vs `0x7308…`, GPU use 8/7 %, VRAM 1429127168 vs 1431224320 B, distinct render timings) — consistent with two separate process lifetimes, not a copied artifact; (d) distinct file mtimes (r1 driver 13:20:51 → r1 log 13:22:12; r2 driver 13:22:47 → r2 log 13:24:03) |
| 4 | Distinct dataset prep runs, distinct jsonl outputs | PASS | 13 renders each (`[ref]`=1 + `[utt]`=12, grep-counted in both transcripts); `wc -l` on disk: `repro-r1/train_raw.jsonl` = 12, `repro-r1/train_with_codes.jsonl` = 12, same for repro-r2 (verifier-executed); dataset_report.txt present in both dirs |
| 5 | Same failure class both times: ImportError FlashAttention2 / flash_attn not installed, non-zero exit | PASS | Verifier grep: ImportError line at transcript line **386 in both files**; `sft-as-is-exit=1` at line **387 in both**; `generator-exit=0` (line 202) and `prepare-exit=0` (line 284) in both; traceback shape identical to the 2026-09-20 capture (sft_12hz.py:161 → :48 from_pretrained → transformers `_flash_attn_2_can_dispatch` raise at modeling_utils.py:2422) |
| 6 | Verbatim class match vs prior capture | PASS | Verifier read `finetune-smoke-2026-09-20.txt` line 1545 directly — byte-identical ImportError text; both transcripts also contain the in-transcript `diff <(grep -m1 …)` → `CLASS-MATCH-VERBATIM`; cross-run `cmp repro-r1/sft-run1-stdout.txt repro-r2/sft-run2-stdout.txt` → identical (verifier-executed; 51 lines each) |
| 7 | Commands differ in output-dir paths only, not in substance | PASS | Verifier diffed all `$ ` command lines from both transcripts after normalizing `repro-r1/repro-r2`, `sft-run1/sft-run2`, run labels → exit 0 (identical command sets). Args match the 2026-09-20 capture: dataset script defaults verified in source (`--manifest` default = `tests/data/finetune_manifest.json`, `--speaker` default = `serena` at scripts/make_finetune_dataset.py:138–141,48); prepare_data args (`--device cuda:0`, tokenizer path, in/out jsonl) identical to phase 4.4; sft args (`--init_model_path` 1.7B-Base, `--num_epochs 2`, `--speaker_name smoke_speaker`, script-default batch 2 / lr 2e-5) identical to phase 4.5 attempt 1 |
| 8 | Transcripts complete with headers, never hand-edited | PASS | Headers verified to contain: GPU (rocm-smi Card Model 0x1586 / "AMD Radeon 8060S Graphics" via torch), gfx (`gfx1151`), ROCm (torch `2.12.0+rocm7.14.0`, hip `7.14.60850`), Python 3.12.3, qwen-tts 0.1.1 (plus transformers 4.57.3, accelerate 1.12.0, flash-attn NOT INSTALLED), upstream SHA, exact commands, exit codes; rocm-smi snapshots wrap every GPU phase (A/B/C before+after). Anti-tamper: evidence files byte-identical (`cmp`) to the tee logs; committed blobs at `eaedaae` (sha256 `0504afa7…`, `6db78216…`) equal on-disk files and the report's claimed hashes; log mtimes (13:22:12 / 13:24:03) precede commit time (13:24:39); the r1/r2 driver printf blocks reproduce the transcript headers verbatim (checked directly, unfiltered) |
| 9 | No source modification anywhere in upstream | PASS | Upstream porcelain empty (verifier, final state); in-transcript before/after zeros; the commit changes exactly 3 downstream paths — 2 new evidence files + `evidence/README.md` +2 rows (`git show --stat` verified: 850 insertions, 0 deletions, no upstream paths; upstream clone is a separate git repo anyway) |
| 10 | CPU regression suite green | PASS | Verifier-executed `.venv/bin/python -m pytest -m 'not gpu' -q` in repo root → **`267 passed, 38 deselected in 15.67s`, exit 0** — exactly the expected 267/38 |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-1-brief.md` (full)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-1-report.md` (full)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-b0952a4..eaedaae.diff` (full, 900 lines)
- `evidence/upstream-372-pristine-failure-run1.txt` (424 lines, via review diff + direct greps)
- `evidence/upstream-372-pristine-failure-run2.txt` (424 lines, via review diff + direct greps)
- `evidence/finetune-smoke-2026-09-20.txt` (phases 4.3/4.4/4.5 regions + line 1545 region)
- `evidence/README.md` (committed rows via `git show`)
- `.work-finetune/repro-r1-driver.sh`, `.work-finetune/repro-r2-driver.sh` (full / header+goal blocks)
- `.work-finetune/repro-r1/`, `.work-finetune/repro-r2/` artifacts (listing, line counts, stdout cmp)
- `scripts/make_finetune_dataset.py` (argparse defaults, lines 138–141 and 48)
- `.upstream/Qwen3-TTS/finetuning/sft_12hz.py` (grep line 51 only, read-only)

## 5. Exact commands executed (by the verifier)

```
git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS status --porcelain          # empty
git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS rev-parse HEAD              # 022e286b…
grep -n flash_attention_2 .upstream/Qwen3-TTS/finetuning/sft_12hz.py                    # 51
git rev-parse HEAD; git log --oneline -3; git status --porcelain                        # eaedaae, clean
git show --stat --format='%H %P %s' eaedaae                                             # 3 files, +850
git show -s --format='commit-time=%ci' eaedaae                                          # 13:24:39 +0800
sha256sum evidence/upstream-372-pristine-failure-run{1,2}.txt                           # match report
git show eaedaae:evidence/upstream-372-pristine-failure-run{1,2}.txt | sha256sum        # match disk
grep -n "ImportError: FlashAttention2" evidence/upstream-372-pristine-failure-run{1,2}.txt
grep -nE "exit=" evidence/upstream-372-pristine-failure-run{1,2}.txt
diff <(grep '^\$ ' run1 | sed s/repro-r1/repro-rN/…) <(grep '^\$ ' run2 | sed s/repro-r2/repro-rN/…)   # exit 0
diff <(sed-normalized run2 transcript) <(sed-normalized run1 transcript)                # noise-only deltas
cmp evidence/run1.txt .work-finetune/repro-r1.log; cmp evidence/run2.txt .work-finetune/repro-r2.log  # identical
cmp .work-finetune/repro-r1/sft-run1-stdout.txt .work-finetune/repro-r2/sft-run2-stdout.txt           # identical
wc -l repro-r{1,2}/{train_raw,train_with_codes}.jsonl sft-run{1,2}-stdout.txt          # 12/12/12/12/51/51
grep -c '^\[ref\]|^\[utt\]|^\[render\]' on both transcripts                            # 1/12/13 both
ls -la --time-style=full-iso drivers + logs (mtime chronology)
grep -nE 'repro-r1|sft-run1' repro-r2-driver.sh                                         # no hits (exit 1)
grep -n 'default|finetune_manifest|serena' scripts/make_finetune_dataset.py
time .venv/bin/python -c 'import torch'                                                 # 1.35 s (timeline calibration)
.venv/bin/python -m pytest -m 'not gpu' -q                                              # 267 passed, 38 deselected
```

## 6. Exit codes (verifier-observed)

- Upstream `status --porcelain`: exit 0, output empty (0 bytes).
- `rev-parse HEAD` (upstream): exit 0 → `022e286b98fbec7e1e916cb940cdf532cd9f488e`.
- `grep flash_attention_2`: exit 0 → line 51.
- All `cmp` byte-identity checks: exit 0 (evidence==logs; run1-sft==run2-sft).
- Normalized command-set diff run1 vs run2: exit 0.
- r2-driver contamination grep: exit 1 (no r1 references) — expected.
- Pytest: exit 0 (`267 passed, 38 deselected`).
- In-evidence recorded exits (verifier re-read from transcripts): `generator-exit=0`,
  `prepare-exit=0`, `sft-as-is-exit=1` in BOTH runs (lines 202/284/387 of each file).

## 7. Runtime evidence inspected

- Transcript line 386 both files: `ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed. Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2.`
- Line 1545 of `finetune-smoke-2026-09-20.txt`: byte-identical prior-capture line (verifier read it directly).
- rocm-smi snapshots: product (0x1586, gfx1151), use %, VRAM totals wrap phases A/B/C before and after in both transcripts; values differ per run (authentic separate captures).
- Filesystem chronology: r1 driver 13:20:51.635 → r1 log (header 13:20:58, last write) 13:22:12.278 → r2 driver 13:22:47.974 → r2 log (header 13:22:49, last write) 13:24:03.040 → commit 13:24:39. No file predates its cause; nothing modified after commit (tree clean).
- Timeline plausibility: ~74 s per run is consistent with 53.6 s of recorded per-render time (sum of the 13 `[render] took=` values), ~1.35 s interpreter+torch startup (verifier-measured), warm model/MIOpen caches.
- Environment header values in both transcripts: python 3.12.3, torch 2.12.0+rocm7.14.0, hip 7.14.60850, GPU "AMD Radeon 8060S Graphics", qwen-tts 0.1.1, transformers 4.57.3, accelerate 1.12.0, flash-attn NOT INSTALLED, upstream SHA `022e286b…`.

## 8. Regression tests

`.venv/bin/python -m pytest -m 'not gpu' -q` (repo root, verifier-executed after the review):
**267 passed, 38 deselected in 15.67s — exit 0.** Matches the expected CPU-suite baseline exactly; no regressions introduced by the commit (which touched only evidence files + README index rows).

## 9. Claims audit (implementer report vs verifier findings)

| Claim | Result |
|---|---|
| Commit `eaedaae`, parent `b0952a4`, not pushed, branch main | Confirmed (SHA, parent; `git log` shows local main at eaedaae) |
| 424 lines each; sha256 `0504afa7…` / `6db78216…` | Confirmed (verifier sha256 of disk files and of committed blobs) |
| Evidence = byte-identical `cp` of tee logs | Confirmed (`cmp` both pairs) |
| 13 renders (1 ref + 12 utts), 12-line jsonl both runs | Confirmed (grep counts; on-disk `wc -l`) |
| Failure at transcript line 386 in both; exit 1 both | Confirmed (verifier grep: 386/387 both) |
| `RUN1-RUN2-SFT-STDOUT-IDENTICAL` | Confirmed (verifier `cmp`) |
| `CLASS-MATCH-VERBATIM` vs 2026-09-20 line 1545 | Confirmed (verifier read line 1545; text identical) |
| Pristine table (0 lines / same SHA before+after each run, final) | In-transcript records verified; final state independently re-verified |
| "Separate bash <driver> invocations" | Supported: two driver files, no cross-references, disjoint mtimes, per-run noise differences |
| "Same defaults as 2026-09-20; only output dirs redirected" | Confirmed against argparse defaults and the phase 4.3/4.4/4.5 command lines |
| Disclosed deviation: `tee` + `PIPESTATUS[0]` instead of the smoke's `tail -40` + wrapper exit | Accurate and disclosed; the recorded exit is the python process's own (1) |
| Downstream diff at commit = 2 evidence files + 2 README rows | Confirmed (`git show --stat`: 3 files, 850 insertions, 0 deletions) |

No claim was found to be false, inflated, or unsupported.

## 10. Problems found

None blocking. Observations (none affect the verdict):

1. **cwd of phase C differs cosmetically from 2026-09-20**: the smoke ran `sft_12hz.py` from `.upstream/Qwen3-TTS/finetuning`, the repro from the upstream repo root with absolute path args. Immaterial: the script consumes only its explicit path arguments, and the failure occurs inside `from_pretrained` (sft_12hz.py:48) before any data or output path is touched; traceback and error text are identical.
2. **Run duration is tight (~74 s each)** but internally consistent: 53.6 s of in-transcript render time + 1.35 s interpreter/torch startup (verifier-measured) + warm-cache model loads and a construction-time failure that precedes weight loading. The two runs agree; nothing depends on this timing.
3. The 2026-09-20 capture's `as-is-run-exit=0` wrapper-exit errata is properly avoided here by `PIPESTATUS[0]`; this is a disclosed capture-mechanics improvement, not an invocation change.

## 11. Why PASS is justified

Every acceptance criterion of Step 1.5 is backed by evidence the verifier produced or
re-derived itself, not by trusting the implementer: the pristine invariant was re-executed
now (empty porcelain, pinned SHA `022e286b…`, root-cause line 51 still present); both
evidence files were independently grepped (ImportError at line 386, `sft-as-is-exit=1`,
`generator-exit=0`, `prepare-exit=0` in both); the two runs were shown to execute the
identical command set modulo the fresh output-dir tokens (normalized command diff exit 0),
from separate driver processes with disjoint artifacts, fresh dirs (in-transcript
pre-existence check), genuinely re-rendered datasets (13 renders, distinct timings,
distinct MIOpen memory maps), and 12-line jsonl outputs verified on disk; the failure
class matches the 2026-09-20 capture byte-for-byte (line 1545 read directly; cross-run
sft stdout byte-identical via `cmp`); anti-tamper checks all hold (evidence == tee logs,
committed blobs == disk, mtimes precede commit, drivers regenerate headers verbatim);
the commit modifies only the two evidence files plus two README index rows; and the CPU
suite re-run by the verifier is green at exactly the expected 267 passed / 38 deselected.
Attempts to falsify (command-substance diff, driver cross-contamination grep, archive
tamper probes, timeline chronology, invocation-vs-prior-capture comparison) all came back
clean. The disclosed deviations (fresh output paths; tee/PIPESTATUS capture mechanics)
are permitted by the brief's fresh-dirs rule and improve on a documented errata without
changing the invocation's substance.
