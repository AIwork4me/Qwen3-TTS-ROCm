# rc02 Task 2 — independent release verification (Radeon Reference Closure v0.2)

Verifier role: independent falsifier. Implementer's claims were checked against
source, git state, transcripts, raw per-process streams, and on-disk artifacts;
every mandated command was executed by the verifier, not trusted from the report.

## 1. Verdict

**PASS.**

## 2. Exact commit / working-tree SHA reviewed

- Reviewed commit: `12dfe67` (`12dfe6774da010c73bb810b9ec39666d4fbe0711`, full SHA) —
  "docs(upstream): record Qwen3-TTS #372 root cause and controlled isolation experiment",
  parent `5a32ac8`. Commit touches exactly 2 files (`evidence/README.md` +1,
  `evidence/upstream-372-root-cause.md` +430, new) per `git show --stat`.
- Working tree at review start: `git status --porcelain` → empty (clean), HEAD on
  `12dfe67` (branch main).
- Evidence file on disk byte-matches the review diff content (430 lines).
- Upstream pinned SHA under test everywhere: `022e286b98fbec7e1e916cb940cdf532cd9f488e`.

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | All 8 brief questions answered with file:line citations | **PASS** | evidence/upstream-372-root-cause.md §2 answers all 8; ~20 citations independently re-opened by verifier (see §9). Two near-nits, non-load-bearing (see §10) |
| 2 | Isolation diff exactly one token; no other worktree change | **PASS (cryptographic)** | Blob-hash reproduction: pinned-commit blob of `finetuning/sft_12hz.py` = `c1f3f4684f3f53927a660f06e58beb6a4107f89a`; verifier's one-token reconstruction (`sed '51s/flash_attention_2/sdpa/'` of the pristine file) hashes to `b5eae64e02b3e67fd831b02ba45778b61137a55c` — both match the recorded diff header `index c1f3f46..b5eae64` in iso-driver.log:37. Pre-run `git status --porcelain` in the transcript shows the worktree's only delta was ` M finetuning/sft_12hz.py` (no other modified or untracked files). A one-hunk/one-line/one-token diff with matching content hash is the strongest reproducible proof the experiment carried nothing else |
| 3 | Full chain in FIX worktree | **PASS** | Fresh dir proven (log line 50: `iso/` did not pre-exist); prepare-exit=0; sft-iso-exit=0 with loss lines `Epoch 0 Loss 13.2097` / `Epoch 1 Loss 11.9999` present in BOTH the driver log and the independent raw stream `iso/sft-iso-raw.log`; both checkpoints on disk with `model.safetensors` = 3,833,402,520 B each (matches claim digit-for-digit; mtimes 13:41:53/13:42:00 match); official-API reload `_attn_implementation='sdpa'` (root + talker) in `iso/reload-raw.log`; audio independently re-parsed by verifier: mono 16-bit sr=24000, 96,000 frames = 4.0000 s, peak 0.2275, rms 0.03978, 88,016/96,000 nonzero samples — finite, non-silent, 24 kHz, matching the claimed WAV-SANE line exactly |
| 4 | GPU residency evidence real (in-process) | **PASS** | `reload-raw.log` records in-process `torch.cuda.get_device_name(0): AMD Radeon 8060S Graphics` and `mem_get_info free=75.84 GiB total=80.00 GiB`; both prints are computed live in `iso_reload_check.py:35-40` (audited: no hard-coded success strings anywhere in the script; all values derived from the loaded model/audio, with asserts) |
| 5 | Worktree restored pristine | **PASS (verifier-executed)** | `git -C .upstream/Qwen3-TTS-fix status --porcelain` → empty; `rev-parse HEAD` → `022e286b98fbec7e1e916cb940cdf532cd9f488e`; branch → `fix/finetuning-attn-implementation`; line 51 → `attn_implementation="flash_attention_2",` (pristine again); pristine clone `.upstream/Qwen3-TTS` status → 0 lines |
| 6 | Alternative hypotheses + remaining uncertainties present and substantive | **PASS** | §5: five alternative hypotheses (architecture coupling; second hidden dependency; data/schema; version skew; install-flash-attn-instead) each rejected with cited mechanism or empirical counter; §6: five uncertainty items (execution-only scope, single scale/model, flash-installed not tested, APU VRAM-counter caveat, watcher under-sampling) — substantive, not boilerplate |
| 7 | Two disclosed errata honest, not undermining | **PASS** | Phase-A `generator-exit=0` label and missing phase-A raw side-log both disclosed at the end of the evidence doc. Phase-A success is instead proven by artifacts the verifier confirmed on disk (12 utt wavs, `train_raw.jsonl`, `ref_speaker.wav`, `dataset_report.txt` with 12 rows) and by downstream phases consuming them (prepare → 12 coded lines → training ran). Phases B/C/C2/D captured true process exits. Nothing undermines the success proof |
| 8 | Root-cause claim properly scoped, no overclaim | **PASS** | Statement: "a loader-configuration choice hard-coded in the fine-tuning script (`sft_12hz.py:51`), not a capability of the model, not a data/schema problem, and not a second hidden flash-attn dependency" — matches the traceback mechanism (init-time dispatch check `modeling_utils.py:2422`, before any kernel) and the demonstrated sdpa run; §6 explicitly bounds it ("the claim proven is 'sdpa removes the blocker', not 'flash_attention_2 would also work if a build existed'"; single scale/model; execution-scope only). No overclaim found |

## 4. Files reviewed

- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-2-brief.md`
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-2-report.md`
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-5a32ac8..12dfe67.diff`
- `evidence/upstream-372-root-cause.md` (primary evidence, 430 lines)
- `evidence/reference-closure-ground-truth-2026-09-21.md` (§8 verbatim sft_12hz.py, §9 line-51 grep)
- `evidence/upstream-372-pristine-failure-run1.txt` (ImportError at line 386; `sft-as-is-exit=1`; `CLASS-MATCH-VERBATIM`)
- `evidence/upstream-372-pristine-failure-run2.txt` (existence + `sft-as-is-exit=1` line 387)
- `evidence/finetune-smoke-2026-09-20.txt` (blob pair line 1556; 18.024 GiB line 1597; smoke losses 13.6470/15.0211; `RELOAD-AND-SYNTHESIS-OK` line 1638)
- `.work-finetune/iso-driver.log` (311 lines; sha256 verified) and raw streams `iso/sft-iso-raw.log`, `iso/probe-raw.log`, `iso/reload-raw.log`
- `.work-finetune/iso_reload_check.py` (audited for hard-coded prints — none)
- `.work-finetune/iso/` artifacts (checkpoints, wav, jsonl, dataset report)
- Upstream sources: `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py`, `dataset.py`, `prepare_data.py`, `README.md`, `finetuning/README.md`, `examples/*`
- Installed package: `qwen_tts/inference/qwen3_tts_model.py`, `qwen_tts/cli/demo.py`, `qwen_tts/core/models/modeling_qwen3_tts.py`, `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`, `qwen_tts/core/tokenizer_25hz/vq/whisper_encoder.py`, `transformers/modeling_utils.py`
- Downstream: `src/qwen3_tts_rocm/loader.py`, `.gitignore`

## 5. Exact commands executed (by the verifier)

1. `git rev-parse HEAD` / `git status --porcelain` / `git log --oneline -5` (repo root)
2. `git show --stat --oneline 12dfe67`
3. `git -C .upstream/Qwen3-TTS-fix status --porcelain`
4. `git -C .upstream/Qwen3-TTS-fix rev-parse HEAD`
5. `git -C .upstream/Qwen3-TTS-fix rev-parse --abbrev-ref HEAD`
6. `grep -n 'flash_attention_2' .upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py`
7. `git -C .upstream/Qwen3-TTS status --porcelain | wc -l`
8. `git -C .upstream/Qwen3-TTS worktree list`
9. `sha256sum .work-finetune/iso-driver.log` + `wc -l`
10. `git -C .upstream/Qwen3-TTS-fix rev-parse 022e286b...:finetuning/sft_12hz.py` (pristine blob)
11. `sed '51s/flash_attention_2/sdpa/' <pristine sft_12hz.py> > /tmp/sft_one_token.py` + `git -C .upstream/Qwen3-TTS-fix hash-object /tmp/sft_one_token.py` + `diff pristine /tmp/sft_one_token.py`
12. `.venv/bin/python` (stdlib wave/struct) re-parse of `reload_synth_en01.wav` + `json.load` of checkpoint `config.json`
13. `sed -n` over every cited line range (sft_12hz.py 34-42/48-52/55; qwen3_tts_model.py 83-86/100-101/112; demo.py 104-108/605-612; modeling_qwen3_tts.py 788-789/941-942/1872-1874/1888; tokenizer_v2 153/917; whisper_encoder.py 26-40; loader.py 81/180-184/201-224; modeling_utils.py 2076/2686/2714/2422)
14. `grep -rin 'flash' .upstream/Qwen3-TTS-fix/finetuning/`
15. `grep -n 'flash_attention_2' README.md / finetuning/README.md / examples/*.py` (upstream)
16. `grep -cin 'flash\|attn' .upstream/Qwen3-TTS-fix/finetuning/prepare_data.py` → 0
17. `.venv/bin/python -c "from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS; ..."` → `sdpa` present
18. Greps over `evidence/upstream-372-pristine-failure-run1.txt`, `-run2.txt`, `evidence/finetune-smoke-2026-09-20.txt` (ImportError, exits, blob pair, 18.024 GiB, CLASS-MATCH)
19. `ls`/`find` over `.work-finetune/iso/` and `output/checkpoint-*/` (existence + sizes)
20. `.venv/bin/python -m pytest -m 'not gpu' -q` (repo root)

## 6. Exit codes

- All git read commands: exit 0.
- `status --porcelain` on FIX worktree: exit 0, **empty output** (pristine).
- `hash-object` of one-token reconstruction: exit 0 → `b5eae64e02b3...` (matches recorded diff header).
- `.venv/bin/python` wav/config parse: exit 0.
- `pytest -m 'not gpu' -q`: **exit 0** — `267 passed, 38 deselected in 16.05s` (exactly the expected counts).

## 7. Runtime evidence inspected

- `iso-driver.log` (sha256 `05be3ad742b3b3038ebacb1c77ffcba585d84ca30eb84f29092766aef038bdef`, 311 lines — both match the report's claims). Contains: pre-run worktree state (single modified file + full one-hunk diff), fresh-dir proof, per-phase rocm-smi snapshots, watcher samples (8% → 35% busy), phase exits (`prepare-exit=0`, `sft-iso-exit=0`, `probe-exit=0`, `reload-exit=0`, `config-check-exit=0`, `checkout-exit=0`), loss lines, checkpoint listing, closeout pristine checks.
- Raw per-process streams (`sft-iso-raw.log`, `probe-raw.log`, `reload-raw.log`) corroborate the driver-log excerpts verbatim (same losses, same WAV-SANE digits, same device/mem_get_info lines) — the filtered main log was not embellished. `reload-raw.log` additionally retains MIOpen workspace warnings the driver log filtered, consistent with the disclosed filtering policy.
- On-disk: both `model.safetensors` = 3,833,402,520 B (claim exact); `reload_synth_en01.wav` = 192,044 B = 44-byte header + 96,000 samples × 2 B; independent parse: sr 24000, 4.0000 s, peak 0.2275, rms 0.03978, 88,016 nonzero samples; checkpoint `config.json`: `tts_model_type: custom_voice`, `talker_config.spk_id: {'smoke_speaker': 3000}`, `model_type: qwen3_tts`, architectures `['Qwen3TTSForConditionalGeneration']`.
- Control legs: run1/run2 transcripts carry the verbatim `ImportError: FlashAttention2 ... flash_attn seems to be not installed` at run1 line 386 (traceback through `modeling_utils.py:2714` → `:2422`, matching the causal-chain citation), `sft-as-is-exit=1` in both, `CLASS-MATCH-VERBATIM` diffs against the 2026-09-20 capture.
- `iso_reload_check.py` audited line-by-line: every `[iso-reload]` print is computed from the live model or waveform, with `assert sr == 24000`, `assert finite`, `assert nonzero`, `assert 0.5 < dur < 60` before the OK line — no hard-coded success path.

## 8. Regression tests

- CPU suite re-run by verifier: `.venv/bin/python -m pytest -m 'not gpu' -q` → **267 passed, 38 deselected, exit 0** — matches the expected 267/38 and the Task 0 collection baseline (305 total). No regressions.

## 9. Claims audit (implementer claims vs verifier-verified facts)

| Claim | Verifier finding |
|---|---|
| FA2 hard-coded at `sft_12hz.py:51` | Confirmed (grep; ground-truth §8; worktree file) |
| Wrapper forwards kwargs (`qwen3_tts_model.py:83-86/:112`) | Confirmed (`**kwargs` → `AutoModel.from_pretrained(..., **kwargs)`) |
| transformers chain `modeling_utils.py:2076→:2686→:2714→:2422` | Confirmed — all four lines contain the named symbols |
| Generic dispatch `modeling_qwen3_tts.py:788-789/:941-942`; kwargs.pop fallback `:1872-1874/:1888` | Confirmed; `sdpa` is in `ALL_ATTENTION_FUNCTIONS` |
| `demo.py` `--flash-attn/--no-flash-attn` (`:104-108`) | Confirmed; mapping `attn_impl = "flash_attention_2" if args.flash_attn else None` at line **606** (evidence cites 611-612, where `attn_implementation=attn_impl` is passed — see §10) |
| `loader.py:81 _HIP_DEFAULT_ATTN="sdpa"`, `resolve_attn :201-224`, find_spec probe `:180-184` | Confirmed |
| No attn flag in `sft_12hz.py` argparse (`:34-42`) | Confirmed — 7 args, none attention-related |
| `grep -ri flash` over `finetuning/` → only `sft_12hz.py:51` + `README.md:78`; `dataset.py`/`prepare_data.py` clean | Confirmed (grep: exactly those 2 hits; prepare_data 0; dataset.py imports as cited `:16-23`) |
| whisper_encoder try/except graceful fallback `:28-37` | Confirmed (try at 30, banner print at ~36) |
| tokenizer_v2 `_supports_flash_attn` at `:153,:917` | Confirmed (both lines are exactly that flag) |
| README example lines `161,214,254,302,319` + `finetuning/README.md:78` | Confirmed (all five + one, exact) |
| iso-driver.log sha256/line count | Confirmed (`05be3ad7…`, 311) |
| Smoke cross-refs (blob pair `c1f3f46..b5eae64` line 1556; 18.024 GiB line 1597; smoke losses 13.6470/15.0211; RELOAD-AND-SYNTHESIS-OK) | All confirmed present in `finetune-smoke-2026-09-20.txt` |
| Loss/step math (12 optimizer steps) | Consistent: 12 samples / batch 2 = 6 microbatch steps × 2 epochs; `optimizer.step()` (source lines 120-121) not gated on `sync_gradients`; script prints only at `step % 10 == 0`, hence one loss line per epoch — matches transcript |
| `.upstream/` gitignored (`.gitignore` line 27) | Confirmed (line 27 = `.upstream/`) |
| Root-cause scoping | Properly bounded (criterion 8, above); uncertainty section explicitly refuses the "flash would work if installed" claim |

## 10. Problems found

None blocking. Four minor, non-load-bearing observations:

1. **"all four `examples/*.py`" (Q2) is a slight overcount** — only 3 of the 4 example files print `flash_attention_2` (`examples/test_tokenizer_12hz.py` does not). The enumerated citations that ARE given (README lines, finetuning/README:78) all verify; the substantive claim ("only `sft_12hz.py:51` hard-codes it in executable code") is unaffected and independently confirmed.
2. **`demo.py:611-612` citation off by a few lines for the quoted text** — the mapping expression `attn_impl = "flash_attention_2" if args.flash_attn else None` sits at line 606; 611-612 is where `attn_implementation=attn_impl` is consumed by `from_pretrained`. Claim true, citation range imprecise.
3. **The iso driver shell script itself was not preserved** (only its tee'd log; the Task 1 repro drivers were kept as `repro-r1-driver.sh`/`repro-r2-driver.sh`). Mitigated: the raw per-process streams and on-disk artifacts independently corroborate every load-bearing line of the log, and the disclosed errata already cover the one missing side-log.
4. **Experiment-window upper bound "13:46" vs log mtime 13:44** — trivial imprecision in the fact table; the transcript self-bounds the run (13:39:49 start, last artifacts 13:44).

The two disclosed errata (phase-A exit label; missing phase-A raw side-log) are honestly recorded and do NOT undermine the proof: phase-A success rests on disk artifacts and downstream consumption, which the verifier confirmed, and phases B/C/C2/D all captured true process exits.

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence the verifier reproduced or executed personally, not by the implementer's assertions: (1) the one-token isolation is proven cryptographically — the pinned-commit blob and a verifier-reconstructed one-token edit reproduce the exact blob pair recorded in the pre-run diff, so the experiment's worktree demonstrably carried nothing else; (2) the full chain's outputs exist on disk with claim-matching sizes and the synthesized wav re-parses to the exact claimed statistics (24 kHz, 4.0000 s, finite, 88k/96k nonzero samples, peak 0.2275, rms 0.03978); (3) the reload script is audited hard-code-free, so the GPU-residency and `_attn_implementation='sdpa'` prints are live observations; (4) the control legs' ImportError is verbatim in the Task 1 transcripts with exit 1, closing the causal pair; (5) the worktree is pristine right now under the verifier's own commands, at the pinned SHA on the named branch; (6) the CPU suite is green at exactly 267/38; (7) all 8 questions carry file:line citations that the verifier re-opened in the actual sources, and the root-cause statement stays inside what the experiment demonstrates. The only defects found are two citation-precision nits, an unpreserved driver script, and a timestamp rounding — none of which falsifies any criterion.
