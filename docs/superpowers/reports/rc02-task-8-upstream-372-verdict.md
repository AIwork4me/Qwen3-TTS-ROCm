# RC v0.2 Task 8 — Fresh Independent Verifier Final Gate (upstream #372 PR readiness)

Program: Radeon Reference Closure v0.2. Verifier role: fresh, independent, adversarial —
no participation in any implementation; every Task 1-7 PASS treated as a claim to
falsify; nothing modified except this report file; no subagents dispatched.
Date of verification: 2026-09-21.

## 1. Verdict

**PASS** — the #372 evidence chain is coherent, internally consistent, and every
acceptance criterion I could attack is backed by executable or runtime evidence I
re-derived or re-executed myself. Three minor, non-blocking observations (Section 10);
none affects any criterion.

## 2. Exact commit / working-tree SHA reviewed

| Tree | Branch | HEAD (full SHA verified by me) | `status --porcelain` |
|---|---|---|---|
| Downstream `/home/amd/Desktop/Qwen3-TTS-ROCm` | `main` | `b64ade65d4f549acef628d6d57c8e64cd5560bbb` | empty (before this report file) |
| FIX worktree `.upstream/Qwen3-TTS-fix` | `fix/finetuning-attn-implementation` | `48b8644aac8512e6fdc0f4442788bf399c425fdb` | empty |
| Pristine clone `.upstream/Qwen3-TTS` | `main` | `022e286b98fbec7e1e916cb940cdf532cd9f488e` | empty (0 lines) |

- `merge-base --is-ancestor 022e286 HEAD` in the FIX worktree → true; history is
  exactly `022e286` → `0be0026` (`fix(finetuning): make attention implementation
  configurable`) → `48b8644` (`test(finetuning): cover default and explicit attention
  implementation`) — 2 commits, linear, as claimed.
- Downstream `main` history contains all SHAs cited by the seven verifier reports, in
  the correct order: `eaedaae`→`5a32ac8`→`12dfe67`→`faf99b8`→`844b601`→`5bfd448`→
  `47d83fe`→`69a9b4c`→`16c92cc`→`1925540`→`bee03ab`→`b64ade6`. Task 5's superseded
  pre-amend commit `3cb1aa18…` exists as an unreachable commit object
  (`git cat-file -t` → `commit`) — the recorded `--amend` is real, not a story.

## 3. Acceptance criteria checklist

### Challenges A-H (per the Task 8 mandate)

| # | Challenge | Verdict | Independent evidence (mine, this session) |
|---|---|---|---|
| A | Original problem really reproduced from pristine upstream, twice | **PASS** | Both transcripts carry, at identical line positions (386/387), the verbatim `ImportError: FlashAttention2 … flash_attn seems to be not installed` and `sft-as-is-exit=1` (PIPESTATUS[0]), with `generator-exit=0`/`prepare-exit=0` at 202/284, in-transcript pristine checks (SHA + 0-line porcelain) before AND after each run, and `CLASS-MATCH-VERBATIM` against the 2026-09-20 capture. I re-verified the pristine clone is clean at `022e286b…` right now and re-executed the grep proving the hard-code at `022e286:finetuning/sft_12hz.py` line 51 (`attn_implementation="flash_attention_2",`, exit 0). Freshness artifacts on disk: both driver scripts exist; r1/r2 `train_raw.jsonl` have distinct sha256 (`631adf0d…` vs `13b86cbc…`), both 12 lines; r1/r2 sft stdout `cmp` byte-identical (exit 0) — deterministic construction-time failure, as expected. Archived files hash to the index-claimed `0504afa7…`/`6db78216…`. Linchpin re-proven: `flash_attn` `find_spec → None` in the project venv (torch `2.12.0+rocm7.14.0`, transformers `4.57.3` — matching every transcript header). Issue #372 itself re-fetched via `gh` (OPEN, content matches the evidence chain's characterization). |
| B | Hard-coded attention backend really the minimal root cause | **PASS** | (i) Mechanism re-verified in the installed transformers 4.57.3 source: `_flash_attn_2_can_dispatch` (def at modeling_utils.py:2391, called at :2679 and :2714) raises `ImportError … flash_attn seems to be not installed` when `importlib.util.find_spec("flash_attn") is None` — the cited `:2422` raise site — an init-time dispatch check before any kernel. (ii) Causal pair: control legs (A) fail at exactly that site with exit 1; treatment leg (Task 2 iso experiment, transcript on disk at `.work-finetune/iso-driver.log` + raw streams) flipped ONLY the one token (blob pair `c1f3f46..b5eae64`, cryptographically reconstructed by the Task 2 verifier) and the full chain passed with exit 0 at every stage — no second blocker. (iii) Scope exclusions hold: my grep of the whole FIX diff shows the only attention references in `sft_12hz.py` are the new arg (line 40) and the call-site kwarg (line 56); the failure occurs inside `from_pretrained` before `open(args.train_jsonl)` (line 60) is reached — excluding data/schema causes; the alternative-hypotheses section of `upstream-372-root-cause.md` rejects architecture-coupling, hidden-dependency, and version-skew explanations with citations I spot-checked. |
| C | Fix preserves upstream default behavior | **PASS** | Three independent levels: (1) source — patched line 40 is `default="flash_attention_2"` (read by me in the worktree file); (2) unit — I re-ran the FIX-branch suite: 5/5 `OK`, exit 0, including `test_default_is_flash_attention_2` and `test_default_value_reaches_model_loader` (which drives the full real `train()` and captures the kwarg reaching `from_pretrained`); (3) runtime — Round D transcript shows a real default-flag load (no `--attn_implementation`, parser prints `(parser DEFAULT -- no flag given)`) failing with `python-exit=1` and an ImportError line byte-identical to the pristine capture — I re-derived that `diff` myself (`CLASS-MATCH-RE-DERIVED`, exit 0). Default-path behavior on a flash-attn-less host is unchanged by the patch. |
| D | ROCm fine-tuning executes without editing upstream source at runtime | **PASS** | Both E2E transcripts (line 362 each) record the verbatim DIRECT invocation `(cd .upstream/Qwen3-TTS-fix && .venv/bin/python finetuning/sft_12hz.py --attn_implementation sdpa … )` with `sft-direct-exit=0` explicitly labeled `PIPESTATUS[0], NOT of the tee wrapper` (line 386 each). The tee logs still on disk (`.work-finetune/e2e-r{1,2}.log`) contain the script's own loss lines and zero `ImportError` occurrences (grep exit 1). The patch exists only as two commits on the FIX branch — the worktree is porcelain-clean NOW, i.e. what ran is the committed state, not a working-tree edit. No runtime-patch mechanism present: `find` over the worktree for `*.pth`, `sitecustomize.py`, `usercustomize.py`, `conftest.py` → nothing. `sft_12hz.py` ends with `if __name__ == "__main__": train()` — the script runs as a script. |
| E | Two independent E2E training runs passed | **PASS** | Transcripts byte-identical to their tee captures (my `cmp`, both exit 0; sha256 `68156fbb…`/`6877ad19…` match index). Independence re-proven by me on disk NOW: disjoint trees; per-run `train_raw.jsonl` hashes `e8adc00c…` (r1) vs `9d4a6c1d…` (r2) — distinct, matching the JSONs digit-for-digit; zero cross-tree references in either direction (grep exit 1 both ways); distinct verbatim losses (r1 14.6621→10.0124; r2 14.1067→11.9973 — different shuffle orders, genuinely separate trainings); four epoch-1 `model.safetensors` sha256 all distinct (my hashes: `bf87ab9d…`, `464fd883…`, `4bd5f692…`, `308d5c60…` — each matches the transcript/JSON claim); per-run speaker ids in checkpoint configs (`e2e_r1_speaker`/`e2e_r2_speaker`: 3000, parsed by me). Every `model.safetensors` is exactly 3,833,402,520 B. |
| F | Patch generic rather than AMD-specific | **PASS** | My grep over the full `022e286..HEAD` diff for `amd|hip|rocm|radeon|gfx|cuda|platform|version|environ|getenv|device_map` (case-insensitive) → 0 hits. The change adds a plain CLI flag with the upstream default preserved; nothing conditions on vendor, device, or environment. It implements option 1 of the issue's own "Suggested fix" (`--attn_implementation`, default `flash_attention_2`). |
| G | PR diff minimal | **PASS** | My own `git diff 022e286..HEAD --stat`: exactly 2 files, `finetuning/sft_12hz.py` (+10/−5 across 3 hunks) and new `tests/test_sft_attn_implementation.py` (105 lines, stdlib-only imports: `sys`, `unittest`, `pathlib`, `unittest.mock`). I read the entire 151-line diff: the sft_12hz.py delta is precisely (a) factoring the unchanged 7-argument parser into `build_arg_parser()` (moved lines byte-identical — Task 3 proved this by programmatic reconstruction from the sha256-matched Task 0 capture), (b) one new `add_argument("--attn_implementation", …, default="flash_attention_2")`, (c) the one call-site kwarg `attn_implementation=args.attn_implementation`. Nothing else; no whitespace churn (`diff -w` stat identical per Round E, consistent with my read); no dependency/vendor/architecture changes. |
| H | Documentation claims no stronger than evidence | **PASS** | (1) `evidence/README.md` 372-chain rows: every hash I re-derived matches the row (0504afa7/6db78216 pristine runs; 5ab46949/7d1f0050/5e3e9cfc loads; 68156fbb/6877ad19 e2e txts; 0e6c1723 default-semantics; bf87ab9d/464fd883/e8adc00c/9d4a6c1d artifact hashes); rows carry scope guards ("execution validation only, no quality claims", "CUDA runtime explicitly NOT claimed") that the transcripts honor. (2) `docs/finetuning-rocm.md`: PROVEN items are each backed by transcript line ranges I spot-checked; the NOT PROVEN list explicitly disclaims quality/convergence; the workaround description (one-line sdpa override in a gitignored clone, restored pristine) matches the smoke evidence. (3) `README.md` fine-tuning row is scoped to the smoke ("execution-only smoke … no quality/convergence claims") — it UNDER-claims relative to the newer FIX-branch E2E proof, which is safe. Two prose nits found (Section 10), neither a false technical claim. |

### The 12 PR preconditions (Definition of Done)

| # | Precondition | Verdict | Backing |
|---|---|---|---|
| 1 | Double pristine reproduction | **MET** | Challenge A |
| 2 | Causal chain documented | **MET** | `evidence/upstream-372-root-cause.md` §1; mechanism re-verified in installed transformers source (Challenge B) |
| 3 | Minimal fix isolated | **MET** | Task 2 one-token iso (blob-pair proof) + Task 3 reconstruction + my diff read (Challenges B, G) |
| 4 | Targeted SDPA loads ×3 | **MET** | Three loads transcripts; my grep confirms per-file markers: `parsed attn_implementation=sdpa`, `config._attn_implementation=sdpa`, `cuda_available=True`, `ROUND_A_LOAD_OK`, `python-exit=0`, and (supplement leg) `placement_method=device_map`, `param_device=cuda:0`, `mem_alloc_GiB=3.91` in all three; file hashes match the index |
| 5 | E2E run #1 passes | **MET** | Challenge E (r1: all phase exits 0, loss lines, checkpoints on disk) |
| 6 | Independent E2E run #2 passes | **MET** | Challenge E (r2: fully separate tree/dataset/process/checkpoints) |
| 7 | Checkpoint save succeeds | **MET** | 6 checkpoint dirs on disk NOW; every `model.safetensors` 3,833,402,520 B; `config.json` converted to `custom_voice` with per-run `spk_id` (parsed by me) |
| 8 | Checkpoint reload succeeds | **MET** | `reload-exit=0` + `RELOAD-AND-SYNTHESIS-OK` in both transcripts (and the Task 2 iso leg); reload via official `Qwen3TTSModel.from_pretrained` |
| 9 | Post-finetune synthesis succeeds | **MET** | Both wavs re-parsed by me from disk: PCM mono 24,000 Hz 16-bit; r1 3.2000 s, peak 0.3320, rms 0.05800, nonzero 0.9894; r2 3.4400 s, peak 0.1914, rms 0.04274, nonzero 0.9264 — finite, non-silent, matching transcript claims to the last digit |
| 10 | Default FA2 semantics unchanged | **MET** | Challenge C (source default, unit tests re-run, runtime default-flag failure byte-identical to pristine) |
| 11 | Diff contains no unrelated changes | **MET** | Challenge G (2 files; itemized 115+/5−; zero prohibited patterns; stdlib-only additions) |
| 12 | Fresh verifier PASS | **THIS REPORT** | Verdict PASS |

## 4. Files reviewed

- `gh issue view 372 --repo QwenLM/Qwen3-TTS` (live fetch, JSON form — issue OPEN, content as characterized)
- All seven verifier reports: `docs/superpowers/reports/rc02-task-{1..7}-verification.md` (read in full, critically)
- Evidence chain (grep-verified + spot-read): `evidence/upstream-372-pristine-failure-run{1,2}.txt`, `evidence/upstream-372-root-cause.md` (full read), `evidence/upstream-372-round-a-loads{1,2,3}.txt`, `evidence/upstream-372-e2e-run{1,2}.{txt,json}`, `evidence/upstream-372-default-semantics.txt`, `evidence/upstream-372-diff-audit.txt`, `evidence/README.md` (372 rows)
- Docs: `README.md` (fine-tuning row, line 324), `docs/finetuning-rocm.md` (full read)
- Source: `.upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py` (full, 166 lines), `.upstream/Qwen3-TTS-fix/tests/test_sft_attn_implementation.py` (via diff), `git show 022e286:finetuning/sft_12hz.py`, `.venv/.../transformers/modeling_utils.py` (dispatch-check region)
- On-disk artifacts: `.work-finetune/e2e-r{1,2}/` checkpoints + configs + wavs, `.work-finetune/e2e-r{1,2}.log`, `.work-finetune/e2e-r{1,2}-driver.log`, `.work-finetune/repro-r{1,2}/` jsonl + sft stdout + driver scripts, `.work-finetune/iso/`, `.work-finetune/round_d_default_load_stdout.txt`

## 5. Exact commands executed (by this verifier)

```
git status --porcelain; git log --oneline -3; git rev-parse HEAD                        # downstream
git -C .upstream/Qwen3-TTS-fix status --porcelain; log --oneline -5; rev-parse HEAD; branch --show-current
git -C .upstream/Qwen3-TTS-fix rev-parse 022e286; merge-base --is-ancestor 022e286 HEAD; log --format='%H %s' 022e286..HEAD
git -C .upstream/Qwen3-TTS status --porcelain | wc -l; rev-parse HEAD                   # pristine clone
gh issue view 372 --repo QwenLM/Qwen3-TTS --json number,title,state,author,createdAt,body,url
git -C .upstream/Qwen3-TTS-fix diff 022e286..HEAD; diff --stat 022e286..HEAD
git -C .upstream/Qwen3-TTS show 022e286:finetuning/sft_12hz.py | grep -n flash_attention_2; sed -n '48,53p'
cd .upstream/Qwen3-TTS-fix && .venv/bin/python -m unittest tests.test_sft_attn_implementation -v
.venv/bin/python -m pytest -m 'not gpu' -q                                              # repo root
grep <markers> on all 9 upstream-372 evidence files (exits, ImportError, sdpa, OK lines, H1-H4, ROUND_D_RESULT)
sha256sum <6 evidence files> ; cmp evidence/e2e-run{1,2}.txt .work-finetune/e2e-r{1,2}-driver.log
sha256sum <4 epoch-1 model.safetensors> ; stat -c '%s' <all 4 + epoch-0>
.venv/bin/python <wave/struct parse of both reload wavs> ; python3 <json parse of 2 checkpoint configs>
python3 <walk of both e2e JSONs> ; sha256sum repro-r{1,2}/train_raw.jsonl e2e-r{1,2}/data/train_raw.jsonl
cmp repro-r1/sft-run1-stdout.txt repro-r2/sft-run2-stdout.txt
grep -c cross-tree references (e2e-r2 in r1 files; e2e-r1 in r2 files)
git diff 022e286..HEAD | grep -icE 'amd|hip|rocm|radeon|gfx|cuda|platform|version|environ|getenv|device_map'
find .upstream/Qwen3-TTS-fix -name '*.pth' -o -name sitecustomize.py -o -name usercustomize.py -o -name conftest.py
sed -n '2415,2430p' .venv/.../transformers/modeling_utils.py ; grep -n _flash_attn_2_can_dispatch …
diff <(grep -m1 '^ImportError: FlashAttention2' pristine-run1) <(grep -m1 … round_d_default_load_stdout.txt)
grep -m1 'ImportError: FlashAttention2' .work-finetune/e2e-r{1,2}.log ; grep -c 'Epoch 0 | Step 0' …
git log --oneline -14 ; git cat-file -t 3cb1aa18c19e343d647c87fa72e2b5ab98652e7c
.venv/bin/python -c 'importlib.util.find_spec("flash_attn"); torch.__version__; transformers.__version__'
```

## 6. Exit codes (verifier-observed)

- Downstream/FIX/pristine `status --porcelain`: exit 0, empty output, all three trees.
- `merge-base --is-ancestor 022e286 HEAD` (FIX): exit 0.
- `grep flash_attention_2` on `022e286:finetuning/sft_12hz.py`: exit 0 → line 51.
- FIX-branch unittest (my run): exit 0 — `Ran 5 tests … OK` (plus benign SoX/flash-attn import warnings, pre-existing).
- Downstream `pytest -m 'not gpu' -q` (my run): exit 0 — **267 passed, 38 deselected in 15.96s** (exact expected counts).
- `cmp` evidence-vs-tee (both E2E runs) and r1-vs-r2 sft stdout: exit 0 (identical).
- Class-match `diff` (pristine run1 vs Round D default-load stdout): exit 0.
- Prohibited-pattern grep over the FIX diff: exit 1 (zero matches — desired).
- Runtime-patch `find` in FIX worktree: exit 1 (nothing found — desired).
- Cross-tree reference greps (both directions): exit 1 (zero matches — desired).
- `grep ImportError` in E2E tee logs: exit 1 (zero matches — desired).
- `gh issue view 372 --json …`: exit 0 (issue OPEN).
- `flash_attn` probe: `find_spec → None` (absent — the failure-class precondition holds).

## 7. Runtime evidence inspected

- **Pristine failures (×2):** ImportError + `sft-as-is-exit=1` at lines 386/387 of BOTH transcripts; pristine invariants recorded in-transcript before/after; per-run fresh datasets (distinct jsonl hashes, 12 lines each); byte-identical sft stdout across runs; `CLASS-MATCH-VERBATIM` vs the 2026-09-20 capture.
- **Round A loads (×3):** all success markers present with expected counts in all three files, both verbatim (CPU-resident, `mem_alloc_GiB=0.00` by construction) and supplement (GPU-resident: `param_device=cuda:0`, `mem_alloc_GiB=3.91`) legs; no genuine ImportError/traceback anywhere.
- **E2E runs (×2):** direct patched-script invocation exit 0 in both; per-phase exits 0; loss lines verbatim (differing across runs — real independent trainings); checkpoints (exact sizes, distinct hashes), reload (`RELOAD-AND-SYNTHESIS-OK`), and synthesis wavs (re-parsed by me: 24 kHz mono PCM, finite, non-silent, plausible durations, exact claimed statistics).
- **Round D:** unit leg 5/5 OK exit 0 (re-run by me); runtime default-flag leg `python-exit=1` with ImportError byte-identical to pristine (re-derived by me); `ROUND_D_RESULT=PASS`; verbatim statement recorded; no CUDA-validation claim anywhere in the 372-chain docs.
- **Round E:** per-hunk audit H1-H4 with itemized 115+/5− accounting; my re-derived diff equals the audited diff.
- **Environment:** flash-attn absent, torch 2.12.0+rocm7.14.0, transformers 4.57.3, Python 3.12.3 — all re-confirmed live in the venv; transformers dispatch-check raise site re-read in source.

## 8. Regression tests

- FIX branch: `python -m unittest tests.test_sft_attn_implementation -v` — 5/5 OK, exit 0 (re-executed by me).
- Downstream CPU suite: `pytest -m 'not gpu' -q` — 267 passed, 38 deselected, exit 0 (re-executed by me; exact expected counts; the 38 deselected are the registered `gpu`-marker suite).
- Runtime regression anchor: default-flag behavior on this flash-attn-less host is byte-identical (ImportError line) between pristine upstream and the FIX branch — the patch changes nothing on the default path.

## 9. Claims audit (chain coherence — hunting for rubber-stamps)

- Every downstream SHA cited across the seven reports exists in `main`'s history in the correct order (Section 2); Task 5's pre-amend `3cb1aa1` exists as an unreachable object of type commit — the amend narrative is git-verifiable, not asserted.
- No contradictions found between the seven verifier reports and the raw evidence: hashes, sizes, loss values, exit codes, timestamps, and artifact facts all agree wherever two sources overlap. Each report contains at least one independently strong falsification attempt (Task 2's blob-pair cryptographic reconstruction; Task 3's byte-level programmatic reconstruction; Task 4's dummy-file-absence probe; Task 5's head-111 prefix hash vs pre-amend blob; Task 6's cross-run hash distinctness; Task 7's three-way diff byte-compare) — these are not rubber-stamps.
- Implementer-side defects found by the earlier verifiers (Task 2 citation nits; Task 4 abbreviated warning transcription; Task 7 P1 timestamp mislabeling) were disclosed in those reports, are prose-level, and I confirmed none touches evidence or criteria.
- The FIX diff is exactly what Tasks 3/4/7 describe; the docs' claims are all at or below the evidence level.

## 10. Problems found

None blocking. Three minor, non-blocking observations:

1. **Stale prose in `docs/finetuning-rocm.md` Limitations:** line 240 still says "Upstream issue filing remains a human-owner action", while the callout at lines 87-94 (and the live issue, re-fetched by me) records #372 as already filed 2026-09-21. Outdated process sentence, not a false technical claim; recommend a one-line refresh when the doc is next touched.
2. **`README.md` fine-tuning row describes only the 2026-09-20 smoke-era workaround** ("applied inside a gitignored clone, restored pristine") and does not mention the stronger FIX-branch evidence (two direct patched-script E2E runs). Under-claiming — safe — but the row will want updating after the upstream PR lands.
3. **`gh issue view 372` in default (non-JSON) mode fails on this host** with a GraphQL Projects-classic deprecation error; the JSON field-select form works. A capture-environment quirk, no bearing on the chain (the archived `evidence/upstream-issue-2026-09-21.txt` and my live JSON fetch agree).

## 11. Why PASS is justified

I attacked the chain as a falsifier, not a reader: I re-executed the three mandated checks myself (line-51 grep at the pinned commit; FIX-branch unittest suite 5/5 OK; downstream CPU suite exactly 267/38, exit 0), re-derived the FIX diff from git and scanned it for vendor-specific patterns (zero hits), proved the absence of any runtime-patch mechanism in the worktree, re-hashed every load-bearing evidence file and on-disk artifact (all match claims digit-for-digit), re-parsed both synthesis wavs and both checkpoint configs independently, re-derived the default-path failure-class match byte-for-byte, re-confirmed the root-cause mechanism in the installed transformers source, re-confirmed flash-attn absence in the venv, re-fetched the live issue, and verified the full commit topology of all three trees plus every SHA cited by the seven prior reports — including the unreachable pre-amend commit that proves the Task 5 amend story. Every challenge A-H and every one of the 12 PR preconditions is backed by evidence I produced or reproduced myself; the three findings in Section 10 are prose-level and affect no criterion. The chain is ready for upstream PR preparation.
