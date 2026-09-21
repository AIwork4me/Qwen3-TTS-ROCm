# Task 4 Independent Verification Report — Radeon Reference Closure v0.2

**Verifier role:** independent release verifier (assume implementer wrong until proven right)
**Date:** 2026-09-21
**Method:** source review, git-history audit, fresh-process execution (twice), diff
cross-check, mock-surface audit, dummy-file absence probe, downstream regression suite.

---

## 1. Verdict

**PASS**

## 2. Exact commit / working-tree SHA reviewed

| Repo | Branch | HEAD SHA | Status |
|---|---|---|---|
| `.upstream/Qwen3-TTS-fix` | `fix/finetuning-attn-implementation` | `48b8644aac8512e6fdc0f4442788bf399c425fdb` (short `48b8644`) | `status --porcelain` empty (verified before, during and after my test runs; `__pycache__/` is gitignored) |
| Downstream `/home/amd/Desktop/Qwen3-TTS-ROCm` | `main` | `844b601` | `status --porcelain` empty at verification time (only this report file dirties it afterwards, which is the sanctioned exception) |

Commit graph on FIX branch (verified): `48b8644` (tests) → parent `0be0026ee99ae90173009e4a6540b336df360f22` (fix) → `022e286` (upstream base). Author of `48b8644`: `amd <amd@localhost>`, consistent with the report's disclosed one-shot identity flags. Commit stat: 1 file changed, 105 insertions — matches report.

## 3. Acceptance criteria checklist

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1a | Default remains `flash_attention_2` | `test_default_is_flash_attention_2` asserts real `build_arg_parser().parse_args().attn_implementation == "flash_attention_2"`; also `test_default_value_reaches_model_loader` proves the default propagates through real `train()`. Both ran ok in my fresh processes | MET |
| 1b | Explicit `sdpa` selectable | `test_explicit_sdpa_selectable` — real parser, real parse, ok | MET |
| 1c | Selected value reaches `Qwen3TTSModel.from_pretrained` | `test_value_reaches_model_loader` (sdpa) and `test_default_value_reaches_model_loader` (default): real `train()` executes real parse → real `Accelerator(...)` mock → intercepted `from_pretrained`, kwargs captured, `_StopAfterLoad` raised. Both ok. Two-value design defeats hardcoding: a hardcoded `"flash_attention_2"` fails the sdpa test; a hardcoded `"sdpa"` fails the default test | MET |
| 1d | Parser backward-compatible with pre-patch argument set | `test_backward_compatible_parse` — see criterion 7 below. Ok | MET |
| 2 | Mocking limited to `Accelerator` + `from_pretrained` interception | Full-source audit: only `sys.argv`, `sft_12hz.Accelerator`, `sft_12hz.Qwen3TTSModel` are patched. Parser, `build_arg_parser`, `parse_args`, `torch`, `AutoConfig` never mocked. All assertions are on real parsed argparse values (`args.*`) or on kwargs captured from the real `train()` call site (`captured["attn_implementation"]`) — not on behaviour of any mock of the thing under test | MET |
| 3 | stdlib unittest only | Import audit of the test file: `sys`, `unittest`, `pathlib.Path`, `unittest.mock`, plus the module under test `sft_12hz`. No pytest, no third-party test deps | MET |
| 4 | Dummy `--train_jsonl` adaptation does not weaken criteria | (i) Source order in `train()` at `48b8644`: `from_pretrained` (line ~55) precedes `open(args.train_jsonl)` (line ~61). (ii) Falsification probe: `/tmp/dummy.jsonl` does NOT exist on this machine; the tests pass with `assertRaises(_StopAfterLoad)`, so a `FileNotFoundError` was never raised — executable proof the dummy path is never opened. (iii) Supplying `--train_jsonl` mirrors real pre-patch usage (it was `required=True` at `022e286`), so no criterion is weakened: parser tests still exercise the real parser, loader tests still run the real `train()` | MET |
| 5 | Tests pass in a FRESH process, run twice by verifier | Executed twice myself (commands in §5): both `Ran 5 tests in 0.002s` / `OK` / exit 0; stderr byte-identical between runs | MET |
| 6 | FIX topology + downstream clean | `48b8644` → `0be0026` → `022e286` confirmed via `git log --format='%H %P'`; FIX porcelain empty; downstream `main` at `844b601` with porcelain empty | MET |
| 7 | Criterion (d) flag list matches REAL pre-patch arguments | `git show 022e286:finetuning/sft_12hz.py` — the pre-patch parser defines exactly 7 flags: `--init_model_path` (str, default `Qwen/Qwen3-TTS-12Hz-1.7B-Base`), `--output_model_path` (str, default `output`), `--train_jsonl` (str, `required=True`), `--batch_size` (int, 2), `--lr` (float, 2e-5), `--num_epochs` (int, 3), `--speaker_name` (str, `speaker_test`). The test exercises all 7 (six explicitly + `train_jsonl` via `_parse` prefix), asserts each parsed value (including type-coerced `4`, `1e-4`, `1`), and supplies no post-patch flag. Additionally `_parse([])` in test 1a proves no NEW required argument was introduced | MET |

## 4. Files reviewed

- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-4-brief.md`
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-4-report.md`
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/review-task4-fix-branch.diff`
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix/tests/test_sft_attn_implementation.py` (on disk, full read)
- `finetuning/sft_12hz.py` at `48b8644` and at pre-patch base `022e286` (via `git show`)
- `.upstream/Qwen3-TTS-fix/.gitignore` (confirms `__pycache__/` ignored → clean-porcelain claim legitimate)

## 5. Exact commands executed (by verifier)

```
git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix log --oneline -5
git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix status --porcelain
git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix branch --show-current
git -C /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix rev-parse HEAD
git -C /home/amd/Desktop/Qwen3-TTS-ROCm status --porcelain
git -C /home/amd/Desktop/Qwen3-TTS-ROCm rev-parse --short HEAD
git -C /home/amd/Desktop/Qwen3-TTS-ROCm log --oneline -5
git -C .upstream/Qwen3-TTS-fix show 48b8644:finetuning/sft_12hz.py
git -C .upstream/Qwen3-TTS-fix show 022e286:finetuning/sft_12hz.py
git -C .upstream/Qwen3-TTS-fix diff 022e286..48b8644   # compared to review-task4-fix-branch.diff
git -C .upstream/Qwen3-TTS-fix show --stat --oneline 48b8644
git -C .upstream/Qwen3-TTS-fix log --format='%H %P' -1 48b8644
git -C .upstream/Qwen3-TTS-fix log -1 --format='%an <%ae> | %ad' 48b8644
ls -la /tmp/dummy.jsonl                                # absence probe
cd /home/amd/Desktop/Qwen3-TTS-ROCm/.upstream/Qwen3-TTS-fix && \
  /home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python -m unittest tests.test_sft_attn_implementation -v   # run 1
  (same command again)                                                                                 # run 2
cd /home/amd/Desktop/Qwen3-TTS-ROCm && .venv/bin/python -m pytest -m 'not gpu' -q
/home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python --version   # Python 3.12.3
```

## 6. Exit codes

| Command | Exit code |
|---|---|
| Upstream unittest, run 1 (fresh process) | 0 |
| Upstream unittest, run 2 (fresh process) | 0 |
| Downstream `pytest -m 'not gpu' -q` | 0 |
| All git commands above | 0 |
| `diff` of cumulative diff file vs real `git diff 022e286..48b8644` | 0 (byte-identical) |

## 7. Runtime evidence inspected

**Upstream suite, run 1 and run 2 (byte-identical stderr between runs):**

```
test_backward_compatible_parse (...) ... ok
test_default_is_flash_attention_2 (...) ... ok
test_default_value_reaches_model_loader (...) ... ok
test_explicit_sdpa_selectable (...) ... ok
test_value_reaches_model_loader (...) ... ok
----------------------------------------------------------------------
Ran 5 tests in 0.002s
OK
```

(Import-time SoX / flash-attn stderr noise reproduces in my runs too — pre-existing, cosmetic, disclosed by the implementer.)

**Downstream CPU suite:**

```
267 passed, 38 deselected in 15.91s
```

Exactly the expected 267 passed / 38 deselected, exit 0. (Disambiguation: no `.venv` exists inside the upstream worktree, so the `.venv/bin/python` used is necessarily the downstream one and the suite necessarily the downstream suite.)

**Dummy-file probe:** `ls: cannot access '/tmp/dummy.jsonl': No such file or directory` — combined with passing `assertRaises(_StopAfterLoad)`, this is executable proof that `open(args.train_jsonl)` is never reached.

## 8. Regression tests

- Downstream CPU suite re-run by verifier: 267 passed, 38 deselected, exit 0 — no downstream regression.
- Upstream FIX worktree: no pre-existing test infra existed (the `tests/` dir is new); the new suite is itself the regression guard and passes deterministically in two fresh processes.

## 9. Claims audit (implementer report vs observed reality)

| Claim in task-4-report.md | Verifier observation | Agree? |
|---|---|---|
| 5/5 tests pass in two fresh processes | Reproduced exactly, twice, exit 0, identical output | Yes |
| Test file is 105 lines, tests/ dir newly created, stdlib-only | Confirmed on disk, in commit stat (105 insertions), and via import audit | Yes |
| Actual parser args table (7 flags incl. `--train_jsonl` required) | Matches `git show 022e286:finetuning/sft_12hz.py` and patched `48b8644` parser verbatim | Yes |
| Dummy path never opened; `_StopAfterLoad` fires first | Source order + dummy-file-absence probe | Yes |
| FIX commit `48b8644`, parent `0be0026`, clean tree, not pushed | Verified (no remote push performed between; `git log` parent confirms) | Yes |
| SoX warning block "abbreviated in wording only" | Disclosed honestly; cosmetic, does not affect results | Yes |
| Downstream docs commit intentionally skipped (verifier report must exist first) | Consistent with downstream main clean at `844b601` containing only Task ≤3 report commits; orchestrator folds it in post-gate | Yes |
| `torch_dtype=torch.bfloat16` evaluates against real torch | True — `torch` is not mocked anywhere in the test file | Yes |

## 10. Problems found

None blocking. Minor observations, none affecting any acceptance criterion:

1. The implementer's transcribed SoX warning text in report §5 is abbreviated (self-disclosed); my own runs reproduce the real (wordier) SoX block plus the flash-attn warning — cosmetic transcription, not an evidence problem since I re-ran everything myself.
2. The report's "Run 2 … identical" claim lacked pasted output; I re-ran it myself and confirmed byte-identity, so the gap is closed by verifier evidence rather than implementer evidence.
3. `test_backward_compatible_parse` supplies no post-patch flag but also cannot prove the *absence* of new optional flags — however backward compatibility (old invocations still parse) is what the criterion asks for, and `_parse([])` additionally proves no new required flags. No gap against the criterion as written.

## 11. Why PASS is justified

Every acceptance criterion is backed by evidence I produced or independently reproduced, not by trusting the implementer: the four brief criteria map one-to-one (plus a redundant second loader test) onto passing tests executed by me in two fresh processes with exit 0; the mock surface is audited at source level and limited exactly to `Acceler` + `from_pretrained` interception (`sys.argv` patching is the standard parse harness), with all assertions on real parsed values / real captured kwargs; imports are stdlib-only; the dummy-`train_jsonl` adaptation is proven harmless by the strongest possible probe — the file does not exist and the suite still passes, so it is provably never opened; git topology, parentage, cleanliness, and the byte-level identity of the cumulative diff with `git diff 022e286..48b8644` were all verified directly; criterion (d)'s flag list was checked against the actual pre-patch parser at the base commit, not the report's table; and the downstream CPU suite passes at exactly the expected 267/38/exit-0. Attempted falsifications (hardcoded-attn-value bypass, dummy-file open, diff tampering, hidden pytest dependency, commit-topology mismatch, new-required-flag introduction) all fail to materialize. No criterion rests on implementer-supplied output alone.
