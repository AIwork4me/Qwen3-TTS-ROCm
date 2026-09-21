# RC02 Task 15 (upstream drift watcher) — independent verification

Verifier role: adversarial; assumed the implementer report wrong until evidence
showed otherwise. Nothing was modified except this report file (tampered-baseline
experiments used /tmp copies and were deleted; the committed baseline was never
rewritten — end-state `git status --porcelain` is empty).

## 1. Verdict

**PASS** — all 8 acceptance criteria backed by executable/runtime evidence
obtained in this session; zero falsifications succeeded.

## 2. Exact commit / working-tree SHA reviewed

- Commit: `bbc25dbdd447b0026435fe2efaaf5e0b944034e1` ("chore(upstream): add
  upstream drift detection (SHA/PyPI/tree/API surface)"), sole commit in
  `34a4f7e..bbc25db`, 4 files / 486 insertions.
- Working tree: HEAD == `bbc25dbdd447b0026435fe2efaaf5e0b944034e1`,
  `git status --porcelain` empty (clean).
- Remote: `git ls-remote origin main` → `bbc25dbdd447b0026435fe2efaaf5e0b944034e1`
  (and local `origin/main` ref agrees) — **pushed**, contrary to the implementer
  report's stale "push FAILED" claim (see §9).

## 3. Acceptance criteria checklist

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Script stdlib-only; no requests/third-party; no subprocess/gh fallback | PASS | AST walk of the delivered file: module + lazy imports are `__future__, argparse, inspect, json, pathlib, sys, time, urllib.request` (all in `sys.stdlib_module_names`) plus the brief's own lazy `from qwen_tts import Qwen3TTSModel` inside `generate_api_surface()` — the project's pinned dependency (`qwen-tts==0.1.1`), not a new one. Source contains no `subprocess`/`requests`/`socket`/`http.client`/`os.system`/`popen` text; no gh fallback exists (implementer claim confirmed). |
| 2 | Output contract: UNCHANGED→0; DRIFT+diffs→1; network/collection error→graceful+2; `--update-baseline` writes baseline | PASS (all four legs RUN live) | (a) `.venv/bin/python scripts/check_upstream_drift.py` → stdout `UNCHANGED`, exit 0, wall 2m06s. (b) Tampered /tmp copy (`pypi_version` → `9.9.9`, exactly one field), `--baseline /tmp/rc02-tampered-baseline.json` → `UPSTREAM DRIFT DETECTED — …` + JSON body naming only `pypi_version` with `"baseline": "9.9.9", "current": "0.1.1"`, exit 1. (c) All proxy env vars pointed at dead port `127.0.0.1:9` → stderr `NETWORK ERROR (exit 2 -- drift NOT determined): GET https://api.github.com/… URLError … Connection refused`, exit 2 (graceful, no traceback). (d) `--update-baseline --baseline /tmp/rc02-fresh-baseline.json` → `BASELINE UPDATED`, exit 0; written file **byte-identical** (`diff` clean) to committed `docs/upstream-baseline.json` — the committed baseline is exactly reproducible. |
| 3 | Drift output never auto-marks Radeon-compatible; disclaimer present | PASS | Live exit-1 banner reads verbatim: `UPSTREAM DRIFT DETECTED — revalidation required (new upstream capability is NOT auto-marked Radeon-compatible):`; also pinned by `test_cli_drift_exit_1_and_never_auto_green`. Nothing in script output asserts compatibility. |
| 4 | Baseline sane: sha 022e286b…, pypi 0.1.1, non-empty tree hashes, api_surface coverage | PASS | JSON load: `upstream_sha` `022e286b98fbec7e1e916cb940cdf532cd9f488e`; `pypi_version` `0.1.1`; 37 `file_hashes` entries, all non-empty (incl. `finetuning/sft_12hz.py`, `README.md`); `api_surface` keys = `create_voice_clone_prompt, generate_custom_voice, generate_voice_clone, generate_voice_design`. Independent cross-check with the verifier-supplied `inspect.signature` command against the installed package: `live == baseline api_surface` is **True** (exact string equality for all 4 signatures, none skipped). |
| 5 | Tests: network-free diff_state coverage per brief; both suites green | PASS | `pytest tests/test_check_upstream_drift.py -v` → **18 passed** in 0.03s, covering every brief-required case (identical→{} incl. empty sections; sha change; version change; both together; file add/remove/change and all-three-simultaneous with exact sets; api signature change; plus new-method/forward-compat extras). Full CPU suite `pytest -m 'not gpu' -q` → **313 passed, 38 deselected** in 17.5s — exactly the expected counts (marker census: 351 collected; every `requires_download` test is also `gpu`-marked, so `-m 'not gpu'` == CI's `-m "not gpu and not requires_download"` here). |
| 6 | New weekly Action: CPU-only, mirrors ci.yml, doesn't touch ci.yml, YAML parses | PASS | `yaml.safe_load` parses `.github/workflows/upstream-drift.yml`: schedule cron `17 3 * * 1` + `workflow_dispatch`, `permissions: contents: read`, `runs-on: ubuntu-latest` (CPU), `timeout-minutes: 15`; drift step runs `python scripts/check_upstream_drift.py` (nonzero exit fails the job). Action pins `{checkout@3d3c42e5…, setup-python@5fda3b95…}` verified programmatically to be a **subset of ci.yml's pins**; CPU-torch-first + pinned `qwen-tts==0.1.1` mirror ci.yml. `git diff-tree -r bbc25db` file list contains neither `ci.yml` nor `gpu-nightly.yml`. `ruff check .` → "All checks passed!" (ci.yml's lint step would not fail). |
| 7 | No network calls in unit tests | PASS | Grep audit: the only monkeypatching is `drift.collect` replacement; no `urlopen`/`urllib`/`socket`/`requests` usage; `qwen_tts` import in the script is lazy and never triggered by the tests. Falsification attempt: re-ran the 18 tests with `http(s)_proxy=http://127.0.0.1:9` → **18 passed** — no test can reach the network. |
| 8 | Commit bbc25db pushed; tree clean | PASS | `git ls-remote origin main` → `bbc25dbdd…` == HEAD; local `origin/main` ref identical; working tree clean at start and end of verification. |

Sanctioned deltas confirmed as described and bounded: (i) exit-2 convention
(docstring + `--help` epilog + 4 pinning tests; also covers the one-retry loop,
`NetworkError`, BASELINE ERROR exit-2, and the `main(argv=None)` parameter that
makes the CLI testable in-process); (ii) split one-per-line imports (+`time`
for the retry sleep). No other behavioral deviations from the brief's code.

## 4. Files reviewed

- `/home/amd/Desktop/Qwen3-TTS-ROCm/scripts/check_upstream_drift.py` (161 lines, mode 100755)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/docs/upstream-baseline.json` (37 tree hashes + 4 signatures)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/tests/test_check_upstream_drift.py` (226 lines, 18 tests)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.github/workflows/upstream-drift.yml` (new, 50 lines)
- `/home/amd/Desktop/Qwen3-TTS-ROCm/.github/workflows/ci.yml` (read for convention/pin comparison — untouched by bbc25db)
- `.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-15-brief.md`, `task-15-report.md`, `review-34a4f7e..bbc25db.diff`

## 5. Exact commands executed

1. `.venv/bin/python` heredoc: AST import audit + baseline JSON assertions + `yaml.safe_load` on the new workflow
2. `.venv/bin/python -c "import inspect; from qwen_tts import Qwen3TTSModel; [print(n, inspect.signature(...)) ...]"` (verifier-supplied cross-check)
3. Second heredoc: exact `live == baseline api_surface` string comparison
4. `git status --porcelain=v1`; `git log --oneline -5`; `git rev-parse HEAD bbc25db`; `git ls-remote origin main` (×2); `git rev-parse origin/main`; `git show --stat bbc25db`; `git diff-tree --no-commit-id --name-only -r bbc25db | grep -c 'workflows/ci.yml|gpu-nightly'`
5. `.venv/bin/python -m pytest tests/test_check_upstream_drift.py -v`
6. `http_proxy=https_proxy=…=http://127.0.0.1:9 .venv/bin/python -m pytest tests/test_check_upstream_drift.py -q` (no-network proof)
7. `.venv/bin/python -m pytest -m 'not gpu' -q`
8. `.venv/bin/python scripts/check_upstream_drift.py`
9. `cp docs/upstream-baseline.json /tmp/rc02-tampered-baseline.json` + heredoc flipping `pypi_version` → `9.9.9`, then `.venv/bin/python scripts/check_upstream_drift.py --baseline /tmp/rc02-tampered-baseline.json`
10. Dead-proxy run: `http(s)_proxy=http://127.0.0.1:9 .venv/bin/python scripts/check_upstream_drift.py --baseline /tmp/rc02-nonexistent-….json`
11. `.venv/bin/python scripts/check_upstream_drift.py --update-baseline --baseline /tmp/rc02-fresh-baseline.json` then `diff /tmp/rc02-fresh-baseline.json docs/upstream-baseline.json`
12. Pin-subset heredoc comparing `uses: …@<40-hex>` sets of the two workflows; `.venv/bin/python -m ruff check .`
13. Marker census: `pytest -m 'not gpu' --collect-only -q` vs `-m 'not gpu and not requires_download' --collect-only -q`; `grep -nE 'urlopen|urllib|socket|requests|mock|monk' tests/test_check_upstream_drift.py`
14. `/tmp` copies removed after use

## 6. Exit codes

- Live drift run (real baseline): **0** (`UNCHANGED`)
- Tampered-baseline run: **1** (DRIFT, `pypi_version` named)
- Dead-proxy script run: **2** (graceful `NETWORK ERROR` on stderr)
- `--update-baseline` to /tmp: **0** (`BASELINE UPDATED`)
- pytest new file: 0 (18 passed); dead-proxy pytest: 0 (18 passed); CPU suite: 0 (313 passed, 38 deselected)
- ruff: 0; all git read commands: 0 except one repeat `git ls-remote` that failed on the documented github.com:443 connect flake (135 s timeout) — the earlier successful `ls-remote` result (bbc25db on `refs/heads/main`) and the local `origin/main` ref are the evidence of record.

## 7. Runtime evidence inspected

Raw stdout/stderr of every command above, including: the verbatim `UNCHANGED`;
the verbatim drift banner + `{"pypi_version": {"baseline": "9.9.9", "current": "0.1.1"}}`;
the verbatim `NETWORK ERROR (exit 2 -- drift NOT determined): … Connection refused`;
`BASELINE UPDATED` + clean `diff` against the committed baseline; the four live
`inspect.signature` strings; pytest's per-test PASSED lines and the 313/38
summary; `yaml.safe_load`'s parsed trigger/permissions/runs-on; pin-set subset
`True`; wall-clock times (~2 min per networked run, matching the documented
pypi.org IPv6 TLS black-hole quirk — slow, not failed).

## 8. Regression tests

- New: `tests/test_check_upstream_drift.py` — 18/18 passed (0.03 s), also
  18/18 under a dead proxy.
- Full CPU suite `-m 'not gpu'`: 313 passed, 38 deselected, 0 failed, 0 errors
  — the exact expected post-task counts (295 + 18 = 313).
- `ruff check .` clean (what ci.yml's lint step runs on the now-pushed commit).

## 9. Claims audit (implementer report vs verified reality)

- "Push FAILED, main ahead 2" — **stale/false at verification time**: remote
  `refs/heads/main` is bbc25db; nothing unpushed. A later retry evidently
  landed. Deliverable state is correct; the report's push narrative is not.
- "313 passed, 38 deselected" — reproduced exactly.
- "18 tests, no markers, no network" — reproduced (no `pytestmark` in file;
  dead-proxy run green).
- "api_surface = 4 signatures via inspect.signature" — reproduced, and shown
  byte-equal to the installed package's live signatures.
- "pins are a subset of ci.yml's" — reproduced programmatically.
- "anonymous api.github.com works" — confirmed live (UNCHANGED run exercised
  commits/main + git/trees + pypi.org successfully).
- "pypi.org IPv6 quirk ⇒ ~2 min runs" — confirmed (2m06s/2m26s/2m07s walls).
- Undisclosed-in-brief but bounded deviations, all subordinate to the two
  sanctioned deltas: one-retry loop + `NetworkError` + `time` import,
  `BASELINE ERROR` exit-2 path, `main(argv=None)` parameter (enables in-process
  CLI tests), `chmod +x`, one justified `# noqa: BLE001`. None violate any
  criterion; all improve the exit-2 contract.

## 10. Problems found

None blocking. Two non-blocking notes:
1. Implementer report's push-status section is factually stale (push has since
   succeeded) — record-keeping inaccuracy only, no repo impact.
2. `.github/workflows/upstream-drift.yml` has not yet executed on a real
   GitHub runner (first trigger: Monday 03:17 UTC cron or manual dispatch).
   Not verifiable from this host by anyone; static validation (YAML parse,
   pin subset, step commands run locally under the same Python) all pass.
   A repeated `git ls-remote` during the end-check hit the known github.com:443
   flake, corroborating the documented host quirk.

## 11. Why PASS is justified

Every acceptance criterion was tested by execution, not by reading claims:
the real script was run against the live network for all four CLI outcomes
(0/1/2/update), the tampered-baseline drift named exactly the flipped field,
the disclaimer wording was observed verbatim in real output, the baseline's
API surface was proven byte-identical to the installed package's live
signatures, the committed baseline was proven reproducible byte-for-byte via
`--update-baseline`, both test suites produced the exact expected counts, the
tests were proven network-free under a dead proxy, the workflow YAML parses
with ci.yml-pinned actions and ci.yml untouched, the sole deviation set matches
the two sanctioned deltas (plus their mechanical consequences), and commit
bbc25db is pushed with a clean tree. No falsification attempt succeeded.
