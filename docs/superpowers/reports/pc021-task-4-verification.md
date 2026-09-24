# TASK 4 INDEPENDENT VERIFICATION — BENCHMARK V2: CONTROLLED REPRODUCIBILITY (v0.2.1)

**Verdict: PASS** — every gate reproduces independently and exactly: all 480 stored statistics (20 cells × {wall, audio, RTF} × 8 stats) recomputed from the JSON per-run arrays match to the last digit; both 20-row result tables (docs/benchmarks-v2.md and the evidence .txt stdout table) match the recomputed JSON cell-for-cell across all 13 columns; all five per-alias peak numbers match; structure, seed design, phase separation, and statistical-honesty requirements all hold; 4/4 new unit tests, 317-test CPU suite, and ruff all green; live GPU reruns of the script (n=2 and the historically broken n=1 path) both exit 0 with BENCHMARK-V2-OK and internally consistent JSON. Two non-blocking observations recorded below (a benign double-rounding artifact in 10/100 per-run RTF values, and GTT/system-RAM not directly sampled in the rocm-smi bookends) — neither is a data-integrity or overclaim issue.

Verifier: independent subagent, 2026-09-24. Repo at HEAD `8fbb5c2` (branch main), working tree clean except untracked `.work-docker/`. No tracked files modified, nothing committed (this report file is the only write outside /tmp). GPU work executed on this host (AMD Radeon 8060S, gfx1151, torch 2.12.0+rocm7.14.0, HIP 7.14.60850). Full recompute script: `/tmp/task4_verify_stats.py` (verbatim key outputs quoted below).

---

## Check 1 — JSON stats recomputation (THE core gate): 480/480 EXACT MATCH

Loaded `evidence/benchmark-v2-gfx1151-2026-09-24.json`, iterated EVERY cell of EVERY alias (5 aliases × 4 cells = 20 cells), and independently recomputed n/median/min/max/mean/stdev/p10/p90 from `cell["runs"]` using `statistics.median`, `statistics.fmean`, `statistics.stdev` (sample, n−1), and `statistics.quantiles(values, n=100, method="inclusive")` with `round(...,2)` — for all three stored blocks (`wall_stats`, `audio_stats`, `rtf_stats`). Comparison count = 20 cells × 3 blocks × 8 keys = 480; exact matches = 480; mismatches = 0.

Verbatim output of the verifier run:

```
stat comparisons   = 480, exact matches = 480
cells total        = 20   measured runs total = 100
stdev sample(n-1)  = confirmed; cells where pstdev would differ at 2dp: 10
```

Worked example (base en/short — the headline worst cell), stored per-run records and stored vs recomputed stats:

```
base en/short runs (seed, wall, audio, rtf):
   {'seed': 20260999, 'wall_seconds': 9.64, 'audio_seconds': 6.32, 'rtf': 1.52}
   {'seed': 20261000, 'wall_seconds': 8.08, 'audio_seconds': 5.12, 'rtf': 1.58}
   {'seed': 20261001, 'wall_seconds': 10.61, 'audio_seconds': 8.0, 'rtf': 1.33}
   {'seed': 20261002, 'wall_seconds': 9.21, 'audio_seconds': 6.0, 'rtf': 1.53}
   {'seed': 20261003, 'wall_seconds': 7.86, 'audio_seconds': 5.84, 'rtf': 1.35}
stored rtf_stats:  {'n': 5, 'median': 1.52, 'min': 1.33, 'max': 1.58, 'mean': 1.46, 'stdev': 0.11, 'p10': 1.34, 'p90': 1.56}
recomputed       :  {'n': 5, 'median': 1.52, 'min': 1.33, 'max': 1.58, 'mean': 1.46, 'stdev': 0.11, 'p10': 1.34, 'p90': 1.56}
```

## Check 2 — Markdown + transcript tables vs recomputed JSON: 20/20 rows each, ALL columns match

Parsed the results table in `docs/benchmarks-v2.md` and the stdout table in `evidence/benchmark-v2-gfx1151-2026-09-24.txt` (both exactly 20 data rows, identical row sets), and compared every column (alias/lang/len/n/median/min/max/mean/stdev/p10/p90/load/warmup) against the JSON-recomputed values:

```
md table rows=20  txt table rows=20
table cross-check issues: 0
```

Per-alias torch peaks — doc paragraph vs JSON (`peak_alloc_gb` stored at full float precision; doc's 2dp figures checked via `round(x,2)`):

```
peak custom-voice         doc=(4.46,4.58) json=(4.457061290740967,4.58) -> MATCH
peak custom-voice-0.6b    doc=(2.85,3.11) json=(2.8477530479431152,3.11) -> MATCH
peak voice-design         doc=(4.78,5.05) json=(4.784403324127197,5.05) -> MATCH
peak base                 doc=(5.12,5.41) json=(5.120804786682129,5.41) -> MATCH
peak base-0.6b            doc=(3.27,3.56) json=(3.268193244934082,3.56) -> MATCH
```

Claimed headline ranges all reproduce from the JSON: RTF medians 1.04 (cv-0.6b cn/short) … 1.52 (base en/short) ✓; per-cell stdevs 0.02–0.11 ✓; load 1.75–4.62 s ✓; peak alloc 2.85–5.12 GiB ✓; peak reserved 3.11–5.41 GiB ✓ (CHANGELOG, README, README_CN and both evidence-index rows quote exactly these ranges — verified against `git show 8fbb5c2` diffs).

## Check 3 — Structural checks: ALL PASS

- `meta.git_head = 88aa28067cda435038f2876c3aa9e01b8aae89cb` and `git rev-parse 88aa280` → `88aa28067cda435038f2876c3aa9e01b8aae89cb` — identical (the commit that added the single-sample guard is what the archived run recorded).
- `meta.seed_base = 20260924`, `meta.runs_per_cell = 5` present; `meta.prompt_manifest == v1 TEXTS` exactly (compared programmatically by importing `scripts/benchmark.py`), so the manifest is genuinely the fixed v1 grid.
- Every alias block has `load_seconds`, a `warmup` run record (with seed), exactly 4 cells with exactly 5 runs each (100 measured runs total), and `smi_before`/`smi_after` raw dicts whose VRAM strings are non-empty, contain digit-bearing `VRAM Total/Used` lines, and differ before vs after. Temperature/clock snapshots are present and real (edge temps across the 10 bookends: 50.0–66.0 °C; sclk/mclk levels recorded verbatim). No `failures` key in the JSON — consistent with the zero-failures claim.
- v1 script untouched: `git diff 64ab7f5~1..8fbb5c2 -- scripts/benchmark.py` → EMPTY. Commit stats confirm 64ab7f5 adds only `scripts/benchmark_v2.py` + `tests/test_benchmark_v2.py`; 88aa280 touches only those two files; 8fbb5c2 touches only docs/evidence/README/CHANGELOG. Backward compatibility preserved (v2 is a new script; the CI `full-weekly` replication path is untouched).

## Check 4 — Phase separation and seed design: PASS (one clarification vs the task brief)

The script's seed counter increments GLOBALLY across aliases (21 seeds per alias = 1 warmup + 4 cells × 5 runs), not per-alias. Verified: k runs 1..105; warmup k = {1, 22, 43, 64, 85} (one per alias, recorded once each, never in any measured cell); measured seeds = the remaining 100 values, all unique, zero overlap with warmup seeds:

```
global seed design: k=1..105, warmup ks=[1, 22, 43, 64, 85]
warmup seeds       = [20260925, 20260946, 20260967, 20260988, 20261009]
measured seed range= 20260926..20261029 unique=100
```

Note: the task brief sketched the expectation "seed_base+2..+101", which corresponds to a per-alias-reset counter; the implemented global counter is a different (and slightly stronger — no seed reuse across aliases) design that still satisfies every required property: fixed seed list, reproducible, warmup excluded from stats, no overlap. Not a discrepancy. `load_seconds` recorded per alias and in the claimed 1.75–4.62 s range; warmup wall times recorded separately per alias (3.54–9.75 s) and shown in the tables' "warmup s" column; measured stats are warm-model only.

## Check 5 — Statistical honesty: PASS

- Stdev is SAMPLE stdev (n−1): the script uses `statistics.stdev`; in 10 of the 20 cells `statistics.pstdev` would give a different 2dp value, and in none of those does the stored value equal the population variant — population-stdev contamination is ruled out.
- p10/p90 caveat present in `docs/benchmarks-v2.md` ("Statistical caveat (read before quoting p10/p90)… inclusive linear interpolations of a five-point sample — rough order-statistic hints… a p95 or 'worst-case latency' claim cannot be supported by n=5 and is deliberately not made"). A repo-wide grep over the touched docs finds NO p95/worst-case claims anywhere — the only occurrence of those strings is the caveat itself.
- No extrapolation beyond gfx1151: explicit "no extrapolation beyond gfx1151 is implied by any number here", plus a "what these numbers do NOT measure" section (TTFA, concurrency, energy, quality, other GPUs/hosts).
- Five aliases reported separately throughout (20-cell table, per-alias peaks, per-alias phases); no collapsed "Radeon performance" number anywhere — the doc states this is deliberate.

## Check 6 — Unit tests, CPU suite, lint: ALL GREEN

```
$ .venv/bin/python -m pytest tests/test_benchmark_v2.py -q
4 passed in 0.01s

$ .venv/bin/python -m pytest -m "not gpu" -q
317 passed, 40 deselected, 2 warnings in 17.53s

$ .venv/bin/python -m ruff check .
All checks passed!
```

## Check 7 — GPU spot-check (end-to-end script liveness): PASS, twice

Full one-alias rerun (`--aliases custom-voice-0.6b --runs 2 --json-out /tmp/task4-verify-v2.json`): exit code 0, `BENCHMARK-V2-OK`, 4 cells × 2 runs produced, e.g. `custom-voice-0.6b en/medium: RTF median=1.08 min=1.07 max=1.08 mean=1.08 stdev=0.01 p10=1.07 p90=1.08 (n=2)`; the spot JSON's stored stats independently recomputed: 32/32 match, 0 mismatch. Numbers differ from the archived run as expected (fresh session, fewer/different seed draws — structural success is what matters). Bonus: the historically broken `--runs 1` path (the `| tail -1`-masked failure fixed in 88aa280) rerun end-to-end: exit 0, all 4 cells print sane single-sample stats (`stdev=0.0`), `BENCHMARK-V2-OK` — the guard fix is confirmed in situ, not just in unit tests.

## Check 8 — Transcript accuracy and raw log: PASS

Three spot-checked `[v2]` progress lines (custom-voice cn/short, base en/short, base-0.6b en/medium) match the JSON-recomputed stats verbatim (`txt line spot MATCH` × 3). Raw log `/tmp/benchmark-v2-full.log` exists (642461 bytes, 11710+ lines), contains exactly one `BENCHMARK-V2-OK`, `V2_RC=0` at line 11710, zero `FAIL` lines, and its header (`gpu=AMD Radeon 8060S Graphics (gfx1151) torch=2.12.0+rocm7.14.0 … runs=5 seed_base=20260924 max_new_tokens=512`) matches the JSON meta.

---

## Observations (non-blocking)

1. **Per-run RTF double-rounding artifact (expected, benign).** In 10/100 measured runs, `rtf` differs by exactly 0.01 from `round(wall_seconds/audio_seconds, 2)` recomputed from the STORED (already 2dp-rounded) wall/audio values (max |diff| = 0.0100). Cause: the script computes RTF from the unrounded wall/audio before rounding all three for storage; with audio ≈ 3–8 s the theoretical bound is ≈0.0145, so all 10 diffs are within bound and internally consistent. The cell-level `rtf_stats` are computed from the stored (rounded) run values and match recomputation exactly, so this affects only the per-run `rtf` field's reconstructability from its rounded neighbors — a documentation-level nuance, not an integrity issue.
2. **GTT/system RAM not directly sampled.** The rocm-smi bookends record dedicated VRAM only (`--showmeminfo vram`; no GTT line present on this driver setup). On this iGPU with shared/unified memory, VRAM-used reads ~1.09–1.17 GiB even while torch peak allocated reaches 5.12 GiB — i.e., most allocator pages live in GTT/system RAM that the bookend does not itemize. The task's "GPU/system memory observations" is satisfied in substance (verbatim VRAM bookends + per-alias torch peak alloc/reserved + temp/clocks, nothing invented), but a future v3 could add `rocm-smi --showmeminfo all` (GTT line) or a `/proc/meminfo` sample to make the system-RAM side explicit. Nothing in the docs overstates what was captured — the doc itself labels the peaks as "torch peaks" and keeps smi output verbatim.

## Discrepancies

None. Zero mismatches across all quantitative gates (480/480 stats, 40 table rows × 13 columns, 5 peak pairs, seed/phase structure, all claimed ranges). The two items above are observations, not discrepancies: the first is a mathematically expected rounding artifact fully consistent with the stored data, the second is a scope note on an environment probe, with no associated overclaim.
