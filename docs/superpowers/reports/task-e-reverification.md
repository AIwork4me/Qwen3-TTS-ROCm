# Task E re-verification — fix round 1 (procurement evaluation)

- Date: 2026-09-21
- Fix commit under review: `8660f314b120fadee9411860efda3c3c68871d11` ("docs: procurement-evaluation corrections (Razer Core X V2 ships without PSU; memory phrasing; rank-2 band)")
- Scope: re-review of the four prior findings against the fix diff and the updated doc; no code or doc changes made by the reviewer.

## Hygiene checks (all pass)

| Check | Result |
|---|---|
| Diff touches only the evaluation doc | PASS — 1 file: `docs/hardware-v0.3-procurement-evaluation.md` (+25/−18) |
| `8660f31` == `origin/main` == `HEAD` | PASS — all three resolve to `8660f314b120fadee9411860efda3c3c68871d11` |
| Working tree clean | PASS — `git status --porcelain` empty |
| CPU suite `.venv/bin/python -m pytest -m 'not gpu' -q` | PASS — **313 passed, 38 deselected** (16.4 s), matches expected exactly |

## Per-finding verdicts

### Finding 1 — Razer Core X V2 PSU factually wrong ("PSU: included") — **ADDRESSED**

- §3.1 row (line 158) now reads "**NOT included — ships without a PSU** (Razer official FAQ; Tom's Hardware)" with all-in pricing: `$349.99 (~¥2,520) + ~$80–120 (~¥600–900) ATX PSU ≈ $430–470 all-in (~¥3,100–3,400)`.
- Arithmetic verified: $349.99+$80=$429.99≈$430; $349.99+$120=$469.99≈$470. At ¥7.2/USD: $430→¥3,096≈¥3,100; $470→¥3,384≈¥3,400; PSU alone $80–120→¥576–864≈¥600–900. All consistent with the doc's stated rate.
- Rank-2 (§6, line 267) recomputed and split: XTX ¥7,299–8,000 + AG03 ¥1,499 = ¥8,798–9,499 → "~¥8,800–9,500 (AG03)"; XTX + Core X V2 all-in ¥3,100–3,400 = ¥10,399–11,400 → "~¥10,400–11,400 (Core X V2 incl. PSU)". The runner-up paragraph (line 45) and the Rank-2 bracket now carry the same ¥10,400–11,400 figure.
- Grep for `PSU: included` in the doc: **zero hits**. Remaining "PSU included" mentions (AG03 "800 W included" line 155; "(¥1,499, PSU included)" line 267) refer to the AOOSTAR AG03, which does ship with a PSU — factually correct, not stale phrasing.

### Finding 2 — "18.0–18.5 GiB allocated" conflated allocated vs reserved — **ADDRESSED**

- §1 envelope table now reads "peak allocated 18.01–18.03 GiB / reserved 18.48–18.50 GiB (evidence/upstream-372-e2e-run{1,2}.json)"; §1 runner point, §2.1 table header, §5 R3 risk, §6 ranks 1/3, and the §8 program-internal note all use the split phrasing (report claims 9 locations; spot-verified across all sections).
- Independently verified against the cited evidence files:
  - run1: `torch_cuda_max_memory_allocated_gib` 18.013, `..._reserved_gib` 18.477
  - run2: 18.026 / 18.496
  - Ranges 18.01–18.03 / 18.48–18.50 match to two decimals.
- "Needs ≥20 GB card, or batch-size reduction" conclusion unchanged.
- Related headroom figures were recomputed with a correct GB→GiB conversion (24 GB ≈ 22.4 GiB → ~4 GiB headroom vs reserved 18.5 GiB; 20 GB ≈ 18.6 GiB usable → ~0.1–0.15 GiB headroom). Verified: 24×10⁹/2³⁰=22.35; 20×10⁹/2³⁰=18.63. This tightens the old "~5.5 GiB"/"0.5–1.0 GiB" numbers in the conservative direction — consistent with the finding, not new breakage.

### Finding 3 — Rank-2 upper bound loose (8,000+2,520=10,520 vs stated ~10,100) — **ADDRESSED**

- The single "~¥8,800–10,100" band is gone (grep: zero hits). Rank 2 now carries both component-consistent bands: AG03 7,299+1,499=8,798→~8,800 / 8,000+1,499=9,499→~9,500; Core X V2 7,299+3,100=10,399→~10,400 / 8,000+3,400=11,400.
- Spot-checked the other ranks for internal consistency: Rank 1 6,800+3,000=9,800 / 7,500+3,500=11,000 → "~¥9,800–11,000" ✓; Rank 3 3,000+3,000=6,000 / 4,500+3,500=8,000 → "~¥6,000–8,000" ✓.

### Finding 4 — §8 domain-level URLs — **recorded, intentionally NOT fixed (confirmed unchanged)**

- §8 still lists domain-level URLs (razer.com, aoostar.com, amazon.com, bbs.archlinux.org, amd.com, etc.). The only §8 change adds a factual note ("ships WITHOUT a PSU — Razer official FAQ + Tom's Hardware") and a tomshardware.com domain URL to the Razer line. This stays within the recorded-won't-fix disposition; no unintended URL changes.

## New-breakage scan — none found

- PLANNING header intact (lines 3–8: "PLANNING DOCUMENT — not a capability claim; no hardware purchased…"; header/disclaimer greps still present, 3 occurrences).
- No capability language added: the diff touches only cost figures, memory phrasing, headroom arithmetic, rank bands, and source annotations. Rank-2 "capability evidence only" / "RTF NOT directly comparable (mandatory eGPU disclosure)" wording is unchanged.
- Every numeric claim touched moves in the conservative direction (Razer more expensive; 7900 XT headroom smaller; 7900 XTX headroom restated at ~4 GiB instead of an inflated ~5.5 GiB) — all evidence-backed, none loosen a risk statement.
- Commit message accurately describes the three changes.

## Final verdict

**ALL ADDRESSED** (findings 1–3 ADDRESSED; finding 4 recorded as intentionally not fixed, disposition respected).

**New breakage: NO.**
