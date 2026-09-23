# Task E verification — second-architecture procurement evaluation

- Verifier: independent (no subagents, no repo modifications other than this report).
- Date: 2026-09-23. Document under test: `docs/hardware-v0.3-procurement-evaluation.md` @ commit `297b930`.
- Implementer report cross-read: `.superpowers/sdd/gpu-ci-enablement/task-e-report.md`.

## Verdict

**PASS — with one minor factual error flagged for correction** (Razer Core X V2
"PSU included" is wrong; see Problems). The error sits in a secondary
alternative column, does not touch the primary recommendation, and no criterion
gate fails on it.

## Criteria checklist

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | PLANNING-DOCUMENT header + as-of date; no capability claims; README/matrix untouched | PASS | Line 3 banner: "PLANNING DOCUMENT — not a capability claim; no hardware purchased; all prices as-of 2026-09-23"; repeated in footer (line 350–351). Grep for validated/✅/supported/verified: every hit is (a) AMD's ROCm matrix (external fact), (b) price/support-matrix verification language, or (c) the pre-existing validated gfx1151 host fact; no repo capability claims, no ✅ anywhere, all future-tense/planning framing. `git show 297b930 --stat`: exactly one file, `docs/hardware-v0.3-procurement-evaluation.md` (+351); README and all other files untouched. |
| 2 | Source integrity — 4 load-bearing claims re-verified independently | PASS (3/4 fully, 1 partial — see below) | See "Claims re-verified" section. |
| 3 | Architecture IDs correct; "materially different from gfx1151" framing sound | PASS | Live ROCm 7.14.0 matrix page confirms: 7900 XTX/XT/GRE and W7900/W7800 = gfx1100; 7800 XT/7700 XT/W7700 = gfx1101; RDNA4 = gfx1200 (9060 XT) / gfx1201 (9070 series, AI PRO R9700/R9600D). RX 7600 = gfx1102 (Navi33) per standard AMD GCD naming and the Core SDK 10.0.0 listing the doc cites; absent from the 7.14.0 matrix, consistent with the doc's conflict note. Framing sound: gfx1151 is the RDNA3.5 Strix-Halo iGPU (Radeon 8060S, unified memory) vs discrete RDNA3 with dedicated VRAM — exactly the axis distinction the doc draws (§2.3, header). |
| 4 | Memory envelope traces to measured ~18.0–18.5 GiB peak | PASS (minor wording nit) | `evidence/upstream-372-e2e-run1.json`: `torch_cuda_max_memory_allocated_gib = 18.013`, `reserved = 18.477`; run2: `18.026` / `18.496`. Doc's "18.0–18.5 GiB" envelope is correct; the ≥20 GB fine-tune requirement follows. Nit: the doc labels the whole range "allocated", but the 18.5 upper bound is the *reserved* figure (allocated is 18.01–18.03). Inference figure also checks out: `benchmark-smoke-2026-09-21.json` `peak_alloc_gb = 4.927` ≈ doc's ~4.9 GiB. |
| 5 | UNVERIFIED/conflict handling | PASS | RX 7600: §2.2 conflict note leaves it UNVERIFIED / "not-matrix-supported until proven" and §5 excludes it (matches the live matrix — it is indeed absent). RDNA4: §2.3 + §5 Alt A + §6 Alt A record it as alternate requiring explicit mission re-scope away from gfx1100. Cloud: §4 records gfx1100 as community-marketplace-only (vast.ai) with the MI300X/MI325X/MI355X gfx942-CDNA "different validation axis entirely / An MI300X run validates nothing about gfx1100" caveat. |
| 6 | Completeness | PASS | Candidates table (§2.1, 14 rows incl. baseline host); host paths (§3) with the laptop constraint explicit ("A discrete GPU cannot be installed inside the HP ZBook Ultra G1a 14"); risks (§5 table); budget scenarios (§6) with Rank 1 = ~¥9,800–11,000 (arithmetic checks: ¥6,800–7,500 + ¥3,000–3,500); sources-with-URLs (§8). |
| 7 | Commit/tree/CPU suite | PASS | `297b930` == local HEAD == local `origin/main` == `git ls-remote origin main` (`297b930190e32e670a89b06ceaba72ca13020463`); `git status --porcelain` empty; `.venv/bin/python -m pytest -m 'not gpu' -q` → **313 passed, 38 deselected** in 16.81s, exit 0. |

## Claims re-verified (4, with my independent sources)

1. **RX 7900 XTX on the ROCm 7.14.0 official support matrix (gfx1100)** — REPRODUCED.
   Fetched https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
   (production 7.14.0 docs, page dated 2026-07-15): 7900 XTX listed, RDNA3,
   gfx1100. The same fetch reproduced the doc's entire matrix narrative:
   7900 XT/GRE gfx1100; 7800 XT/7700 XT gfx1101; W7900 (+Dual Slot)/W7800
   (32/48 GB) gfx1100; W7700/V710 gfx1101; RX 7600 **not listed**; RDNA4
   entries present; Radeon/Radeon-PRO OS footnote = Ubuntu 24.04.4/22.04.5,
   RHEL 10.1/9.7 (exactly as the doc §2.2 states); TheRock
   community-nightly note with the "not part of the official ROCm release"
   wording the doc quotes.
2. **Used RX 7900 XTX price ~$900 (Sep 2026), moved from $706** — REPRODUCED.
   TweakTown (Sep 2026) reporting eBay used pricing: "RX 7900 XTX rose from
   $706 to $900", part of a broader used-market surge; corroborated by
   BestValueGPU showing RX 7900 XT at $610 used (Sep 2026), matching the
   doc's §8 figure. The rising-market timing note is directionally confirmed.
3. **eGPU enclosure prices** — PARTIAL (one detail disproved).
   Razer Core X V2: $349.99, TB5/USB4, PCIe 4.0 x4, 140 W PD, original Core X
   being phased out — all REPRODUCED (Razer product page/coverage, Tom's
   Hardware Jul 2025). **BUT the doc's §3.1 table says "PSU: included" for the
   Core X V2 — Razer's own FAQ states "The Razer Core X V2 does not include a
   power supply"** (Tom's Hardware and egpu.io concur; no power cable ships in
   the box). AOOSTAR AG03: REPRODUCED — dual TB5/USB4v2 + OCuLink dock with
   800 W PSU included, announced Dec 2025, ~$249 street (MinixPC Apr 2026),
   consistent with the doc's ¥1,499 (~$214).
4. **RDNA4 / RX 9070 XT officially supported (gfx1201) on the same ROCm 7.14.0 matrix** — REPRODUCED.
   Same AMD docs fetch as claim 1: RX 9070 XT / 9070 / 9070 GRE listed as
   gfx1201, RX 9060 XT (+LP) as gfx1200, AI PRO R9700 / R9600D as gfx1201.
   The doc's "RDNA4 surprise" framing and the ROCm-6.4.1 origin (May 2025)
   are consistent with public reporting.

## Problems

1. **Factual error (must fix at next doc revision, not gating):** §3.1
   enclosure table lists the Razer Core X V2 "PSU: included". Razer's official
   FAQ says it does NOT include a power supply (verified via Razer support
   page, Tom's Hardware Jul-2025 launch coverage, egpu.io unboxing). The
   Razer-based eGPU variant therefore costs ~$80–120 (~¥600–900) more than the
   doc implies, and §6 Rank 2's "[or Razer Core X V2 $349.99 ≈ ¥2,520]"
   comparison figure inherits the omission. The primary Rank-2 config
   (AOOSTAR AG03, PSU genuinely included) and the Rank-1 recommendation are
   unaffected.
2. **Minor labeling imprecision:** the memory table says "18.0–18.5 GiB
   allocated"; 18.5 is the *reserved* peak (allocated is 18.01–18.03 GiB).
   Envelope and the ≥20 GB conclusion are unaffected. Relatedly, 7900 XT
   headroom stated as 0.5–1.0 GiB treats 20 GB as 20 GiB; in GiB (18.6) the
   true headroom is smaller (~0.1–0.6 GiB) — this makes the doc's own
   "marginal, batch-reduction territory" verdict more, not less, correct.
3. **Weak provenance practice (not fabrication):** several §8 source entries
   are domain-level URLs (phoronix.com, bestvaluegpu.com, finance.sina.com.cn,
   bilibili.com, jd.com) rather than deep links to the specific
   articles/listings. I could not deep-audit those from the URLs alone;
   however, I independently reproduced every load-bearing number they back
   (claims 1–4), so no unverifiable load-bearing claim remains.
4. **Nit:** Rank 2 upper-bound arithmetic (¥8,000 + ¥2,520 = ¥10,520 vs stated
   ~¥10,100) is loose within its "~" band — immaterial, and compounded by
   problem 1 for the Razer variant.

## Justification

Every criterion gate passes on independent evidence: the commit is exactly
`297b930` == origin/main with a clean tree and a green 313/38 CPU suite; the
doc is a properly fenced planning document that changed no repo claims; its
four load-bearing external facts (ROCm 7.14 matrix membership and gfx IDs,
used XTX pricing, enclosure pricing, RDNA4 support) reproduce from AMD's own
docs and independent Sep-2026 price reporting; the memory envelope traces to
the program's measured 18.013–18.496 GiB evidence; and the conflict handling,
completeness, and budget arithmetic all check out. The single reproducible
error found (Razer Core X V2 ships without a PSU, contradicting the doc's
"included") affects only a secondary alternative column and the comparison
figure for a non-recommended configuration, so it warrants a flagged
correction rather than a failed verification.
