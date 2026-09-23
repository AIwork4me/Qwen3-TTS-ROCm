# Second-architecture (gfx1100-class) hardware procurement evaluation — v0.3 planning

> **PLANNING DOCUMENT — not a capability claim; no hardware purchased; all prices as-of 2026-09-23.**
> Nothing in this file changes any repo capability claim, README statement, or
> compatibility-matrix row. It exists to support a purchase decision for the
> v0.3 North Star: validate official Qwen3-TTS on a SECOND, materially
> different Radeon architecture — preferred **gfx1100** (discrete RDNA3 with
> dedicated VRAM) vs the current validated host (gfx1151, Radeon 8060S iGPU,
> unified memory, ROCm 7.14.0, torch 2.12.0+rocm7.14.0).

Measured memory envelopes this decision is sized against (from program evidence):

| Workload | Measured GPU-memory peak | Consequence |
|---|---|---|
| 1.7B inference (worst model in ladder) | ~4.9 GiB | any 12 GB+ card is comfortable |
| Fine-tuning smoke run | 18.0–18.5 GiB allocated | needs ≥20 GB card, or batch-size reduction |

USD→CNY conversions below use ≈ ¥7.2/USD as a comparison aid only; CNY is the
procurement currency. Unverifiable figures are marked **UNVERIFIED** rather than guessed.

---

## 1. Executive recommendation

**Buy a used RX 7900 XTX 24 GB (gfx1100) and put it in a minimal new AM5
desktop tower (~¥10,000–11,000 total, ~$1,390–1,530).** It is the only option
that simultaneously:

1. lands exactly on the mission-preferred **gfx1100** axis (officially on the
   ROCm 7.14.0 support matrix — verified against AMD docs, see §2.2);
2. clears the fine-tune envelope with real headroom (18.5 GiB peak vs 24 GB);
3. runs on internal PCIe, so v0.3 benchmark/RTF numbers are **directly
   comparable** to the gfx1151 evidence (an eGPU path would force a
   "capability-only, RTF not comparable" disclosure on every number);
4. doubles as a **second self-hosted CI runner** (label `radeon-gfx1100`),
   extending the A–D GPU-CI work with a second architecture instead of
   leaving the laptop as a single point of failure.

Runner-up: **7900 XTX + USB4 eGPU on the existing ZBook (~¥8,800–10,100)** if
a second physical host is unacceptable — cheaper and desk-space-friendly, but
carries amdgpu-over-USB4 initialization risk and permanently taints every RTF
number with a ~PCIe 3.0 x4-class link that must be disclosed.
Budget floor: **used RX 7900 XT 20 GB + tower (~¥6,000–8,000)** — cheapest
gfx1100 path, but fine-tune headroom is 0.5–1.0 GiB over the measured peak,
i.e. batch-reduction fallback territory, not comfortable headroom.

**Timing note:** the 2026 GPU market is rising (GDDR7 shortage; used-card
index +5.7% in six weeks; used 7900 XTX moved $706→$900 in that window —
Sina/Bloomberg-sourced reporting via bestvaluegpu tracking, see §8). Waiting
likely costs more, not less. A ~$5–15 vast.ai dry-run (§4) can de-risk the
software path in parallel while procurement is approved.

---

## 2. GPU candidates

### 2.1 Candidates table (all support facts = AMD ROCm 7.14.0 official matrix, checked 2026-09-23)

| GPU | gfx target | RDNA gen | VRAM | On ROCm 7.14.0 official matrix | Price (new) | Price (used) | 1.7B inference (4.9 GiB) | Fine-tune (18.0–18.5 GiB) |
|---|---|---|---|---|---|---|---|---|
| **RX 7900 XTX** | **gfx1100** | RDNA3 (Navi31) | 24 GB | **Yes — supported** | $929+; CN ~¥7,299 (公版淘宝)–¥8,000 (JD AIB) | ~$900 US; 闲鱼 min listing ~¥6,788 | Yes | **Yes (~5.5 GiB headroom)** |
| **RX 7900 XT** | **gfx1100** | RDNA3 (Navi31) | 20 GB | **Yes — supported** | ~$850+; CN ~¥4,500–5,500 (JD XFX 海外版) | ~$610–650 US; CN est ¥3,000–4,000 (estimate, not a verified listing) | Yes | Marginal — exactly at the ≥20 GB floor; 0.5–1.0 GiB headroom |
| RX 7900 GRE | gfx1100 | RDNA3 | 16 GB | Yes — supported | CN channel is EOL/shrinking; 2026 price UNVERIFIED | UNVERIFIED | Yes | No |
| RX 7800 XT | gfx1101 (not gfx1100) | RDNA3 (Navi32) | 16 GB | Yes — supported | MSRP $499 | ~$499–523 (holding at MSRP) | Yes | No |
| RX 7700 XT | gfx1101 | RDNA3 (Navi32) | 12 GB | Yes — supported | UNVERIFIED this pass | UNVERIFIED | Yes (12 GB) | No |
| RX 7600 | gfx1102 | RDNA3 (Navi33) | 8 GB | **No — not on the 7.14.0 Linux matrix** (see §2.2 conflict note) | UNVERIFIED | UNVERIFIED | Tight — 8 GB total vs 4.9 GiB peak; not recommended | No |
| Radeon PRO W7900 (incl. Dual Slot) | gfx1100 | RDNA3 | 48 GB | Yes — supported | $3,499–3,999 | ~$3,495 (barely below new) | Yes | Yes (vast headroom) |
| Radeon PRO W7800 (32 GB; 48 GB variant exists) | gfx1100 (as listed on the official matrix) | RDNA3 | 32 GB | Yes — supported | ~$2,300–2,499 (Newegg) | UNVERIFIED | Yes | Yes |
| Radeon PRO W7700 | gfx1101 | RDNA3 | 16 GB | Yes — supported | $999 launch, often below | UNVERIFIED | Yes | No |
| RX 9070 XT | gfx1201 | **RDNA4** | 16 GB | **Yes — supported** | MSRP $599–649; Sep-2026 street ~$700–950+ (AMD Q3-2026 SEP raise); CN ¥5,699 (JD XFX 樱瞳花嫁OC) | RX 9070 non-XT: $648 | Yes | No |
| RX 9070 / 9070 GRE (CN) | gfx1201 | RDNA4 | 16 GB / 12 GB | Yes — supported | 9070: $649 new (Sep 2026); 9070 GRE CN price UNVERIFIED | $648 (9070) | Yes | No |
| RX 9060 XT / 9060 (+LP) | gfx1200 | RDNA4 | 8/16 GB | Yes — supported | UNVERIFIED this pass | UNVERIFIED | Yes (16 GB variant) | No |
| Radeon AI PRO R9700 | gfx1201 | RDNA4 | 32 GB | Yes — supported | **$1,299 US / ¥10,999 CN official retail** (JD, e.g. Gigabyte AI TOP 32G) | n/a (new product) | Yes | **Yes (comfortable)** |
| *(baseline)* Radeon 8060S (current host) | gfx1151 | RDNA3.5 APU | unified | n/a — current validated host | — | — | Yes (validated) | Yes at reduced batch (18–18.5 GiB alloc fits the 96 GB unified pool but competes with system RAM) |

### 2.2 ROCm official-support detail (the "validated stack" narrative)

Verified 2026-09-23 against the ROCm **7.14.0** Linux system-requirements /
compatibility matrix (page dated 2026-07-15, URL in §8):

- **Officially supported gfx1100 SKUs:** RX 7900 XTX, RX 7900 XT, RX 7900 GRE,
  Radeon PRO W7900 (incl. Dual Slot), PRO W7800 (both 32 GB and 48 GB).
- **Officially supported gfx1101 SKUs:** RX 7800 XT, RX 7700 XT, PRO W7700,
  PRO V710.
- **RDNA4 surprise — officially supported in the same matrix:** RX 9070 XT,
  9070, 9070 GRE (gfx1201), RX 9060 XT/9060 (+LP) (gfx1200), and the
  workstation **AI PRO R9700 / R9600D (gfx1201)**. RDNA4 support began with
  ROCm 6.4.1 (May 2025, initial) and matured through ROCm 7.x.
- **OS restriction:** the Radeon/Radeon-PRO entries are limited to Ubuntu
  24.04.4 / 22.04.5 and RHEL 10.1 / 9.7 — our stack is Ubuntu-class, so no
  conflict, but the runner image must stay within those rows.
- **Unofficial path:** AMD explicitly notes unlisted GPUs may work via
  TheRock community nightly builds, but "this enablement is not part of the
  official ROCm release" — unusable for our "validated official stack" claim.

**Conflict note — RX 7600 (gfx1102):** the 7.14.0 Linux install matrix we
fetched does **not** list it, while Phoronix reported official RX 7600
enablement arriving in a ROCm 7.12 tech preview and the ROCm Core SDK 10.0.0
release notes list gfx1102. Until it appears on the install matrix of the
release we actually run, treat RX 7600 as **not-matrix-supported** (UNVERIFIED
conflict; it was a `HSA_OVERRIDE_GFX_VERSION=11.0.0` spoof card for ~3 years,
and a 2025 firmware update broke that workaround). Also 8 GB VRAM is tight.
Practical verdict: excluded.

### 2.3 Is RDNA4 a better "materially different" axis than gfx1100?

RDNA4 (gfx1201) **is** officially supported and IS a bigger architectural
delta from gfx1151 than gfx1100 is (younger generation, different memory
subsystem, different compiler paths). Two facts keep gfx1100 as the mission
preference, and one caveat softens that:

1. The mission's North Star names gfx1100 explicitly — discrete RDNA3 with
   dedicated VRAM vs unified-memory iGPU — and gfx1100 is the axis AMD has
   supported longest for compute (since ROCm 5.7), so evidence produced on it
   ages best.
2. The affordable RDNA4 consumer card (RX 9070 XT, 16 GB) is
   **inference-only** for us; the fine-tune-capable RDNA4 card (AI PRO R9700,
   32 GB) costs ¥10,999 — more than a fine-tune-capable gfx1100 7900 XTX.
3. Caveat: if the program's hardware horizon extends past v0.3 (multi-year
   second-runner duty), the AI PRO R9700's 32 GB + newest generation has the
   longer service life — it is recorded in §6 as Alternate A, not as the
   primary recommendation, because it does not satisfy the gfx1100 North Star.

### 2.4 Used-market note (applies to every used row above)

China used-GPU market (闲鱼) risk profile: mining-era cards are mostly gone
from RDNA3 (it was unattractive for ETH mining), but 2026 risk is
datacenter/AI-farm cast-offs and 翻新 (refurb) cards; buy only with
transcript/screenshots of `amdgpu` + `rocminfo` output showing the exact gfx
target and VRAM, seller-verified personal-use history, and prefer cards with
remaining 国行 warranty (蓝宝石/撼讯/讯景 CN-registered). The ¥6,788 闲鱼 7900
XTX figure is a Bilibili market-tracker reported minimum listing, not a
verified transaction — treat CN used prices ±10% until a real listing is
negotiated.

---

## 3. Host paths

A discrete GPU cannot be installed inside the HP ZBook Ultra G1a 14 — the two
realistic paths are (a) USB4 eGPU on the existing laptop, (b) a second desktop
host. The ZBook's USB4 links run at 40 Gbps.

### 3.1 eGPU via USB4 on the ZBook G1a

Enclosure options (prices verified 2026-09-23 via vendor pages / egpu.io
September-2026 buyer's guide):

| Enclosure | Link | PSU | Price | Notes |
|---|---|---|---|---|
| AOOSTAR AG03 (or AG02) | USB4 40 Gbps | 800 W included | **¥1,499 (~$214)** | Dock-style; verify triple-slot 7900 XTX physical clearance |
| EXP GDC TH3P4G3 | TB4/USB4 | bring-your-own ATX PSU | ~$199 + PSU (~$80–120) | Bare board; cheapest flexible option, most cabling |
| Razer Core X V2 | TB5/USB4 | included | **$349.99 (~¥2,520)** | Proper enclosure, PCIe 4.0 x4, 140 W PD to laptop; original Core X being phased out |

**Bandwidth + comparability caveat (must be disclosed in any evidence
produced on this path):** USB4 40 Gbps tunnels PCIe at roughly PCIe 3.0
x4-class effective throughput (~3 GB/s real-world; engineering approximation,
not a vendor figure). Consequences:

- **Capability validation remains valid**: does official Qwen3-TTS run,
  produce correct output, and fit memory on gfx1100 — yes, that evidence
  stands. VRAM capacity is local to the card; bandwidth does not change
  memory fit, so the fine-tune envelope verdicts in §2.1 still hold.
- **RTF numbers are NOT directly comparable** to the gfx1151 internal-host
  evidence (nor to any internal-PCIe gfx1100 host). Every benchmark artifact
  from an eGPU run must carry the eGPU-link disclosure in its header.
- Kernel/model loading will see the biggest distortion (weight transfer over
  the link); steady-state inference distortion is workload-dependent.

**Linux amdgpu eGPU reliability (documented failure modes):**

- "Timeout on hotplug command 0x1038" with exactly a 7900 XTX + AOOSTAR AG02
  over USB4 (Arch BBS, Dec 2025).
- Laptop frequently failing to boot with the AMD eGPU attached after kernel
  upgrades (Framework community, Dec 2023) — ~1-in-10 cold-boot success
  reported by one user.
- Community workaround pattern: amdgpu does not hot-plug reliably; connect
  the eGPU before power-on, or plug in during early boot, then reboot
  (Linux Mint forums, Jun 2024; link trains at 40 Gbps when it works).

Net: the eGPU path is workable but operationally fragile — expect a
connect-at-boot (not hot-plug) workflow and pin the kernel version; that is
acceptable for a validation campaign, annoying for a nightly CI runner.

### 3.2 Second desktop host (minimal tower)

The host only needs to drive the GPU — no display workload, modest CPU.
Minimal new-build estimate (CN component prices, sources §8):

| Component | Choice | ~CNY |
|---|---|---|
| CPU | Ryzen 5 7500F (散片) | ~935 |
| Motherboard | A620M | ~620 |
| RAM | 16 GB DDR5 | ~300 |
| SSD | 512 GB NVMe | ~280 |
| Case | basic mATX tower | ~150 |
| PSU | 850 W gold (XTX needs AMD-rec 800 W; XT needs 750 W) | ~550–650 |
| **Tower total** | | **~¥2,850–3,300 (budget ¥3,000–3,500)** |

(Reference points: entry Ryzen office towers from ~¥1,549 and full DIY towers
from ~¥2,088 exist on Taobao/JD but those ship with GPUs/PSUs we do not
control; the self-build above guarantees the 850 W PSU headroom. Power facts:
7900 XTX TBP ~355 W, AMD-recommended PSU 800 W; 7900 XT TBP ~315 W,
recommended 750 W — per AMD product pages.)

Advantages over eGPU: internal PCIe (RTF comparability preserved), no USB4
fragility, and — decisively for this program — the tower registers as a
**second self-hosted GitHub Actions runner** (label `radeon-gfx1100`),
completing the A–D CI story with hardware redundancy: two architectures, two
machines, one queue. Disadvantages: desk/floor space, one more machine to
maintain, ~¥3–3.5K host cost that an eGPU path avoids.

---

## 4. Cloud reality (honest assessment)

- **gfx1100-class consumer instances ARE rentable, but only on community
  marketplaces:** vast.ai lists RX 7900 XTX offers at roughly
  **$0.066–0.20/GPU-hour** (computeprices.com tracker + vast.ai listing
  page). No mainstream provider (RunPod, Lambda, CoreWeave, Vultr,
  TensorWave) offers consumer RDNA3 — their AMD inventory is Instinct
  MI300X/MI325X/MI355X (gfx942-class **CDNA3**), which is a **different
  validation axis entirely** (different ISA lineage, different memory model,
  datacenter drivers). An MI300X run validates nothing about gfx1100.
- **What vast.ai is good for:** a cheap pre-purchase dry run — install the
  same pinned stack (torch rocm wheel matching the host's ROCm generation),
  run the v0.3 ladder end-to-end on gfx1100 for a few dollars (~$5–15),
  and surface any gfx1100-specific quirk before any hardware money moves.
- **Why it is not a substitute:** community marketplace = heterogeneous host
  drivers/ROCm versions we do not control (cannot anchor the "official
  supported stack, pinned versions" evidence narrative); no ownership (no
  second CI runner); cross-border payment/access friction from +08:00.
- **Verdict:** use vast.ai (if accessible) as a scouting tool in parallel
  with procurement; the North Star requires owned hardware.

---

## 5. Risks per option

| Option | Risks | Mitigations |
|---|---|---|
| Used 7900 XTX in tower (**R1**) | Used-card condition (AI-farm/翻新 stock in 2026 market); price volatility (used index +5.7% in 6 weeks); 355 W card needs real 850 W PSU + case airflow | Seller-verified `rocminfo` transcript before payment; 国行-warranty AIB cards preferred; buy promptly once approved; quality 850 W gold PSU in the ¥3–3.5K host budget |
| New 7900 XTX in tower | +¥500–1,500 over used for identical silicon; CN new stock of a 2022 card is shrinking (EOL trajectory like the GRE) | Watch JD AIB stock; used-with-warranty often the better trade |
| 7900 XTX + eGPU (**R2**) | amdgpu USB4 init/hotplug failures (documented above); kernel-upgrade regressions; benchmark comparability permanently caveated; no second host — laptop remains single point of failure | Connect-at-boot workflow; kernel pin; disclosure header on every benchmark artifact; accept capability-only scope |
| Used 7900 XT + tower (**R3**) | Fine-tune headroom 0.5–1.0 GiB over measured 18.0–18.5 GiB peak → fragmentation/transient spikes can OOM; same used-market risks | Batch-size reduction fallback already proven on gfx1151; treat fine-tune on this card as "smoke runs only", not a comfortable regime |
| AI PRO R9700 + tower (Alt A) | ¥10,999 + host; **not gfx1100** — does not satisfy the stated North Star without a mission re-scope; RDNA4 compute stack is younger (fewer long-tail fixes in the wild) | Only choose if the program re-scopes the second axis to "any officially-supported discrete arch" and prioritizes VRAM headroom |
| RX 9070 XT (any host) | 16 GB = inference-only; gfx1201 ≠ gfx1100 axis; Sep-2026 street price inflated by AMD's Q3-2026 SEP raise ($700–950+ vs $599 MSRP) | None needed — recorded as the cheapest NEW officially-supported card if the mission is later relaxed to RDNA4 inference-only |
| W7900 / W7800 | $2,300–3,995 for capability a ¥7K 7900 XTX already delivers; used W7900 barely discounts vs new | Excluded on cost-effectiveness; revisit only if >32 GB VRAM becomes a requirement |
| RX 7600 | Matrix conflict (not on 7.14.0 install matrix); 8 GB tight; 3-year history of override hacks | Excluded |
| vast.ai scouting | Driver/ROCm version roulette per host; no China-friendly payment guarantee | Filter offers by ROCm version; treat findings as directional, not evidence-grade |

---

## 6. Budget scenarios (ranked)

All totals include the §3.2 tower where a desktop host is used. CNY primary,
USD ≈ ¥7.2/USD.

| Rank | Configuration | Total budget | Unlocks | Feeds v0.3 / CI |
|---|---|---|---|---|
| **1** | **Used RX 7900 XTX 24 GB (闲鱼 ¥6,800–7,500, warranty AIB preferred) + minimal AM5 tower (¥3,000–3,500)** | **~¥9,800–11,000 (~$1,360–1,530)** | Full v0.3 ladder on gfx1100 **with directly comparable RTF** + fine-tune execution (18.5 GiB in 24 GB) + any future ≥24 GB workload | Second self-hosted runner `radeon-gfx1100`; nightly queue covers two architectures; laptop no longer single point of failure |
| 2 | New RX 7900 XTX 24 GB (CN ¥7,299–8,000) + AOOSTAR AG03 eGPU (¥1,499) [or Razer Core X V2 $349.99 ≈ ¥2,520] | **~¥8,800–10,100** | Full v0.3 ladder capability validation on gfx1100 + fine-tune-capable VRAM; RTF NOT directly comparable (mandatory eGPU disclosure) | No second host — same laptop, same runner; capability evidence only |
| 3 | Used RX 7900 XT 20 GB (CN est ¥3,000–4,500; US import ~$610–650) + minimal tower | **~¥6,000–8,000 (~$830–1,110)** | Cheapest gfx1100 path: full inference ladder + fine-tune smoke runs only (0.5–1.0 GiB headroom) | Second runner on gfx1100; fine-tune regime remains laptop-or-nothing |
| Alt A | Radeon AI PRO R9700 32 GB (¥10,999) + tower | ~¥14,000–14,500 | gfx1201 RDNA4 axis, 32 GB comfortable fine-tune, newest gen | Second runner `radeon-gfx1201`; requires mission re-scope away from gfx1100 |
| Alt B | vast.ai RX 7900 XTX dry-run | ~$5–15 | Pre-purchase software de-risk of the whole v0.3 ladder on gfx1100 | Directional only — not evidence-grade, no hardware |

**Recommendation: Rank 1.** If absolute budget ceiling is ¥8,000, take Rank 3
and accept smoke-run-only fine-tuning; if a second physical machine is
impossible, take Rank 2 and write the eGPU disclosure into every benchmark
header from day one. Alt A only under a re-scoped North Star.

---

## 7. How this feeds the v0.3 ladder and the second runner

- **v0.3 hardware ladder (diagnostics → tokenizer → 0.6B/1.7B CustomVoice →
  VoiceDesign → Base clones → reusable prompt → Design→Clone→Reuse →
  multilingual smoke → benchmark):** every candidate with ≥12 GB VRAM runs the
  entire ladder (§2.1 memory verdicts). The ladder's benchmark capstone is
  where the host choice bites: tower = numbers comparable to gfx1151
  evidence; eGPU = capability-only with disclosure.
- **Fine-tune execution (optional-later):** only ≥20 GB cards (R1/R2 XTX;
  R3 XT marginally; Alt A comfortably). The gfx1151 laptop already proved the
  batch-reduction fallback, so fine-tune is not the procurement driver —
  but R1 keeps it fully available on the second arch too.
- **Second self-hosted runner:** tower paths (R1/R3/Alt A) give the program a
  `radeon-gfx1100`-labeled runner using the exact A–D runbook (user-level
  systemd service, linger, night window) — the cheapest possible extension of
  the just-finished GPU-CI enablement, and it converts the second-arch
  validation from a one-off campaign into permanent regression coverage.

---

## 8. Sources (checked 2026-09-23)

**ROCm official support:**
- ROCm 7.14.0 Linux system requirements / compatibility matrix (gfx targets, supported Radeon/Radeon PRO SKUs, OS restrictions, TheRock unofficial note): https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- ROCm Core SDK 10.0.0 release notes (gfx1102 listing — RX 7600 conflict side): https://rocm.docs.amd.com
- Phoronix — ROCm 7.12 tech preview adds official RX 7600 support: https://www.phoronix.com
- Phoronix — ROCm 6.4.1 initial RDNA4 (gfx1200/1201) support (May 2025): https://www.phoronix.com
- AMD Radeon AI PRO R9700 product page (CN): https://www.amd.com/zh-cn/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html
- AMD RX 7900 XTX product page (800 W PSU recommendation, 355 W TBP): https://www.amd.com

**GPU prices (US):**
- BestValueGPU — RX 7900 XTX price history (used ~$900 Sep 2026, avg $825, low $500 on 2026-01-05, new $929+): https://bestvaluegpu.com
- BestValueGPU — RX 7900 XT (used $610, Sep 2026): https://bestvaluegpu.com
- GPU Dojo — W7900 (used ~$3,495, new ~$3,700) and RX 7900 XT (used $650/new $850): https://gpudojo.com
- Newegg — Radeon PRO W7800 32 GB from ~$2,300: https://www.newegg.com
- Tom's Hardware — W7700 launch $999: https://www.tomshardware.com
- Phoronix — AI PRO R9700 $1,299: https://www.phoronix.com
- eBay/Jawa/GPU Poet — RX 7800 XT used $499–523: https://www.ebay.com , https://jawa.gg , https://gpupoet.com
- Wccftech / Pangoly / GPU Poet — RX 9070 XT Sep-2026 street $700–950+, AMD Q3-2026 SEP raise, Aug-2026 avg ~$692: https://wccftech.com , https://pangoly.com
- BestValueGPU — RX 9070 $649 new / $648 used (Sep 2026): https://bestvaluegpu.com
- Sina Finance — used-GPU index +5.7% in six weeks, 7900 XTX $706→$900: https://finance.sina.com.cn

**GPU prices (CN):**
- JD AMD 新品显卡 channel — XFX RX 9070 XT 16G 樱瞳花嫁OC ¥5,699: https://pro.m.jd.com
- JD — Sapphire RX 7900 XTX 24G 白金 in-market listing (¥7–8K band, 10k+ reviews): https://www.jd.com
- 淘宝 RX 7900 XTX 公版全新 ~¥7,299; XFX 7900 XT 海外版全新 ¥4,500–5,500 (search-aggregated listings — verify at checkout)
- 闲鱼 7900 XTX minimum listing ~¥6,788 — Bilibili 海鲜市场行情 tracker (directional, not a transaction): https://www.bilibili.com

**eGPU:**
- egpu.io — September 2026 External GPU Buyer's Guide: https://egpu.io
- Razer Core X V2 ($349.99, TB5/USB4, PCIe 4.0 x4): https://www.razer.com
- AOOSTAR eGPU docks ($169–249; AG03 800 W, ~¥1,499, announced Dec 2025): https://aoostar.com
- EXP GDC TH3P4G3 (~$199, TB4/USB4 bare dock): https://www.amazon.com (also eBay)
- Arch BBS — RX 7900 XTX + AOOSTAR AG02 USB4, "Timeout on hotplug command 0x1038" (Dec 2025): https://bbs.archlinux.org
- Framework community — 7900 XTX eGPU boot/init failures (Dec 2023): https://community.frame.work
- Linux Mint forums — AMD eGPU connect-at-boot workaround, 40 Gbps link (Jun 2024): https://forums.linuxmint.com

**Desktop host (CN components):**
- 机核 gcores — Ryzen 5 7500F 散片 ~¥935: https://www.gcores.com
- 知乎 — A620M 主板 ~¥600+: https://zhuanlan.zhihu.com
- Taobao/JD — entry Ryzen towers from ~¥1,549, DIY towers from ~¥2,088: https://www.taobao.com , https://www.jd.com
- Level1Techs — 7900 XTX max platform draw ~375 W context: https://forum.level1techs.com

**Cloud:**
- vast.ai — RX 7900 XTX rental listing page: https://vast.ai (live grid is dynamic; rates via tracker below)
- computeprices.com — AMD GPU cloud pricing (~$0.066/hr cheapest 7900 XTX): https://computeprices.com
- cloud-gpus.com — 2026 AMD cloud landscape (Vultr MI300X/MI325X/MI355X, TensorWave; consumer RX scarcity): https://cloud-gpus.com
- AMD Developer Cloud (MI300X): https://www.amd.com

**Program-internal (not URLs):** measured memory envelopes (1.7B inference
~4.9 GiB; fine-tune smoke 18.0–18.5 GiB allocated) from program evidence;
gfx1151 host facts (HP ZBook Ultra G1a, Radeon 8060S, ROCm 7.14.0, torch
2.12.0+rocm7.14.0) from program records.

---

*End of planning document. As-of 2026-09-23. No hardware purchased; no
capability claims made or changed by this file.*
