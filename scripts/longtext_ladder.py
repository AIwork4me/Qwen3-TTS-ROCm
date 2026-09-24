#!/usr/bin/env python3
"""Long-text generation ladder for qwen3-tts-rocm on gfx1151 (v0.2.1 Task 5A).

Measures how official-API synthesis behaves as input length grows — wall
time, output audio duration, RTF, torch peak memory, waveform sanity, and
whether generation ended naturally (EOS) or was capped by ``max_new_tokens``
— across a FIXED manifest of increasing text lengths (short / medium /
long / very-long) in English and Chinese.

Official-API only (``generate_custom_voice``; no internals are touched).
Token counts are NOT directly observable through the official return value
(waveforms only), so the manifest records CHARACTER counts and the
truncation verdict is an INFERENCE: a render whose audio duration reaches
~ (max_new_tokens / 12 Hz) is flagged ``hit_cap=True``; shorter renders are
flagged ``natural_eos_presumed=True``. Both labels say exactly what they
mean — no "maximum supported length" claim is made anywhere unless a real
boundary were established experimentally (none is).

Token-budget policy (binding): short/medium tiers keep the 512-token
latency guardrail like every GPU suite; long/very-long tiers pass the
OFFICIAL DEFAULT budget (no ``max_new_tokens`` kwarg; the checkpoint's
``generate_config.json`` default of 2048 applies) so long inputs are not
artificially beheaded — accepting that a degenerate sampling loop can run
long (documented upstream behavior; the run is bounded by the 2048 cap,
measured ≲ 23 min worst case on this iGPU).

Usage::

    .venv/bin/python scripts/longtext_ladder.py --alias custom-voice \
        --json-out evidence/longtext-custom-voice.json

Importing this module is lightweight (stdlib only); torch imports lazily.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import LANG_KEYS

#: Fixed ladder manifest — zh and en at four increasing lengths. Character
#: counts (incl. punctuation/spaces) are recorded per item in the output.
MANIFEST: dict[str, dict[str, str]] = {
    "zh": {
        "short": "今天的天气真不错，适合去公园散步。",
        "medium": ("今天下午我们去公园走了走，湖边的风很舒服，孩子追着鸽子跑，"
                    "笑得停不下来，回家的路上还买了一份刚出炉的烧饼。"),
        "long": ("清晨的城市慢慢醒来，街角的早餐铺已经排起了小队，蒸笼里冒着白气，"
                  "老板娘一边收钱一边招呼熟客。公交车进站，学生们背着书包挤上车，"
                  "有人低头看书，有人戴着耳机小声跟唱。太阳升高之后，写字楼玻璃上的"
                  "反光落在马路上，行人加快脚步，新的一天就这样开始了。"),
        "very-long": ("清晨的城市慢慢醒来，街角的早餐铺已经排起了小队，蒸笼里冒着白气，"
                       "老板娘一边收钱一边招呼熟客。公交车进站，学生们背着书包挤上车，"
                       "有人低头看书，有人戴着耳机小声跟唱。太阳升高之后，写字楼玻璃上的"
                       "反光落在马路上，行人加快脚步，新的一天就这样开始了。中午的菜市场"
                       "最热闹，鱼贩子吆喝着今天的鲫鱼新鲜，菜摊前老太太认真挑着青菜，"
                       "一毛两毛地讨价还价。傍晚河堤上有人跑步有人遛狗，路灯亮起来的时候，"
                       "晚风把白天的燥热一点点吹散。夜里的居民楼一扇扇窗户暗下去，只有"
                       "便利店的灯还亮着，给晚归的人留一份热乎的宵夜。"),
    },
    "en": {
        "short": "The weather is lovely today, perfect for a walk in the park.",
        "medium": ("This afternoon we took a slow walk around the lake. The breeze "
                    "was gentle, the kids chased pigeons and laughed the whole way, "
                    "and we stopped for a warm flatbread on the road home."),
        "long": ("The city wakes slowly in the early morning. A small queue has "
                  "already formed at the corner breakfast shop, steam rising from "
                  "the bamboo baskets while the owner greets regulars and makes "
                  "change at the same time. Buses pull in and students squeeze on "
                  "with their backpacks, some reading, others humming along with "
                  "their earbuds. As the sun climbs, reflections slide off the "
                  "office towers onto the street, footsteps quicken, and another "
                  "ordinary day begins."),
        "very-long": ("The city wakes slowly in the early morning. A small queue has "
                       "already formed at the corner breakfast shop, steam rising from "
                       "the bamboo baskets while the owner greets regulars and makes "
                       "change at the same time. Buses pull in and students squeeze on "
                       "with their backpacks, some reading, others humming along with "
                       "their earbuds. As the sun climbs, reflections slide off the "
                       "office towers onto the street, footsteps quicken, and another "
                       "ordinary day begins. By noon the wet market is at its loudest, "
                       "fish vendors shouting about the fresh crucian carp, grandmothers "
                       "picking through greens and haggling over dimes. In the evening "
                       "the river path fills with joggers and dog walkers, and when the "
                       "streetlights come on, the breeze finally melts the day's heat "
                       "away. One by one the apartment windows go dark, leaving only the "
                       "convenience store glowing, saving something warm for whoever "
                       "comes home late."),
    },
}

#: 12 Hz codec → 512 tokens ≈ 42.7 s, 2048 tokens ≈ 170.7 s of audio.
HZ = 12
CAP_TOLERANCE = 0.90  # duration >= 90% of the budget's audio ceiling → flagged hit_cap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Long-text ladder via the official API (v0.2.1 Task 5A)")
    ap.add_argument("--alias", default="custom-voice",
                    help="registry alias (default: custom-voice; CustomVoice path only)")
    ap.add_argument("--speaker-index", type=int, default=0,
                    help="index into get_supported_speakers() (default: 0)")
    ap.add_argument("--seed", type=int, default=20260924,
                    help="torch.manual_seed before each tier (fixed)")
    ap.add_argument("--json-out", default="evidence/longtext.json",
                    help="machine-readable output path")
    return ap.parse_args(argv)


def _budget_tokens(tier: str) -> int | None:
    """512 guardrail for short/medium; None (= official default 2048) for long tiers."""
    return 512 if tier in ("short", "medium") else None


def _run_tier(model, lang: str, tier: str, text: str, args) -> dict:
    import torch

    from qwen3_tts_rocm import testing

    torch.manual_seed(args.seed)
    kwargs: dict = {"text": text, "language": LANG_KEYS[lang]}
    alias_family = args.alias.split("-0.6b")[0]
    if alias_family == "custom-voice":
        kwargs["speaker"] = model.get_supported_speakers()[args.speaker_index]
        method = "generate_custom_voice"
    elif alias_family == "voice-design":
        from benchmark import VOICE_DESIGN_INSTRUCT
        kwargs["instruct"] = VOICE_DESIGN_INSTRUCT
        method = "generate_voice_design"
    else:
        raise SystemExit(f"longtext ladder drives CustomVoice/VoiceDesign entry points; "
                         f"got alias {args.alias!r}")
    budget = _budget_tokens(tier)
    if budget is not None:
        kwargs["max_new_tokens"] = budget
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    wavs, sr = getattr(model, method)(**kwargs)
    wall = time.perf_counter() - t0
    testing.assert_wav_sane(wavs[0], sr_expected=sr)
    dur = float(wavs[0].shape[-1]) / sr
    ceiling = (budget if budget is not None else 2048) / HZ
    hit_cap = dur >= CAP_TOLERANCE * ceiling
    return {
        "lang": lang, "tier": tier, "chars": len(text), "sr": sr,
        "wall_seconds": round(wall, 2), "audio_seconds": round(dur, 2),
        "rtf": round(wall / dur, 2),
        "token_budget": budget if budget is not None else "official_default(2048)",
        "audio_ceiling_seconds": round(ceiling, 2),
        "hit_cap": bool(hit_cap), "natural_eos_presumed": not hit_cap,
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available() else None,
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
        if torch.cuda.is_available() else None,
        "ok": True,
    }


def _rss_gb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return -1.0


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance probe never blocks the run
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    from qwen3_tts_rocm import loader

    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "torch": torch.__version__, "hip": str(torch.version.hip),
        "git_head": _git_head(), "alias": args.alias, "seed": args.seed,
        "manifest_chars": {lk: {t: len(x) for t, x in tv.items()}
                           for lk, tv in MANIFEST.items()},
        "truncation_note": ("hit_cap/natural_eos_presumed are inferences from audio "
                            "duration vs token budget at 12 Hz; the official API does "
                            "not expose token counts — no maximum-length claim made"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        meta["gpu"] = f"{p.name} ({getattr(p, 'gcnArchName', '?')})"

    print(f"== longtext ladder == alias={args.alias} seed={args.seed}")
    results: list[dict] = []
    failed = 0
    model = loader.load(args.alias)
    try:
        for lang in ("zh", "en"):
            for tier in ("short", "medium", "long", "very-long"):
                text = MANIFEST[lang][tier]
                try:
                    rec = _run_tier(model, lang, tier, text, args)
                except Exception:  # noqa: BLE001 - verbatim failure recording
                    tb = traceback.format_exc()
                    print(f"[longtext] FAIL {lang}/{tier}\n{tb}", flush=True)
                    results.append({"lang": lang, "tier": tier, "chars": len(text),
                                    "ok": False, "error": tb.strip().splitlines()[-1],
                                    "error_verbatim": tb})
                    failed += 1
                    continue
                rec["rss_gb_after"] = _rss_gb()
                results.append(rec)
                print(f"[longtext] OK {lang}/{tier} chars={rec['chars']} "
                      f"wall={rec['wall_seconds']}s audio={rec['audio_seconds']}s "
                      f"rtf={rec['rtf']} budget={rec['token_budget']} "
                      f"hit_cap={rec['hit_cap']} peak={rec['peak_alloc_gb']}GiB "
                      f"rss={rec['rss_gb_after']}GiB", flush=True)
    finally:
        loader.unload(model)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[longtext] wrote {out}", flush=True)

    print("| lang | tier | chars | wall s | audio s | RTF | budget | hit_cap | peak GiB | rss GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("ok"):
            print(f"| {r['lang']} | {r['tier']} | {r['chars']} | {r['wall_seconds']} | "
                  f"{r['audio_seconds']} | {r['rtf']} | {r['token_budget']} | "
                  f"{r['hit_cap']} | {r['peak_alloc_gb']} | {r['rss_gb_after']} |")
    if failed:
        print(f"ERROR: {failed} tier(s) failed (verbatim above)", file=sys.stderr)
        return 1
    print("LONGTEXT-LADDER-OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
