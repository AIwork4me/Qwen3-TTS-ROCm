#!/usr/bin/env python3
"""Reproducible RTF benchmark for qwen3-tts-rocm (Task 18).

Real-time factor (RTF) = wall-clock seconds / seconds of audio produced.
RTF > 1 means generation is slower than realtime; lower is better.  Wall time
covers ONE complete official-API call ``getattr(model, method)(...)`` -- text
encoding, decode, and codec-to-waveform conversion -- measured with
``time.perf_counter``.  Audio duration is ``len(wav) / sr`` of the returned
waveform.  Per-call first-token latency is NOT measurable through the official
non-streaming API and is deliberately omitted.

Methodology (binding decisions)
------------------------------
* Official calls are keyword-first; each alias uses its ONLY supported official
  entry point: ``generate_custom_voice`` / ``generate_voice_design`` /
  ``generate_voice_clone`` (the Base model rejects the other two).
* Sampling kwargs bound to the OFFICIAL-neutral worst case:
  ``do_sample=True, temperature=1.0`` explicitly; ``top_k/top_p``,
  ``repetition_penalty`` and sub-talker knobs stay with the model's own
  ``generate_config.json`` defaults (nothing tuned for speed).
* ``max_new_tokens`` defaults to 512 (the project latency guardrail): at 12 Hz
  this caps any render near ~42 s of audio while bounding degenerate-loop wall
  time (T13 measured a single 2048-token runaway at ~23 min on this iGPU).
* Warmup policy: one cn/short generation per alias, DISCARDED (kernel + code
  path warmup), then every (language x length) cell is measured
  ``--runs`` times (default 2).  The model is loaded once per alias via
  :func:`qwen3_tts_rocm.loader.load` (bf16 + sdpa HIP defaults) and unloaded
  before the next alias.
* Numbers drift with clock, thermals and background load on an iGPU; treat any
  single run as indicative.  See docs/benchmarks.md for interpretation.

Usage::

    .venv/bin/python scripts/benchmark.py                 # defaults below
    .venv/bin/python scripts/benchmark.py --no-warmup \
        --aliases custom-voice,base --max-new-tokens 256
    .venv/bin/python scripts/benchmark.py --aliases custom-voice-0.6b,base-0.6b \
        --json-out evidence/benchmark-06b-<date>.json     # the 0.6B pair

Outputs: markdown-ready table rows on stdout AND a JSON document (default
``evidence/benchmark.json``) with {meta:{host,gpu,torch_version_hip,date,args,
git_head,upstream_qwen3_tts_sha,per_alias},results:[per-cell records incl.
per-run RTF lists]}.  Per-alias metrics -- ``load_seconds`` (the timed
``loader.load(alias)``) and ``peak_alloc_gb`` (torch's peak allocated GiB
across the alias's whole run, reset right after load) -- are printed per
alias, stored under meta.per_alias, and attached to every cell of that alias
(``None`` when no CUDA device is visible).

Note for tests: importing this module is intentionally LIGHTWEIGHT (stdlib
only at import time); torch/qwen_tts/loader are imported lazily inside
functions so unit tests can pin the pure RTF/statistics helpers anywhere.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "BASE_REF_TEXT",
    "LANG_KEYS",
    "TEXTS",
    "UPSTREAM_QWEN3_TTS_SHA",
    "VOICE_DESIGN_INSTRUCT",
    "cell_summary",
    "main",
    "md_row",
    "rtf_of",
    "with_alias_metrics",
]

# ---------------------------------------------------------------------------
# Benchmark payload constants (inline by design: no external fixtures needed).
# ---------------------------------------------------------------------------

#: Text samples per language key -> {"short" (~20 chars), "medium" (~60 chars)}.
TEXTS: dict[str, dict[str, str]] = {
    "cn": {
        "short": "今天的天气真不错，适合去公园散步。",
        "medium": (
            "杭州西湖的春天格外迷人，苏堤两畔的柳枝随微风轻轻摆动，"
            "三三两两的游人沿着湖边缓步而行，尽情享受着这段难得的午后时光。"
        ),
    },
    "en": {
        "short": "Today is a good day.",
        "medium": "The quick brown fox jumps over the lazy dog near the riverbank.",
    },
}

#: Language-key -> raw language name accepted by the official API.
LANG_KEYS: dict[str, str] = {"cn": "Chinese", "en": "English"}

#: Fixed style instruction for voice-design (instruct is mandatory there).
VOICE_DESIGN_INSTRUCT = "用平静自然的语气说话"

#: Approximate transcript of the bundled synthetic reference clip used by the
#: Base alias (mirrors tests/test_voice_clone_workflow.py; known limitation:
#: the clip is speech-like babble, not a real human recording).
BASE_REF_TEXT = "This tiny synthetic voice was cloned for automated testing."

#: Upstream QwenLM/Qwen3-TTS HEAD SHA at the Task 0 ground-truth audit
#: (2026-09-20; evidence/ground-truth-2026-09-20.md).  Recorded verbatim into
#: the JSON meta so every archived benchmark names the upstream it measured
#: against without re-deriving it at run time.
UPSTREAM_QWEN3_TTS_SHA = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

#: Default alias list (registry aliases from qwen3_tts_rocm.models.ALIASES):
#: the three 1.7B entry points.  The 0.6B checkpoints are fully supported via
#: ``--aliases custom-voice-0.6b,base-0.6b`` (see build_call); they are not in
#: the default because the default run mirrors the published 1.7B numbers in
#: docs/benchmarks.md, and the 0.6B pair has its own archived evidence run
#: (evidence/benchmark-06b-2026-09-20.json).
DEFAULT_ALIASES = "custom-voice,voice-design,base"

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_RUNS = 2
DEFAULT_JSON_OUT = "evidence/benchmark.json"

MD_HEADER = "| alias | lang | len | runs | median RTF | best RTF | worst RTF | median wall s |"
MD_SEP = "|---|---|---|---|---|---|---|---|"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_benchmark.py without any GPU/torch).
# ---------------------------------------------------------------------------


def rtf_of(wall_s: float, audio_s: float) -> float:
    """Real-time factor: wall seconds per second of audio (>1 slower than realtime).

    Non-positive audio duration is impossible for sane output; mapping it to
    infinity keeps the failure loud instead of hiding it behind a fake 0.
    """
    if audio_s <= 0:
        return float("inf")
    return wall_s / audio_s


def cell_summary(walls: list[float], audios: list[float]) -> dict:
    """Aggregate one (alias, lang, len) cell's runs into reportable statistics."""
    rtfs = [rtf_of(w, a) for w, a in zip(walls, audios)]
    return {
        "runs": len(rtfs),
        "wall_s": [round(w, 3) for w in walls],
        "audio_s": [round(a, 3) for a in audios],
        "rtf_values": [round(r, 4) for r in rtfs],
        "median_rtf": round(statistics.median(rtfs), 4),
        "min_rtf": round(min(rtfs), 4),
        "max_rtf": round(max(rtfs), 4),
        "median_wall_s": round(statistics.median(walls), 3),
        "median_audio_s": round(statistics.median(audios), 3),
    }


def md_row(alias: str, lang_key: str, length_key: str, summary: dict) -> str:
    """One markdown-ready table row (printed verbatim into evidence + docs)."""
    return (
        f"| {alias} | {lang_key} | {length_key} | {summary['runs']} "
        f"| {summary['median_rtf']:.2f} | {summary['min_rtf']:.2f} "
        f"| {summary['max_rtf']:.2f} | {summary['median_wall_s']:.2f} |"
    )


def with_alias_metrics(cell: dict, *, load_seconds: float,
                       peak_alloc_gb: float | None) -> dict:
    """Attach the per-alias metrics to one result *cell* (pure rounding only).

    ``load_seconds`` is the timed ``loader.load(alias)``; ``peak_alloc_gb`` is
    torch's peak allocated GiB across that alias's whole run.  ``None`` for
    the peak (CPU host / probe failure) is recorded verbatim so the absence
    stays visible instead of masquerading as a measurement.  Mutates and
    returns *cell* so callers can append it in one expression.
    """
    cell["load_seconds"] = round(float(load_seconds), 3)
    cell["peak_alloc_gb"] = (None if peak_alloc_gb is None
                             else round(float(peak_alloc_gb), 3))
    return cell


def reset_peak_gpu_memory() -> None:
    """Zero torch's peak-allocation counter; a no-op without a CUDA device.

    Guarded by ``torch.cuda.is_available()`` (lazy import inside) so CPU unit
    tests of the pure helpers stay lightweight.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_alloc_gb() -> float | None:
    """Peak GPU bytes allocated (GiB) since the last reset; ``None`` off-GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30


# ---------------------------------------------------------------------------
# Environment metadata (guarded, mirrors src/qwen3_tts_rocm/env.py probes).
# ---------------------------------------------------------------------------


def _gpu_description() -> str:
    """Human-readable GPU line from whatever torch actually reports (never raises)."""
    try:
        import torch

        props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        if props is None:
            return "unavailable"
        arch = getattr(props, "gcnArchName", None) or "?"
        cus = getattr(props, "multi_processor_count", None)
        vram_gib = getattr(props, "total_memory", 0) / 1024**3
        return f"{props.name} ({arch}, multi_processor_count={cus}, torch-visible memory {vram_gib:.1f} GiB)"
    except Exception as exc:  # noqa: BLE001 - diagnostics never block the bench
        return f"probe failed: {exc}"


def _git_head() -> str:
    """``git rev-parse HEAD`` of the repo the script runs in; never raises.

    Falls back to ``"unknown"`` outside a git checkout or when git is absent,
    so the benchmark never dies on provenance probing.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001,S110 - provenance probes never block the bench
        pass
    return "unknown"


def collect_meta(args: argparse.Namespace, aliases: list[str]) -> dict:
    """JSON 'meta' section: pinned platform facts + exact CLI args (pure read)."""
    import torch  # lazy: keep module import lightweight for unit tests

    host_ram_gib = 0.0
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    host_ram_gib = int(line.split()[1]) / 1024**2
                    break
    except OSError:  # pragma: no cover - non-Linux probing only
        pass

    def _cpu_name() -> str:
        """Marketing CPU name from /proc/cpuinfo (platform.processor() often
        returns the bare arch 'x86_64' on Linux, which is useless here)."""
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name\t: "):
                        return line.split("\t: ", 1)[1].strip()
        except OSError:  # pragma: no cover - non-Linux probing only
            pass
        return platform.processor() or "?"

    return {
        "host": f"{platform.node()} ({platform.platform()})",
        "cpu": _cpu_name(),
        "gpu": _gpu_description(),
        "torch_version_hip": f"{torch.__version__} (HIP {getattr(torch.version, 'hip', None)})",
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "upstream_qwen3_tts_sha": UPSTREAM_QWEN3_TTS_SHA,
        "args": {
            **vars(args),
            "json_out": str(args.json_out),
            # resolved values that fully determine the run:
            "aliases_resolved": aliases,
            "texts": TEXTS,
            "sampling_kwargs": {
                "do_sample": True,
                "temperature": 1.0,
                "top_k/top_p/repetition_penalty": "official generate_config.json defaults",
            },
            "dtype_attn_device": "loader.load defaults: bfloat16 + sdpa on cuda:0",
            "warmup_policy": "one discarded cn/short generation per alias when --warmup",
        },
        "host_total_ram_gib": round(host_ram_gib, 1),
    }


# ---------------------------------------------------------------------------
# Generation plumbing (all heavy imports lazy).
# ---------------------------------------------------------------------------


def _load_base_ref_audio() -> tuple[object, int]:
    """(wav float32, sr) from the bundled license-clean synthetic ref clip."""
    from importlib.resources import files

    import soundfile as sf

    path = Path(str(files("qwen3_tts_rocm") / "demo" / "assets" / "ref_en.wav"))
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return wav.astype("float32"), int(sr)


def build_call(model: object, alias: str, text: str, language: str) -> tuple[str, dict]:
    """Resolve (official-method-name, kwargs) for *alias*; keyword-first calls.

    Each alias maps to its family's ONLY supported official entry point
    (size variants of the same family share it): ``custom-voice`` and
    ``custom-voice-0.6b`` -> ``generate_custom_voice``; ``voice-design`` ->
    ``generate_voice_design``; ``base`` and ``base-0.6b`` ->
    ``generate_voice_clone`` (the Base models reject the other two).

    Raises KeyError for unknown aliases so typos fail loudly before any load.
    """
    common = {"text": text, "language": language}
    if alias in ("custom-voice", "custom-voice-0.6b"):
        return "generate_custom_voice", {**common, "speaker": model.get_supported_speakers()[0]}
    if alias == "voice-design":
        return "generate_voice_design", {**common, "instruct": VOICE_DESIGN_INSTRUCT}
    if alias in ("base", "base-0.6b"):
        wav, sr = _load_base_ref_audio()
        return "generate_voice_clone", {**common, "ref_audio": (wav, sr), "ref_text": BASE_REF_TEXT}
    raise KeyError(f"unknown alias {alias!r}; benchmark supports: custom-voice, "
                   f"custom-voice-0.6b, voice-design, base, base-0.6b")


def run_cell(
    model: object, alias: str, lang_key: str, length_key: str, max_new_tokens: int, n_runs: int
) -> tuple[str, dict]:
    """Warm-measured cell: `n_runs` timed generations; returns (text, summary)."""
    text = TEXTS[lang_key][length_key]
    language = LANG_KEYS[lang_key]
    method, call_kwargs = build_call(model, alias, text, language)
    gen = getattr(model, method)

    walls: list[float] = []
    audios: list[float] = []
    run_meta: list[dict] = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        wavs, sr = gen(**call_kwargs, do_sample=True, temperature=1.0, max_new_tokens=max_new_tokens)
        wall = time.perf_counter() - t0
        wav = wavs[0]
        audio = len(wav) / int(sr)
        rtf = rtf_of(wall, audio)
        walls.append(wall)
        audios.append(audio)
        run_meta.append(
            {"run": i + 1, "wall_s": round(wall, 3), "audio_s": round(audio, 3), "rtf": round(rtf, 4)}
        )
        print(
            f"[bench] alias={alias} lang={lang_key} len={length_key} "
            f"run={i + 1}/{n_runs} wall={wall:.2f}s audio={audio:.2f}s "
            f"rtf={rtf:.2f}",
            flush=True,
        )
    return method, cell_summary(walls, audios) | {"runs_detail": run_meta}


def warmup(model: object, alias: str) -> None:
    """Discarded cn/short generation (kernel/code-path warmup per alias).

    WARNING: this deliberately passes NO ``max_new_tokens``, so the model's
    own ``generate_config.json`` default (2048) applies.  On ``base`` a
    degenerate sampling loop can then burn ~23 minutes in this single call
    (measured: 1396 s on gfx1151); keeping it uncapped reproduces the
    official-default degenerate behavior as recorded evidence.  Pass
    ``--no-warmup`` to skip it.
    """
    method, kwargs = build_call(model, alias, TEXTS["cn"]["short"], LANG_KEYS["cn"])
    getattr(model, method)(**kwargs, do_sample=True, temperature=1.0)


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="benchmark",
        description="RTF benchmark across registered Qwen3-TTS aliases (Task 18)",
    )
    ap.add_argument(
        "--aliases",
        default=DEFAULT_ALIASES,
        help=f"comma-separated registry aliases (default: {DEFAULT_ALIASES})",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="official sampling cap per render (default: %(default)s; "
        "512 = project latency guardrail, ~42s audio ceiling)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="measured generations per text sample (default: %(default)s)",
    )
    ap.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="discard one warmup generation per alias (default: on)",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=Path(DEFAULT_JSON_OUT),
        help=f"where the JSON report lands (default: {DEFAULT_JSON_OUT})",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
    if not aliases or args.runs < 1:
        print("ERROR: need at least one alias and >=1 run per sample", file=sys.stderr)
        return 2

    # Heavy/lazy imports happen here, NOT at module import time.
    from qwen3_tts_rocm import loader, models

    unknown = [a for a in aliases if a not in models.ALIASES]
    if unknown:
        print(f"ERROR: unknown alias(es) {unknown}; known: {list(models.ALIASES)}", file=sys.stderr)
        return 2

    meta = collect_meta(args, aliases)
    print("== qwen3-tts-rocm RTF benchmark ==")
    print(f"# date={meta['date']}")
    print(f"# host={meta['host']}")
    print(f"# gpu={meta['gpu']}")
    print(f"# torch={meta['torch_version_hip']}")
    print(
        f"# args: aliases={','.join(aliases)} max_new_tokens={args.max_new_tokens} "
        f"warmup={args.warmup} runs={args.runs}"
    )

    results: list[dict] = []
    rows: list[str] = []
    per_alias: dict[str, dict] = {}  # alias -> {load_seconds, peak_alloc_gb}
    for alias in aliases:
        print(f"[bench] loading alias={alias} ...", flush=True)
        t0 = time.perf_counter()
        model = loader.load(alias)
        load_seconds = time.perf_counter() - t0
        print(f"[bench] loaded alias={alias} took={load_seconds:.1f}s", flush=True)
        # Peak counter reset AFTER load: the high-water mark then reflects the
        # alias's whole measured run (warmup + cells), not the load itself.
        reset_peak_gpu_memory()
        alias_results: list[dict] = []
        try:
            if args.warmup:
                tw = time.perf_counter()
                warmup(model, alias)
                print(
                    f"[bench] warmup alias={alias} took={time.perf_counter() - tw:.1f}s (discarded)",
                    flush=True,
                )
            for lang_key in ("cn", "en"):
                for length_key in ("short", "medium"):
                    method, summary = run_cell(
                        model, alias, lang_key, length_key, args.max_new_tokens, args.runs
                    )
                    alias_results.append(
                        {
                            "alias": alias,
                            "method": method,
                            "lang": lang_key,
                            "len": length_key,
                            "text": TEXTS[lang_key][length_key],
                            **summary,
                        }
                    )
                    rows.append(md_row(alias, lang_key, length_key, summary))
        finally:
            # Peak read at alias end (before unload drops the weights): the
            # high-water mark of the whole run. None on a CPU-only host.
            peak = peak_alloc_gb()
            per_alias[alias] = {
                "load_seconds": round(load_seconds, 3),
                "peak_alloc_gb": None if peak is None else round(peak, 3),
            }
            peak_repr = "n/a (no CUDA device)" if peak is None else f"{peak:.2f} GiB"
            print(
                f"[bench] alias={alias} load_seconds={load_seconds:.1f} "
                f"peak_alloc_gb={peak_repr}",
                flush=True,
            )
            for cell in alias_results:
                with_alias_metrics(cell, load_seconds=load_seconds,
                                   peak_alloc_gb=peak)
            results.extend(alias_results)
            print(f"[bench] unloading alias={alias}", flush=True)
            loader.unload(model)

    meta["per_alias"] = per_alias

    print(MD_HEADER)
    print(MD_SEP)
    for row in rows:
        print(row)

    out = {"meta": meta, "results": results}
    json_out: Path = args.json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[bench] wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
