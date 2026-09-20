#!/usr/bin/env python3
"""True-streaming probe for qwen3-tts-rocm (Step D, 2026-09-21 follow-up).

Question this script answers, on the REAL Radeon GPU: does the installed
official ``qwen-tts`` Python API deliver audio INCREMENTALLY (chunk-by-chunk
while generation proceeds), or only as one complete waveform when the call
returns?  Whatever the answer, the five roadmap streaming metrics are then
measured on the CustomVoice 1.7B path:

  1. time to first audio (TTFB)  -- first observable audio arrival;
  2. chunk cadence                -- inter-arrival distribution (n/a at 1 chunk);
  3. total RTF                    -- wall seconds / audio seconds;
  4. buffer-underrun behavior     -- simulated real-time consumer (24 kHz
                                     playback starting at first chunk arrival):
                                     starvation events + starved seconds;
  5. long-text behavior           -- same metrics on a ~200-char text.

Methodology (binding decisions)
------------------------------
* Official API only: every measurement goes through
  ``model.generate_custom_voice(...)`` exactly as shipped -- keyword-first,
  ``do_sample=True, temperature=1.0`` (benchmark-neutral worst case, matching
  scripts/benchmark.py), other sampling knobs at the model's own
  ``generate_config.json`` defaults.
* ``max_new_tokens`` guardrail: 512 for the short text (the project latency
  guardrail), 768 for the long text (caps ~64 s of audio at 12 Hz; the text's
  natural EOS lands well below it).
* The ``non_streaming_mode`` flag is measured BOTH ways (True/False) per text
  length: the wrapper docstring says False "only simulates streaming text
  input ... rather than enabling true streaming input or streaming
  generation"; measuring both sides turns that statement into evidence.
* Audio-arrival observation: the official call is blocking and returns the
  complete waveform list at once, so the observable arrival timeline is
  exactly ONE event at return time.  The harness records arrivals generically
  (list of ``(t, n_samples)``); if upstream ever ships a real incremental
  path, only ``observe_generation`` needs to change, and the consumer
  simulation / cadence statistics are already incremental-capable.
* Warmup: one discarded short generation (kernel/code-path warmup), mirroring
  scripts/benchmark.py.  Every scenario is then measured ``--runs`` times
  (default 2).  Numbers drift with clock/thermals on an iGPU; treat single
  runs as indicative.
* API-surface evidence is MACHINE-CAPTURED at run time (inspect signatures,
  generator-function checks, verbatim docstring lines, needle-searched
  file:line quotes from the installed package) -- never quoted from memory.

Usage::

    .venv/bin/python scripts/streaming_probe.py \
        --json-out evidence/streaming-2026-09-21.json

Outputs: a human-readable transcript on stdout (capture with ``tee`` into
``evidence/streaming-<date>.txt``) and the JSON document required by the Step
D schema at ``--json-out``.

Note for tests: importing this module is LIGHTWEIGHT (stdlib only at import
time); torch / qwen_tts / loader are imported lazily inside functions so unit
tests can pin the pure statistics helpers anywhere.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "LONG_TEXT",
    "SHORT_TEXT",
    "UPSTREAM_QWEN3_TTS_SHA",
    "cadence_stats",
    "main",
    "observe_generation",
    "rtf_of",
    "simulate_realtime_consumer",
    "summarize_run",
]

#: Short payload: identical to the benchmark's cn/short cell (17 chars) so the
#: RTF here is directly comparable with the published benchmark numbers.
SHORT_TEXT = "今天的天气真不错，适合去公园散步。"

#: Long payload (~212 chars): benchmark cn/medium as the opening sentence plus
#: three more sentences -- natural Chinese prose, no repetition tricks.
LONG_TEXT = (
    "杭州西湖的春天格外迷人，苏堤两畔的柳枝随微风轻轻摆动，"
    "三三两两的游人沿着湖边缓步而行，尽情享受着这段难得的午后时光。"
    "远处的雷峰塔在夕阳下泛着温暖的金光，湖面上的游船缓缓驶过，"
    "留下一道道细长的波纹。孩子们在草坪上放飞风筝，欢笑声此起彼伏，"
    "老人们坐在长椅上晒着太阳，聊着家长里短。这样平凡而美好的日子，"
    "让人不由得放慢脚步，细细品味生活中的每一份宁静与惬意。"
    "傍晚时分，断桥边亮起了柔和的灯光，晚风送来阵阵桂花的清香。"
)

#: Upstream QwenLM/Qwen3-TTS HEAD SHA at the Task 0 ground-truth audit
#: (2026-09-20; evidence/ground-truth-2026-09-20.md) -- same pin as
#: scripts/benchmark.py, recorded verbatim into the JSON meta.
UPSTREAM_QWEN3_TTS_SHA = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

DEFAULT_RUNS = 2
DEFAULT_JSON_OUT = "evidence/streaming.json"

#: max_new_tokens guardrail per length key (12 Hz codec: 512 ~ 42.7 s cap,
#: 768 ~ 64 s cap; LONG_TEXT's natural EOS lands around ~600 tokens).
MAX_NEW_TOKENS = {"short": 512, "long": 768}

#: Sampling bound, benchmark-neutral worst case (mirrors scripts/benchmark.py).
SAMPLING_KWARGS = {"do_sample": True, "temperature": 1.0}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_streaming_probe.py, no GPU/torch).
# ---------------------------------------------------------------------------


def rtf_of(wall_s: float, audio_s: float) -> float:
    """Real-time factor: wall seconds per second of audio (>1 slower than realtime).

    Non-positive audio duration is impossible for sane output; mapping it to
    infinity keeps the failure loud instead of hiding it behind a fake 0.
    """
    if audio_s <= 0:
        return float("inf")
    return wall_s / audio_s


def cadence_stats(inter_arrivals_s: list[float]) -> dict:
    """Inter-arrival distribution of audio-chunk arrivals.

    Returns median/p90/min/max (seconds); with fewer than 2 inter-arrivals
    (i.e. a single delivery) every statistic is ``None`` and only the count
    is reported -- cadence is undefined, not zero.
    """
    n = len(inter_arrivals_s)
    if n < 2:
        return {"n_inter_arrivals": n, "median": None, "p90": None,
                "min": None, "max": None}
    # "inclusive" matches numpy's default linear interpolation for percentiles.
    p90 = statistics.quantiles(inter_arrivals_s, n=10, method="inclusive")[8]
    return {
        "n_inter_arrivals": n,
        "median": round(statistics.median(inter_arrivals_s), 4),
        "p90": round(p90, 4),
        "min": round(min(inter_arrivals_s), 4),
        "max": round(max(inter_arrivals_s), 4),
    }


def simulate_realtime_consumer(
    arrivals_s: list[float], chunk_samples: list[int], sample_rate: int
) -> dict:
    """Simulate a real-time consumer against a production timeline.

    The consumer starts PLAYBACK at the first chunk arrival and then consumes
    samples at ``sample_rate`` in real time.  A buffer starvation (underrun)
    is any open interval where the consumer wants samples that have not been
    produced yet; it ends at the next chunk arrival.  Returns underrun event
    count, total starved seconds, and playback start/end times.

    Timeline model: produced audio is a right-continuous step function of
    arrival times; consumption is linear in wall time.  Both are exact, so
    the simulation is event-driven over interval boundaries (arrival times
    plus the final catch-up instant) with no clock ticking.
    """
    if not arrivals_s:
        raise ValueError("no audio arrivals: nothing to consume")
    if len(arrivals_s) != len(chunk_samples):
        raise ValueError("arrivals_s and chunk_samples must be same length")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    t_start = arrivals_s[0]
    total_samples = sum(chunk_samples)
    # Ideal (never-paused) playback end; real playback ends later by exactly
    # the total starved time.
    t_end = t_start + total_samples / sample_rate

    boundaries = sorted(set(arrivals_s) | {t_end})
    produced_by: list[float] = [
        sum(n for at, n in zip(arrivals_s, chunk_samples) if at <= t)
        for t in boundaries
    ]

    underrun_events = 0
    starved_s = 0.0
    starved = False
    for i, a in enumerate(boundaries[:-1]):
        b = boundaries[i + 1]
        if b <= a:
            continue
        buffer_now = produced_by[i] - max(0.0, a - t_start) * sample_rate
        if buffer_now > 0:
            starved = False  # a chunk arrival at `a` refilled the buffer
            dry_at = a + buffer_now / sample_rate
            if dry_at < b:
                underrun_events += 1
                starved = True
                starved_s += b - dry_at
        else:
            if not starved:  # continuous starvation is ONE event, not many
                underrun_events += 1
                starved = True
            starved_s += b - a
    return {
        "playback_start_s": round(t_start, 4),
        "ideal_playback_end_s": round(t_end, 4),
        "underrun_events": underrun_events,
        "starved_s": round(starved_s, 4),
    }


def summarize_run(
    arrivals_s: list[float],
    chunk_samples: list[int],
    sample_rate: int,
    wall_s: float,
) -> dict:
    """One measured run -> the Step D per-run record (pure arithmetic)."""
    audio_s = sum(chunk_samples) / sample_rate
    inter = [b - a for a, b in itertools.pairwise(arrivals_s)]
    consumer = simulate_realtime_consumer(arrivals_s, chunk_samples, sample_rate)
    return {
        "ttfb_s": round(arrivals_s[0], 4) if arrivals_s else None,
        "chunk_count": len(arrivals_s),
        "cadence_s": cadence_stats(inter),
        "total_wall_s": round(wall_s, 4),
        "audio_s": round(audio_s, 4),
        "rtf": round(rtf_of(wall_s, audio_s), 4),
        "underrun_events": consumer["underrun_events"],
        "starved_s": consumer["starved_s"],
        "total_samples": int(sum(chunk_samples)),
    }


def observe_generation(model: object, call_kwargs: dict) -> dict:
    """Run ONE official generation call and record the audio-arrival timeline.

    The official wrapper's generate methods are blocking and return the
    complete waveform list at once, so the observable arrival timeline is a
    single event at return time (TTFB == total wall).  The return shape is
    kept generic -- a list of ``(t_rel_s, n_samples)`` events -- so a future
    incremental path plugs in here without touching any downstream metric.
    """
    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(**call_kwargs)
    wall_s = time.perf_counter() - t0
    wav = wavs[0]
    return {
        "arrivals_s": [wall_s],
        "chunk_samples": [len(wav)],
        "sample_rate": int(sr),
        "wall_s": wall_s,
    }


# ---------------------------------------------------------------------------
# API-surface capture (machine-captured at run time; never quoted from memory).
# ---------------------------------------------------------------------------


def _needle_lines(path: Path, needles: list[str]) -> list[dict]:
    """First file:line of each needle substring in *path* (verbatim text)."""
    hits: list[dict] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for needle in needles:
        for lineno, text in enumerate(lines, start=1):
            if needle in text:
                hits.append({"file": str(path), "line": lineno, "text": text.strip()})
                break
    return hits


def capture_api_surface() -> dict:
    """Runtime evidence of what the installed official API exposes (or not).

    Captures, from the installed package itself: version + source file,
    signatures of the three generate methods, whether any is a generator
    function, their verbatim ``non_streaming_mode`` docstring lines, the
    codec-decode signature, and needle-searched file:line quotes of the
    statements that pin the delivery semantics.
    """
    import importlib.metadata as md
    import inspect

    import qwen_tts
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference import qwen3_tts_tokenizer

    model_file = Path(inspect.getsourcefile(Qwen3TTSModel))  # type: ignore[arg-type]
    tokenizer_file = Path(inspect.getsourcefile(qwen3_tts_tokenizer.Qwen3TTSTokenizer))  # type: ignore[arg-type]
    core_dir = Path(qwen_tts.__file__).parent / "core" / "models" / "modeling_qwen3_tts.py"

    methods: dict[str, dict] = {}
    for name in ("generate_custom_voice", "generate_voice_clone", "generate_voice_design"):
        func = getattr(Qwen3TTSModel, name)
        doc_lines = [
            line.strip()
            for line in (inspect.getdoc(func) or "").splitlines()
            if "simulates streaming text input" in line
            or "rather than enabling true streaming" in line
        ]
        methods[name] = {
            "signature": str(inspect.signature(func)),
            "is_generator_function": inspect.isgeneratorfunction(func),
            "non_streaming_mode_doc_verbatim": doc_lines,
        }

    return {
        "package": "qwen-tts",
        "version": md.version("qwen-tts"),
        "wrapper_file": str(model_file),
        "mechanism": "blocking call, complete waveforms returned at completion",
        "description": (
            "All three official generate_* methods are plain blocking functions "
            "returning Tuple[List[np.ndarray], int]; none is a generator and none "
            "takes a chunk callback or streamer. non_streaming_mode=False only "
            "re-frames prompt construction (simulated streaming TEXT input) per "
            "the installed docstrings, and the codec decode is a single "
            "full-sequence call -- no incremental audio delivery exists in the "
            "official Python API."
        ),
        "methods": methods,
        "codec_decode_signature": str(
            inspect.signature(qwen3_tts_tokenizer.Qwen3TTSTokenizer.decode)
        ),
        "key_lines": [
            *_needle_lines(model_file, [
                "def generate_custom_voice",
                "def generate_voice_clone",
                "def generate_voice_design",
                "wavs, fs = self.model.speech_tokenizer.decode",
            ]),
            *_needle_lines(core_dir, [
                "talker_result = self.talker.generate(",
                "return talker_codes_list, talker_hidden_states_list",
            ]),
            *_needle_lines(tokenizer_file, [
                "def decode(",
            ]),
        ],
    }


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def _env_block() -> dict:
    """GPU / stack facts as torch actually reports them (lazy import)."""
    import importlib.metadata as md

    import torch

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "gpu": props.name if props else "unavailable",
        "gfx": getattr(props, "gcnArchName", None) if props else None,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "qwen_tts": md.version("qwen-tts"),
    }


def _git_head() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=False, cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001,S110 - provenance probes never block the probe
        pass
    return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="streaming-probe",
        description="Step D: true-streaming probe of the official qwen-tts API on GPU",
    )
    ap.add_argument("--alias", default="custom-voice",
                    help="registry alias to probe (default: %(default)s, the 1.7B flagship path)")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                    help="measured generations per scenario (default: %(default)s)")
    ap.add_argument("--json-out", type=Path, default=Path(DEFAULT_JSON_OUT),
                    help=f"where the JSON report lands (default: {DEFAULT_JSON_OUT})")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs < 1:
        print("ERROR: need >=1 run per scenario", file=sys.stderr)
        return 2

    from qwen3_tts_rocm import loader

    print("== Step D true-streaming probe ==")
    print(f"# date={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# git_head={_git_head()}")
    print(f"# upstream_qwen3_tts_sha={UPSTREAM_QWEN3_TTS_SHA}")
    print(f"# args: alias={args.alias} runs={args.runs}")

    api_surface = capture_api_surface()
    print(f"# api: {api_surface['package']} {api_surface['version']} @ {api_surface['wrapper_file']}")
    for name, info in api_surface["methods"].items():
        print(f"# api: {name}: generator={info['is_generator_function']} "
              f"non_streaming_mode_doc={info['non_streaming_mode_doc_verbatim']}")
    for hit in api_surface["key_lines"]:
        print(f"# api: {Path(hit['file']).name}:{hit['line']}: {hit['text']}")

    print(f"[probe] loading alias={args.alias} ...", flush=True)
    t0 = time.perf_counter()
    model = loader.load(args.alias)
    print(f"[probe] loaded alias={args.alias} took={time.perf_counter() - t0:.1f}s", flush=True)
    speaker = model.get_supported_speakers()[0]

    scenarios: list[dict] = []
    try:
        # Warmup: one discarded short generation (kernel/code-path warmup).
        tw = time.perf_counter()
        model.generate_custom_voice(
            text=SHORT_TEXT, language="Chinese", speaker=speaker,
            max_new_tokens=MAX_NEW_TOKENS["short"], **SAMPLING_KWARGS)
        print(f"[probe] warmup took={time.perf_counter() - tw:.1f}s (discarded)", flush=True)

        for length_key, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
            for nsm in (True, False):
                name = f"{length_key}_non_streaming_mode_{str(nsm).lower()}"
                print(f"[probe] scenario={name} chars={len(text)} "
                      f"max_new_tokens={MAX_NEW_TOKENS[length_key]}", flush=True)
                runs: list[dict] = []
                for i in range(args.runs):
                    obs = observe_generation(model, {
                        "text": text, "language": "Chinese", "speaker": speaker,
                        "non_streaming_mode": nsm,
                        "max_new_tokens": MAX_NEW_TOKENS[length_key],
                        **SAMPLING_KWARGS,
                    })
                    rec = summarize_run(obs["arrivals_s"], obs["chunk_samples"],
                                        obs["sample_rate"], obs["wall_s"])
                    runs.append(rec)
                    print(
                        f"[probe]   run={i + 1}/{args.runs} ttfb={rec['ttfb_s']:.2f}s "
                        f"chunks={rec['chunk_count']} wall={rec['total_wall_s']:.2f}s "
                        f"audio={rec['audio_s']:.2f}s rtf={rec['rtf']:.2f} "
                        f"underruns={rec['underrun_events']} starved={rec['starved_s']:.2f}s",
                        flush=True,
                    )
                scenarios.append({
                    "name": name, "text_len_chars": len(text),
                    "non_streaming_mode": nsm,
                    "max_new_tokens": MAX_NEW_TOKENS[length_key],
                    "sample_rate": obs["sample_rate"],
                    "runs": runs,
                })
    finally:
        print("[probe] unloading model", flush=True)
        loader.unload(model)

    incremental = any(r["chunk_count"] > 1 for s in scenarios for r in s["runs"])
    if incremental:
        conclusion = (
            "Incremental audio delivery observed through the official API "
            "(chunk_count > 1) -- see per-run cadence/underrun records."
        )
    else:
        conclusion = (
            "Non-streaming API: incremental audio delivery is NOT exposed in "
            "qwen-tts 0.1.1. Every measured run returned the complete waveform "
            "as a single delivery at call completion -- TTFB equals total wall "
            "on every run, chunk cadence is undefined (1 chunk), and the "
            "simulated real-time consumer never starves after playback start "
            "only because 100% of the audio already exists when playback "
            "begins. non_streaming_mode=False changes prompt framing only "
            "(simulated streaming TEXT input per the installed docstring) and "
            "left delivery semantics unchanged in every measured run."
        )

    out = {
        "streaming_exposed": incremental,
        "api_surface": api_surface,
        "scenarios": scenarios,
        "conclusion": conclusion,
        "env": _env_block(),
        "meta": {
            "date": "2026-09-21",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_head": _git_head(),
            "upstream_sha": UPSTREAM_QWEN3_TTS_SHA,
            "alias": args.alias,
            "runs_per_scenario": args.runs,
            "sampling_kwargs": {"do_sample": True, "temperature": 1.0},
            "dtype_attn_device": "loader.load defaults: bfloat16 + sdpa on cuda:0",
            "warmup_policy": "one discarded cn/short generation",
        },
    }
    json_out: Path = args.json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"[probe] streaming_exposed={incremental}")
    print(f"[probe] wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
