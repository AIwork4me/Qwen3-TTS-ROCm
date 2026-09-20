# tests/test_benchmark.py
"""Task 18: unit pin for scripts/benchmark.py's PURE statistics helpers.

Scope note: the benchmark driver itself (model loads, generations, JSON emit)
is exercised live on the GPU and recorded under evidence/ -- no test doubles
are attempted here because the whole value of the tool is measuring the real
pipeline.  What IS trivially pinnable and worth protecting against silent
regression is the RTF arithmetic + markdown row shape that every reported
number flows through.  The script is loaded via importlib from its path
because ``scripts/`` is not an installed package; the module stays
stdlib-only at import time by design, so this import costs nothing.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("benchmark_script", _SCRIPT)
bench = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("benchmark_script", bench)
_spec.loader.exec_module(bench)


def test_rtf_of_basic_and_degenerate():
    """RTF = wall/audio; non-positive audio maps to inf (loud failure)."""
    assert math.isclose(bench.rtf_of(6.0, 3.0), 2.0)
    assert math.isclose(bench.rtf_of(1.0, 4.0), 0.25)
    assert math.isinf(bench.rtf_of(6.0, 0.0))
    assert math.isinf(bench.rtf_of(6.0, -1.0))


def test_cell_summary_median_min_max():
    """Median (not mean) drives the headline; min/max give the spread."""
    walls = [4.0, 6.0, 12.0]
    audios = [4.0, 4.0, 4.0]
    s = bench.cell_summary(walls, audios)
    assert s["runs"] == 3
    assert math.isclose(s["median_rtf"], 1.5)  # median of [1.0, 1.5, 3.0]
    assert math.isclose(s["min_rtf"], 1.0)
    assert math.isclose(s["max_rtf"], 3.0)
    assert math.isclose(s["median_wall_s"], 6.0)
    assert s["rtf_values"] == [1.0, 1.5, 3.0]


def test_md_row_shape_is_markdown_ready():
    """Rows must paste straight into docs/benchmarks.md as table cells."""
    s = bench.cell_summary([10.0, 20.0], [10.0, 10.0])  # rtfs: 1.0 / 2.0
    row = bench.md_row("custom-voice", "cn", "short", s)
    cells = [c.strip() for c in row.split("|")][1:-1]
    assert cells[:4] == ["custom-voice", "cn", "short", "2"]
    assert cells[4:] == ["1.50", "1.00", "2.00", "15.00"]


class _FakeCVModel:
    """Just enough official surface for build_call's custom-voice branch."""

    def get_supported_speakers(self):
        return ["aiden", "vivian"]


def test_build_call_widened_alias_set():
    """0.6B aliases resolve to their family's official entry point (Task 1)."""
    text, lang = "你好", "Chinese"
    method, kwargs = bench.build_call(_FakeCVModel(), "custom-voice-0.6b", text, lang)
    assert method == "generate_custom_voice"
    assert kwargs == {"text": text, "language": lang, "speaker": "aiden"}

    method, kwargs = bench.build_call(_FakeCVModel(), "custom-voice", text, lang)
    assert method == "generate_custom_voice"
    assert kwargs["speaker"] == "aiden"

    method, kwargs = bench.build_call(object(), "base-0.6b", "Hello", "English")
    assert method == "generate_voice_clone"
    assert kwargs["text"] == "Hello" and kwargs["language"] == "English"
    assert kwargs["ref_text"] == bench.BASE_REF_TEXT
    wav, sr = kwargs["ref_audio"]
    assert sr > 0 and wav.ndim == 1  # bundled clip read through the official tuple

    method, kwargs = bench.build_call(object(), "voice-design", text, lang)
    assert method == "generate_voice_design"
    assert kwargs["instruct"] == bench.VOICE_DESIGN_INSTRUCT


def test_build_call_unknown_alias_lists_all_five():
    """Typos fail loudly BEFORE any load, naming the full supported set."""
    try:
        bench.build_call(_FakeCVModel(), "custom-voice-06b", "x", "Chinese")
    except KeyError as exc:
        for name in ("custom-voice", "custom-voice-0.6b", "voice-design",
                     "base", "base-0.6b"):
            assert name in str(exc)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("unknown alias must raise KeyError")


def test_with_alias_metrics_rounding_and_none_peak():
    """Per-alias metrics attach rounded to every cell; None peak stays None."""
    cell = bench.with_alias_metrics({"alias": "base-0.6b"},
                                    load_seconds=12.3456,
                                    peak_alloc_gb=2.71828)
    assert cell["load_seconds"] == 12.346
    assert cell["peak_alloc_gb"] == 2.718

    cpu_cell = bench.with_alias_metrics({"alias": "base-0.6b"},
                                        load_seconds=5.0, peak_alloc_gb=None)
    assert cpu_cell["load_seconds"] == 5.0
    assert cpu_cell["peak_alloc_gb"] is None  # absence stays visible, not 0
