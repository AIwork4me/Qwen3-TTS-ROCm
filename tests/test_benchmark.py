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
