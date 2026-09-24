# tests/test_benchmark_v2.py
"""Unit tests for the v2 benchmark's pure statistics helper (v0.2.1 Task 4).

The script is loaded via importlib from its path exactly like the v1
benchmark tests; importing it is stdlib-only by design.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_v2.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_v2", _SCRIPT)
benchmark_v2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_v2)  # type: ignore[union-attr]


def test_cell_stats_v2_known_values():
    """median/min/max/mean/stdev of a fixed 5-sample match hand computation."""
    s = benchmark_v2.cell_stats_v2([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["median"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 5.0
    assert s["mean"] == 3.0
    assert s["stdev"] == 1.58  # sample stdev sqrt(2.5)=1.5811…, rounded to 2dp


def test_cell_stats_v2_p10_p90_within_min_max():
    """Inclusive-interpolated p10/p90 are order statistics inside [min, max]."""
    s = benchmark_v2.cell_stats_v2([0.5, 1.5, 2.5, 3.5, 10.0])
    assert s["min"] <= s["p10"] <= s["median"] <= s["p90"] <= s["max"]


def test_cell_stats_v2_single_sample_zero_stdev():
    """n=1 must not crash (statistics.quantiles needs >=2 points)."""
    s = benchmark_v2.cell_stats_v2([2.0])
    assert s["stdev"] == 0.0 and s["median"] == s["mean"] == s["p10"] == s["p90"] == 2.0


def test_cell_stats_v2_empty():
    assert benchmark_v2.cell_stats_v2([]) == {}
