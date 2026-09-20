# tests/test_streaming_probe.py
"""Step D: CPU unit tests for the pure helpers of scripts/streaming_probe.py.

The streaming probe itself is a GPU measurement (re-run by the verifier via
``.venv/bin/python scripts/streaming_probe.py``); everything testable without
a GPU lives in pure functions pinned here: RTF, chunk-cadence statistics,
the real-time consumer simulation (hand-traced timelines with engineered
starvation), and the per-run summarizer that composes them.

Import policy (mirrors tests/test_benchmark.py): the probe module imports
stdlib-only at module scope; torch/qwen_tts/loader stay lazy inside
functions, so this file needs no GPU and no model downloads.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    """Load a scripts/ module by path (scripts/ is not an installed package)."""
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(f"{name}_script", mod)
    spec.loader.exec_module(mod)
    return mod


probe = _load("streaming_probe")
bench = _load("benchmark")

SHORT_TEXT = probe.SHORT_TEXT
LONG_TEXT = probe.LONG_TEXT
cadence_stats = probe.cadence_stats
rtf_of = probe.rtf_of
simulate_realtime_consumer = probe.simulate_realtime_consumer
summarize_run = probe.summarize_run


class TestRtfOf:
    def test_basic_ratio(self):
        assert rtf_of(2.0, 4.0) == 0.5
        assert rtf_of(6.0, 3.0) == 2.0

    def test_nonpositive_audio_is_loud(self):
        assert math.isinf(rtf_of(1.0, 0.0))
        assert math.isinf(rtf_of(1.0, -2.0))


class TestCadenceStats:
    def test_single_inter_arrival_is_undefined_not_zero(self):
        stats = cadence_stats([0.5])
        assert stats == {"n_inter_arrivals": 1, "median": None, "p90": None,
                         "min": None, "max": None}

    def test_empty(self):
        assert cadence_stats([])["n_inter_arrivals"] == 0

    def test_distribution(self):
        stats = cadence_stats([1.0, 2.0, 3.0, 10.0])
        assert stats["n_inter_arrivals"] == 4
        assert stats["median"] == pytest.approx(2.5)
        # inclusive linear interpolation == numpy default: 3 + 0.7*(10-3)
        assert stats["p90"] == pytest.approx(7.9)
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["p90"] == pytest.approx(
            statistics.quantiles([1.0, 2.0, 3.0, 10.0], n=10, method="inclusive")[8]
        )


class TestSimulateRealtimeConsumer:
    SR = 24_000

    def test_single_chunk_never_starves(self):
        """Whole audio present at playback start -> no starvation possible."""
        out = simulate_realtime_consumer([1.0], [24_000], self.SR)
        assert out["playback_start_s"] == 1.0
        assert out["ideal_playback_end_s"] == 2.0
        assert out["underrun_events"] == 0
        assert out["starved_s"] == 0.0

    def test_gap_after_first_chunk_one_continuous_event(self):
        """0.5 s of audio, then a 1.5 s wait: ONE event, 1.5 s starved.

        Hand trace: dry at t=0.5, refilled at t=2.0; ideal end 1.5 s, real
        end 3.0 s -- starvation is exactly the 0.5..2.0 window.
        """
        out = simulate_realtime_consumer([0.0, 2.0], [12_000, 24_000], self.SR)
        assert out["underrun_events"] == 1
        assert out["starved_s"] == pytest.approx(1.5)

    def test_two_separate_events(self):
        """Refill-then-dry-again: two distinct starvation windows.

        Hand trace: 0.5 s chunk drains at 0.5; refill at 0.9 buys 0.1 s;
        dry again 1.0..3.0; final refill at 3.0. Starved = 0.4 + 2.0 = 2.4 s
        across 2 events.
        """
        out = simulate_realtime_consumer(
            [0.0, 0.9, 3.0], [12_000, 12_000, 24_000], self.SR)
        assert out["underrun_events"] == 2
        assert out["starved_s"] == pytest.approx(2.4)

    def test_ideal_paced_chunks_no_starvation(self):
        """Chunks sized/arrived to drain exactly at each refill: no events."""
        out = simulate_realtime_consumer(
            [0.0, 0.2, 0.4], [4_800, 4_800, 4_800], self.SR)
        assert out["underrun_events"] == 0
        assert out["starved_s"] == 0.0

    def test_late_single_chunk_starves_while_waiting(self):
        """0.2 s chunk then a 4.8 s wait for the rest: one event, 4.8 s starved."""
        out = simulate_realtime_consumer([0.0, 5.0], [4_800, 24_000], self.SR)
        assert out["underrun_events"] == 1
        assert out["starved_s"] == pytest.approx(4.8)

    def test_guards(self):
        with pytest.raises(ValueError):
            simulate_realtime_consumer([], [], self.SR)
        with pytest.raises(ValueError):
            simulate_realtime_consumer([0.0], [1, 2], self.SR)
        with pytest.raises(ValueError):
            simulate_realtime_consumer([0.0], [1], 0)


class TestSummarizeRun:
    def test_single_delivery_record(self):
        """The measured-shape run: one arrival at wall time, all samples."""
        rec = summarize_run([4.2], [48_000], 24_000, wall_s=4.2)
        assert rec["ttfb_s"] == 4.2
        assert rec["chunk_count"] == 1
        assert rec["cadence_s"]["n_inter_arrivals"] == 0
        assert rec["cadence_s"]["median"] is None
        assert rec["audio_s"] == 2.0
        assert rec["rtf"] == pytest.approx(2.1)
        assert rec["underrun_events"] == 0
        assert rec["starved_s"] == 0.0
        assert rec["total_samples"] == 48_000

    def test_multi_chunk_record(self):
        rec = summarize_run([1.0, 2.0, 3.5], [24_000, 24_000, 24_000], 24_000,
                            wall_s=3.5)
        assert rec["chunk_count"] == 3
        assert rec["cadence_s"]["n_inter_arrivals"] == 2
        assert rec["cadence_s"]["median"] == pytest.approx(1.25)
        assert rec["audio_s"] == 3.0
        assert rec["rtf"] == pytest.approx(3.5 / 3.0, rel=1e-3)


class TestPayloads:
    def test_texts_match_brief_lengths(self):
        assert len(SHORT_TEXT) == 17
        assert len(LONG_TEXT) >= 200

    def test_short_text_is_the_benchmark_cn_short_cell(self):
        """Comparability: identical payload to scripts/benchmark.py cn/short."""
        assert SHORT_TEXT == bench.TEXTS["cn"]["short"]
