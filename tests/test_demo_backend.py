# tests/test_demo_backend.py
"""Task 15: headless demo backend (synthesis service + history store).

Covers ``qwen3_tts_rocm.demo.backend`` -- the browser-free layer the Gradio UI
(Task 17) thin-wraps:  official-surface calls only, every business decision
testable without a browser.  Everything here is CPU-hermetic: models are
:class:`~qwen3_tts_rocm.testing.FakeTTSModel` instances injected through the
service ``factory`` hook, the speech tokenizer through ``tokenizer_factory``
(the real one needs multi-GB weights); no ``gpu`` marker anywhere.

Ground truth for validation semantics = installed official demo
(``qwen_tts/cli/demo.py::run_instruct / run_voice_design / run_voice_clone /
save_prompt / load_prompt_and_gen``): same bilingual error strings, same
check ORDER (text -> [speaker|instruction|ref-audio -> ref-text]), text always
stripped before it reaches the model.

Step-1 gate::

    pytest tests/test_demo_backend.py -v          # whole file red before impl
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qwen3_tts_rocm import loader
from qwen3_tts_rocm.demo import (
    HistoryStore,
    SynthesisService,
    display_map,
    format_error,
)
from qwen3_tts_rocm.demo.backend import (
    DEFAULT_GEN_KWARGS,
    _coerce_pair,
    _normalize_gen_kwargs,
)
from qwen3_tts_rocm.testing import FakeTTSModel, make_tone

CHINESE = re.compile("[\u4e00-\u9fff]")  # diagnostics must carry Chinese text


# ---------------------------------------------------------------------------
# Test doubles injected through the service's factory hooks
# ---------------------------------------------------------------------------


class RecordingFactory:
    """``alias -> FakeTTSModel`` factory; remembers which aliases were built."""

    def __init__(self) -> None:
        self.alias_calls: list[str] = []
        self.fakes: dict[str, FakeTTSModel] = {}

    def __call__(self, alias: str) -> FakeTTSModel:
        """A FRESH fake per load() call -- like real weights reload."""
        self.alias_calls.append(str(alias))
        self.fakes[str(alias)] = FakeTTSModel()
        return self.fakes[str(alias)]


class FakeTokenizer:
    """Official-surface stand-in for ``Qwen3TTSTokenizer`` (getters + codec)."""

    def __init__(self) -> None:
        self.encode_inputs: list[tuple[np.ndarray, int]] = []

    def get_model_type(self) -> str:
        return "fake_codec"

    def get_input_sample_rate(self) -> int:
        return 24000

    def get_output_sample_rate(self) -> int:
        return 24000

    def get_encode_downsample_rate(self) -> int:
        return 1920

    def get_decode_upsample_rate(self) -> int:
        return 1920

    def encode(self, audios, sr=None, return_dict=True):
        arr = np.asarray(audios)
        self.encode_inputs.append((arr, int(sr)))
        holder = type("Encoded", (), {})()
        holder.audio_codes = [np.zeros((12, 4), dtype=np.int64)]
        return holder

    def decode(self, encoded):
        wav, sr = make_tone(seconds=0.25)
        return ([wav], sr)


def make_service() -> tuple[SynthesisService, RecordingFactory]:
    factory = RecordingFactory()
    return SynthesisService(factory=factory), factory


def make_tokenized_service() -> tuple[SynthesisService, RecordingFactory, list]:
    svc, factory = make_service()
    made: list[FakeTokenizer] = []

    def tok_factory() -> FakeTokenizer:
        made.append(FakeTokenizer())
        return made[-1]

    svc = SynthesisService(factory=factory, tokenizer_factory=tok_factory)
    return svc, factory, made


# ---------------------------------------------------------------------------
# Generation wrappers: internal tuples, kwargs guardrail, call recording
# ---------------------------------------------------------------------------


def test_custom_voice_returns_internal_tuple_and_records_call():
    svc, factory = make_service()
    sr, wav = svc.custom_voice("custom-voice", " Hello world ", "Auto", "Ryan",
                               instruct="angry")
    assert isinstance(sr, int) and sr == 24000
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32 and wav.ndim == 1
    model = factory.fakes["custom-voice"]
    call = model.calls[-1]
    assert call["method"] == "generate_custom_voice"
    assert call["text"] == ["Hello world"]       # stripped exactly like official
    assert call["instruct"] == ["angry"]
    assert call["gen_kwargs"] == {"max_new_tokens": 512}   # latency guardrail


def test_custom_voice_blank_instruct_becomes_none_like_official():
    svc, factory = make_service()
    svc.custom_voice("custom-voice", "Hi there", "Auto", "Vivian", instruct="   ")
    assert factory.fakes["custom-voice"].calls[-1]["instruct"] is None


def test_normalize_gen_kwargs_merges_defaults_under_user_overrides():
    assert DEFAULT_GEN_KWARGS == {"max_new_tokens": 512}   # 2048 once took 23 min!
    merged = _normalize_gen_kwargs({"max_new_tokens": 2048, "temperature": None})
    assert merged == {"max_new_tokens": 2048}              # user wins, None dropped
    assert _normalize_gen_kwargs(None) == {"max_new_tokens": 512}
    assert _normalize_gen_kwargs({}) == {"max_new_tokens": 512}


@pytest.mark.parametrize("bad_text", ["", " ", "\n\t "])
def test_empty_text_raises_before_model_touch(bad_text):
    svc, factory = make_service()
    with pytest.raises(ValueError, match="Text is required"):
        svc.custom_voice("custom-voice", bad_text, "Auto", "Ryan")
    with pytest.raises(ValueError, match="Target text is required"):
        svc.voice_clone("custom-voice", bad_text, "Auto", ref_audio=None)
    assert factory.alias_calls == []            # not even loaded


def test_custom_voice_requires_speaker_before_model_touch():
    svc, factory = make_service()
    with pytest.raises(ValueError, match="Speaker is required") as exc:
        svc.custom_voice("custom-voice", "Valid text", "Auto", "  ")
    assert CHINESE.search(str(exc.value))       # bilingual like official demo
    assert factory.alias_calls == []


def test_validation_order_text_is_checked_first():
    svc, factory = make_service()
    with pytest.raises(ValueError, match="Text is required"):   # not Speaker...
        svc.custom_voice("custom-voice", "", "", "")
    with pytest.raises(ValueError, match="Target text is required"):
        svc.voice_clone("custom-voice", " \t", "Auto", ref_audio=None,
                        xvec_only=True)
    assert factory.alias_calls == []


def test_voice_design_requires_instruction_and_records_call():
    svc, factory = make_service()
    with pytest.raises(ValueError, match="Voice design instruction is required"):
        svc.voice_design("voice-design", "A long time ago...", "Auto",
                         instruct=None)
    assert factory.alias_calls == []                     # validated pre-touch

    sr, wav = svc.voice_design("voice-design", "once upon a time ", "Auto",
                               instruct=" calm and slow ")
    model = factory.fakes["voice-design"]
    assert isinstance(sr, int) and isinstance(wav, np.ndarray)
    call = model.calls[-1]
    assert call["method"] == "generate_voice_design"
    assert call["text"] == ["once upon a time"]
    assert call["instruct"] == ["calm and slow"]
    assert call["gen_kwargs"]["max_new_tokens"] == 512


def test_voice_clone_validates_audio_then_ref_text_in_official_order():
    svc, factory = make_service()
    good_ref = (16000, np.zeros(1600, dtype=np.float32))

    with pytest.raises(ValueError, match="Reference audio is required"):
        svc.voice_clone("base", "Target words", "Auto", ref_audio=None,
                        ref_text="transcript")
    with pytest.raises(ValueError, match="Reference audio is required"):
        svc.voice_clone("base", "Target words", "Auto", ref_audio=(0, good_ref[1]))
    with pytest.raises(ValueError, match="Reference text is required when"):
        svc.voice_clone("base", "Target words", "Auto", ref_audio=good_ref,
                        ref_text=None)

    model = factory.fakes.get("base")     # nothing reached the model yet
    assert model is None or model.calls == []
    assert factory.alias_calls == []      # validation NEVER loads the model


def test_voice_clone_xvec_only_happy_path_normalizes_reference_pair():
    svc, factory = make_service()
    stereo = np.stack([np.zeros(800, np.float32), np.ones(800, np.float32)])
    sr, wav = svc.voice_clone("base", " target text ", "Auto",
                              ref_audio=(48000, stereo), ref_text=None,
                              xvec_only=True)
    assert isinstance(sr, int) and sr == 24000
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    model = factory.fakes["base"]
    call = model.calls[-1]
    assert call["method"] == "generate_voice_clone"
    at_wav, at_sr = call["ref_audio"]      # OFFICIAL pair order (wav, sr)
    assert at_sr == 48000 and at_wav.ndim == 1       # channels averaged
    assert call["text"] == ["target text"]
    assert call["x_vector_only_mode"] is True


def test_language_display_resolution_is_case_insensitive_to_lowercase_raws():
    svc, factory = make_service()
    svc.custom_voice("custom-voice", "你好世界", "CHINESE", "Ryan")
    call = factory.fakes["custom-voice"].calls[-1]
    assert call["language"] == ["chinese"]     # official raw value (lowercase)


def test_clone_prompt_from_ref_returns_items_and_reports_create_call():
    svc, factory = make_service()
    ref = (24000, np.zeros(2400, dtype=np.float32))

    items = svc.clone_prompt_from_ref("base", ref_audio=ref, ref_text=" hello ",
                                      xvec_only=False)
    assert isinstance(items, list) and len(items) == 1
    item = items[0]
    assert item.icl_mode is True and item.x_vector_only_mode is False
    model = factory.fakes["base"]
    assert model.calls[-1]["method"] == "create_voice_clone_prompt"

    with pytest.raises(ValueError, match="Reference audio is required"):
        svc.clone_prompt_from_ref("base", ref_audio=None, ref_text="t")
    with pytest.raises(ValueError, match="Reference text is required when"):
        svc.clone_prompt_from_ref("base", ref_audio=ref, ref_text=" ",
                                  xvec_only=False)


def test_voice_clone_with_prompt_happy_path_plus_empty_items_guard():
    svc, factory = make_service()
    prompt = factory_models_prompt(svc)
    assert factory.alias_calls == ["base"]

    sr, wav = svc.voice_clone_with_prompt("base", " speak this ", "Auto", prompt)
    assert isinstance(sr, int) and isinstance(wav, np.ndarray)
    model = factory.fakes["base"]
    call = model.calls[-1]
    assert call["voice_clone_prompt"] is prompt
    assert call["ref_audio"] is None and call["gen_kwargs"]["max_new_tokens"] == 512

    with pytest.raises(ValueError, match="Empty voice items"):
        svc.voice_clone_with_prompt("base", "speak", "Auto", [])
    assert len(model.calls) == 2               # guard fired without a new call


def factory_models_prompt(svc: SynthesisService):
    """Build prompt items from the service's own base-model fake."""
    ref = (24000, np.zeros(1200, dtype=np.float32))
    return svc.clone_prompt_from_ref("base", ref_audio=ref, ref_text="txt")


# ---------------------------------------------------------------------------
# Model cache: LRU slot size ONE, tracked current alias, unload_all
# ---------------------------------------------------------------------------


def test_lru_slot_unloads_previous_model_on_alias_switch(monkeypatch):
    svc, factory = make_service()
    unloaded: list[object] = []
    monkeypatch.setattr(loader, "unload", lambda m: unloaded.append(m))

    first = svc.get("custom-voice")
    svc.get("base")

    assert unloaded == [first]                 # LRU evicted exactly the old one
    assert svc.current_alias == "base"
    assert factory.alias_calls == ["custom-voice", "base"]


def test_get_same_alias_reuses_cached_model_without_reload():
    svc, factory = make_service()
    m1 = svc.get("custom-voice")
    m2 = svc.get("custom-voice")
    assert m1 is m2 and factory.alias_calls == ["custom-voice"]
    assert svc.current_alias == "custom-voice"


def test_unload_all_clears_cache_then_reload_recreates(monkeypatch):
    svc, factory = make_service()
    seen: list[object] = []
    monkeypatch.setattr(loader, "unload", lambda m: seen.append(m))
    model = svc.get("base")

    svc.unload_all()

    assert seen == [model]
    assert svc.current_alias is None
    fresh = svc.get("base")
    assert fresh is not model                  # truly reloaded post-teardown
    assert factory.alias_calls == ["base", "base"]


class CountingFactory:
    """``alias -> FakeTTSModel`` factory for concurrency hammers.

    Counts every build, tags each model with the alias it was built for (so
    torn cache state is detectable) and optionally delays the build to widen
    the race window the way a real multi-second weight load would.
    """

    def __init__(self, build_delay: float = 0.0) -> None:
        self.alias_calls: list[str] = []
        self._delay = build_delay

    def __call__(self, alias: str) -> FakeTTSModel:
        self.alias_calls.append(str(alias))
        if self._delay:
            time.sleep(self._delay)
        model = FakeTTSModel()
        model.built_for = str(alias)
        return model


def _run_threads(n: int, worker) -> list[Any]:
    """Start *n* barrier-synced workers, join them, return (thread errors)."""
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def target(rank: int) -> None:
        try:
            barrier.wait(timeout=10)
            worker(rank)
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(rank,)) for rank in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return errors


def test_get_hammered_from_eight_threads_loads_factory_exactly_once(monkeypatch):
    """The cache lock must collapse same-alias races to ONE factory load.

    Eight threads race ``get`` on the same alias behind a barrier; exactly one
    multi-second model load may happen and every caller must observe the very
    same model object (never a redundant second build).
    """
    monkeypatch.setattr(loader, "unload", lambda m: None)
    factory = CountingFactory(build_delay=0.02)   # widen the race window
    svc = SynthesisService(factory=factory)
    results: list[Any] = []

    errors = _run_threads(8, lambda _rank: results.append(svc.get("base")))

    assert errors == []
    assert factory.alias_calls == ["base"]        # ONE load, not eight
    assert len(results) == 8 and all(m is results[0] for m in results)
    assert svc.cached_aliases() == ("base",)
    assert svc.current_alias == "base"


def test_get_alternating_alias_hammer_keeps_single_slot_consistent(monkeypatch):
    """Concurrent alias switching keeps the size-ONE slot and its contents sane.

    After the hammer joins: at most one resident alias, and every model ever
    returned was built for the alias the caller asked for (no torn state).
    """
    monkeypatch.setattr(loader, "unload", lambda m: None)
    factory = CountingFactory()
    svc = SynthesisService(factory=factory)
    results: list[tuple[str, Any]] = []

    def worker(rank: int) -> None:
        for i in range(25):
            alias = ("base", "custom-voice")[(rank + i) % 2]
            results.append((alias, svc.get(alias)))

    errors = _run_threads(8, worker)

    assert errors == []
    assert len(svc.cached_aliases()) <= 1          # size-ONE slot invariant
    assert all(m.built_for == a for a, m in results)   # no torn state
    assert set(factory.alias_calls) <= {"base", "custom-voice"}
    assert svc.current_alias in (None, *svc.cached_aliases())


# ---------------------------------------------------------------------------
# Speech-tokenizer codec roundtrip (lazy, cached separately from the LRU)
# ---------------------------------------------------------------------------


def test_codec_roundtrip_returns_meta_and_builds_tokenizer_once():
    svc, _, made = make_tokenized_service()
    tone, tone_sr = make_tone(seconds=0.5)

    sr_out, audio, meta = svc.codec_roundtrip((tone_sr, tone))

    assert isinstance(sr_out, int) and sr_out == 24000
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
    assert set(meta) >= {
        "model_type", "input_sample_rate", "output_sample_rate",
        "encode_downsample", "decode_upsample", "codes_shape",
    }
    assert meta["model_type"] == "fake_codec"
    assert meta["codes_shape"] == (12, 4)
    assert len(made) == 1                      # lazily built ONCE, then cached
    seen_wav, seen_sr = made[0].encode_inputs[0]   # VALUE contract: official
    assert seen_sr == tone_sr                      # astype() copies, so compare
    assert seen_wav.dtype == np.float32            # samples, not object identity
    assert np.array_equal(seen_wav, tone)          # (0.8 tone passes through)

    svc.codec_roundtrip((tone_sr, tone))       # second call reuses the tokenizer
    assert len(made) == 1


def test_codec_roundtrip_rejects_malformed_input_pairs():
    svc, _, made = make_tokenized_service()
    for bad in (None, (24000,), ("not-int", np.zeros(4)), (-5, np.zeros(4)),
                (24000, None)):
        with pytest.raises(ValueError, match=r"\(sr, wav\)"):
            svc.codec_roundtrip(bad)
    assert made == []                          # never constructed a tokenizer


# ---------------------------------------------------------------------------
# Reference-audio normalization: _coerce_pair is a LINE-FAITHFUL port of the
# official qwen_tts.cli.demo._normalize_audio (int rescale -> float normalize
# -> clip -> stereo mean, in exactly that order).  Ground truth is the
# INSTALLED upstream function, imported straight from site-packages, and the
# comparison is bit-for-bit (array_equal) so future upstream diffs stay
# eyeball-able at the sample level.
# ---------------------------------------------------------------------------


def _official_normalize():
    """The installed official ``_normalize_audio`` (CPU-safe import)."""
    upstream = pytest.importorskip("qwen_tts.cli.demo")
    return upstream._normalize_audio


def test_coerce_pair_int16_full_scale_matches_official_bit_for_bit():
    official = _official_normalize()
    rng = np.random.default_rng(7)
    raw = rng.integers(-32768, 32768, size=1600, dtype=np.int16)
    raw[0] = -32768                            # guarantee true full scale

    wav, sr = _coerce_pair((16000, raw), "need (sr, wav)")

    assert sr == 16000
    assert wav.dtype == np.float32 and wav.ndim == 1
    assert np.array_equal(wav, official(raw))          # bit-for-bit, same /32768
    assert abs(float(np.abs(wav).max()) - 1.0) < 1e-6  # full scale -> peak 1.0


def test_coerce_pair_loud_float_is_peak_normalized_like_official():
    official = _official_normalize()
    loud = (np.sin(np.linspace(0.0, 100.0, 1000)) * 2.0).astype(np.float32)

    wav, _sr = _coerce_pair((24000, loud), "need (sr, wav)")

    assert np.array_equal(wav, official(loud))     # divide by (max + 1e-12)
    assert abs(float(np.abs(wav).max()) - 1.0) < 1e-6


def test_coerce_pair_already_normalized_float_passes_through():
    official = _official_normalize()
    quiet = np.linspace(-0.5, 0.5, 512, dtype=np.float32)

    wav, _sr = _coerce_pair((16000, quiet), "need (sr, wav)")

    assert np.array_equal(wav, official(quiet))
    assert np.array_equal(wav, quiet)              # within [-1, 1]: untouched


def test_coerce_pair_stereo_int16_normalizes_before_channel_mean():
    official = _official_normalize()
    rng = np.random.default_rng(11)
    stereo = rng.integers(-32768, 32768, size=(800, 2)).astype(np.int16)

    wav, _sr = _coerce_pair((16000, stereo), "need (sr, wav)")

    assert wav.ndim == 1 and wav.shape == (800,)
    # Official op ORDER (rescale -> clip -> mean LAST) is what makes this
    # bit-for-bit; mean-first implementations differ in float rounding.
    assert np.array_equal(wav, official(stereo))
    assert wav.dtype == np.float32


def test_bundled_reference_asset_present_for_the_ui_layer():
    """Carry-forward contract: assets/ref_en.wav ships inside the package."""
    from importlib.resources import files

    path = Path(str(files("qwen3_tts_rocm") / "demo" / "assets" / "ref_en.wav"))
    assert path.is_file() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# HistoryStore: FIFO cap 30, newest-first summaries, thread-safe ids
# ---------------------------------------------------------------------------


def test_history_store_caps_at_thirty_fifo_with_monotonic_ids():
    store = HistoryStore()
    wav = np.zeros(2400, dtype=np.float32)
    ids = [store.add(f"clip-{i}", 24000, wav) for i in range(31)]

    assert ids == sorted(ids) and len(set(ids)) == 31   # monotonic unique ids
    summaries = store.list()
    assert len(summaries) == 30               # oldest clip-0 was evicted FIFO
    assert {s["label"] for s in summaries} <= {f"clip-{i}" for i in range(1, 31)}
    assert all(s["label"] != "clip-0" for s in summaries)


def test_history_list_summaries_are_newest_first_with_duration():
    store = HistoryStore()
    short, sr = make_tone(seconds=0.5)
    i1 = store.add("short", sr, short)
    i2 = store.add("longer", sr, np.zeros(int(sr * 1.5), np.float32))

    summaries = store.list()
    assert [s["id"] for s in summaries] == [i2, i1]     # newest first
    by_id = {s["id"]: s for s in summaries}
    assert abs(by_id[i1]["duration_s"] - 0.5) < 0.01
    assert abs(by_id[i2]["duration_s"] - 1.5) < 0.01
    assert by_id[i1]["created_at"] > 0                  # numeric timestamp kept


def test_history_item_remove_roundtrip_unknown_ids_raise_bilingual():
    store = HistoryStore()
    wav, sr = make_tone(seconds=0.25)
    hid = store.add("take", sr, wav)

    back_sr, back_wav = store.item(hid)
    assert back_sr == sr and np.array_equal(back_wav, wav)

    store.remove(hid)
    with pytest.raises(ValueError, match="unknown history id") as exc:
        store.item(hid)
    assert CHINESE.search(str(exc.value))
    with pytest.raises(ValueError, match="unknown history id"):
        store.remove(hid)


# ---------------------------------------------------------------------------
# Choice mapping (official title-case convention) + error formatting
# ---------------------------------------------------------------------------


def test_display_map_follows_official_title_case_convention():
    display, mapping = display_map(["auto", "chinese", "ono_anna", "uncle_fu"])
    assert display == ["Auto", "Chinese", "Ono Anna", "Uncle Fu"]
    assert mapping["Ono Anna"] == "ono_anna"
    assert mapping["Uncle Fu"] == "uncle_fu"

    empty_display, empty_mapping = display_map([])
    assert empty_display == [] and empty_mapping == {}


def test_format_error_is_bilingual_status_string():
    message = format_error(ValueError("boom"))
    assert message.startswith("ValueError: boom")
    assert "请检查输入或查看终端日志" in message and "/ check input or see terminal log" in message
