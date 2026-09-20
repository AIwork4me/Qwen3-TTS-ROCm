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
        self.fakes: dict[str, FakeTTSModel] = {}
        self._delay = build_delay

    def __call__(self, alias: str) -> FakeTTSModel:
        self.alias_calls.append(str(alias))
        if self._delay:
            time.sleep(self._delay)
        model = FakeTTSModel()
        model.built_for = str(alias)
        self.fakes[str(alias)] = model
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


def test_get_teardown_failure_never_leaks_the_load_ticket(monkeypatch):
    """A failure AFTER the ticket is installed must not poison ``_loading``.

    Regression: the ticket-time pre-load eviction (``loader.unload`` raising
    during teardown) used to abort ``get`` with the ticket still installed --
    every later ``get(alias)`` then waited forever on an event nobody would
    ever set.  Both waits use join timeouts so a reopened leak window fails
    fast instead of hanging the suite.
    """
    factory = CountingFactory()
    svc = SynthesisService(factory=factory)
    resident = svc.get("base")                 # something to evict on switch

    real_unload = loader.unload

    def exploding_unload(model):
        if model is resident:                  # evict-on-switch blows up ONCE
            raise RuntimeError("unload exploded (ticket-time teardown)")
        real_unload(model)

    monkeypatch.setattr(loader, "unload", exploding_unload)

    errors: list[BaseException] = []

    def failing_worker() -> None:
        try:
            svc.get("custom-voice")            # ticket -> teardown -> boom
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            errors.append(exc)

    t1 = threading.Thread(target=failing_worker, daemon=True)
    t1.start()
    t1.join(timeout=10)
    assert not t1.is_alive()
    assert len(errors) == 1 and "ticket-time teardown" in str(errors[0])

    # The ticket MUST be released: a healthy reload completes (no hang) and no
    # stale reservation is left in _loading.
    monkeypatch.setattr(loader, "unload", lambda m: None)
    ok: list[Any] = []
    t2 = threading.Thread(target=lambda: ok.append(svc.get("custom-voice")),
                          daemon=True)
    t2.start()
    t2.join(timeout=10)
    assert not t2.is_alive()                   # would hang forever if leaked
    assert ok and ok[0].built_for == "custom-voice"
    assert svc.cached_aliases() == ("custom-voice",)
    assert svc._loading == {}                  # white-box: no poisoned ticket


def test_get_waiter_wakes_when_winner_install_teardown_fails(monkeypatch):
    """A failure in the winner-install teardown must still WAKE ticket waiters.

    Regression (re-review hammer: waiters stranded 51/60 rounds): the success
    body used to pop the ticket BEFORE the winner-install teardown, so a
    teardown failure there skipped the wake-up and left waiters parked on the
    event forever.  The finally must be the single cleanup authority, so any
    post-install failure wakes waiters (they re-check and proceed).

    Deterministic handshake: the owner's factory blocks until the test has
    PROVEN the waiter parked (the ticket event's ``wait`` is traced on the
    instance) -- no sleep-based race.  join timeouts fail fast, never hang.
    """
    inner = CountingFactory()
    owner_in_factory = threading.Event()
    release_owner = threading.Event()
    waiter_parked = threading.Event()

    def gated(alias: str) -> FakeTTSModel:
        """Hold the owner inside the factory (ticket held) until released."""
        if alias == "custom-voice" and not release_owner.is_set():
            owner_in_factory.set()
            assert release_owner.wait(timeout=10)
        return inner(alias)

    svc = SynthesisService(factory=gated)      # starts EMPTY: ticket-time
    real_unload = loader.unload                # teardown has no victim at all
    exploded = False

    def exploding_unload(model):
        nonlocal exploded
        if model is inner.fakes.get("base") and not exploded:
            exploded = True                    # ONLY the winner-install pass
            raise RuntimeError("unload exploded (winner-install teardown)")
        real_unload(model)                     # best-effort: never raises else

    monkeypatch.setattr(loader, "unload", exploding_unload)

    owner_errors: list[BaseException] = []

    def owner() -> None:
        try:
            svc.get("custom-voice")            # ticket -> load -> teardown BOOM
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            owner_errors.append(exc)

    to = threading.Thread(target=owner, daemon=True)
    to.start()
    assert owner_in_factory.wait(timeout=10)   # ticket is installed and held

    ticket = svc._loading["custom-voice"]      # white-box: trace the parking
    real_wait = ticket.wait

    def traced_wait(timeout=None):
        waiter_parked.set()                    # from here, parking is proven
        return real_wait(timeout)

    ticket.wait = traced_wait

    def waiter() -> None:
        svc.get("custom-voice")

    tw = threading.Thread(target=waiter, daemon=True)
    tw.start()
    assert waiter_parked.wait(timeout=10)      # parked on the ticket event
    time.sleep(0.05)                           # let it block inside real_wait

    # A third thread makes a DIFFERENT alias resident while the ticket is held:
    # this is the victim the owner's winner-install teardown must unload (the
    # only way that teardown has work to do in a deterministic single run).
    tb = threading.Thread(target=lambda: svc.get("base"), daemon=True)
    tb.start()
    tb.join(timeout=10)
    assert not tb.is_alive() and "base" in inner.fakes

    release_owner.set()                        # factory returns -> teardown BOOM
    to.join(timeout=10)
    assert not to.is_alive()
    assert exploded
    assert len(owner_errors) == 1 and "winner-install teardown" in str(owner_errors[0])

    tw.join(timeout=10)
    assert not tw.is_alive()                   # would hang forever if unwoken
    back = svc.get("custom-voice")             # healthy: slot usable afterwards
    assert back.built_for == "custom-voice"
    assert svc.cached_aliases() == ("custom-voice",)
    assert svc._loading == {}                  # white-box: no poisoned ticket


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


# ---------------------------------------------------------------------------
# UX-fix B-4: per-tab automatic model routing.  The sidebar switcher is global
# while each tab calls a fixed backend capability, so the service owns an
# alias->kind table plus a per-kind default and resolves BEFORE every tab
# generation (audited failure: sidebar=VoiceDesign + Tab (2) Preset Speakers
# used to raise the official "does not support generate_custom_voice" error).
# ---------------------------------------------------------------------------


def test_resolve_alias_mapping_constants_cover_every_generatable_alias():
    """B-4 ground truth tables: alias -> kind, and one default alias per kind."""
    from qwen3_tts_rocm.demo.backend import DEFAULT_FOR_KIND, KIND_OF_ALIAS

    assert KIND_OF_ALIAS == {
        "custom-voice": "custom_voice",
        "custom-voice-0.6b": "custom_voice",
        "voice-design": "voice_design",
        "base": "base",
        "base-0.6b": "base",
    }
    assert DEFAULT_FOR_KIND == {
        "custom_voice": "custom-voice",
        "voice_design": "voice-design",
        "base": "base",
    }


def test_resolve_alias_same_kind_passthrough_keeps_the_radio_selection():
    """A matching pick (incl. the 0.6B size) is kept verbatim, no switch."""
    svc, _factory = make_service()
    assert svc.resolve_alias("custom_voice", "custom-voice") == ("custom-voice", False)
    assert svc.resolve_alias("custom_voice", "custom-voice-0.6b") == ("custom-voice-0.6b", False)
    assert svc.resolve_alias("voice_design", "voice-design") == ("voice-design", False)


def test_resolve_alias_cross_kind_switches_to_the_kind_default():
    """The audited mismatch: tab kind wins over the global sidebar pick."""
    svc, _factory = make_service()
    assert svc.resolve_alias("custom_voice", "voice-design") == ("custom-voice", True)
    assert svc.resolve_alias("voice_design", "custom-voice") == ("voice-design", True)
    assert svc.resolve_alias("voice_design", "custom-voice-0.6b") == ("voice-design", True)


def test_resolve_alias_falsy_or_unknown_alias_falls_back_to_default():
    """None/blank/stale-page aliases (server restarted, browser kept old
    state) fall back to the tab's default with switched=True -- never an
    error, never a load of a nonexistent alias."""
    svc, _factory = make_service()
    assert svc.resolve_alias("custom_voice", None) == ("custom-voice", True)
    assert svc.resolve_alias("voice_design", "") == ("voice-design", True)
    assert svc.resolve_alias("base", "   ") == ("base", True)
    assert svc.resolve_alias("custom_voice", "ghost-alias") == ("custom-voice", True)


@pytest.mark.parametrize("radio", ["base", "base-0.6b", "custom-voice", "voice-design", None])
def test_resolve_alias_base_family_routes_both_sizes(radio):
    """Clone tabs need the base capability: both sizes pass through, the
    rest (incl. falsy) route to the default base alias."""
    svc, _factory = make_service()
    alias, switched = svc.resolve_alias("base", radio)
    if radio in ("base", "base-0.6b"):
        assert (alias, switched) == (radio, False)
    else:
        assert (alias, switched) == ("base", True)


# ---------------------------------------------------------------------------
# Voice Studio (⑥ 音色工坊): design -> preview -> save -> reuse over the
# official model split.  The service composes ONLY official calls through
# qwen3_tts_rocm.voice_workflow: generate_voice_design on the tab's
# VoiceDesign alias, then create_voice_clone_prompt / generate_voice_clone on
# Base (the official wrapper hard-gates each on tts_model_type).  The lazy
# prompt-model factory keeps the size-1 LRU honest: VoiceDesign is evicted
# exactly when the prompt phase begins.
# ---------------------------------------------------------------------------


def make_studio_service(tmp_path) -> tuple[SynthesisService, RecordingFactory]:
    """Fake-backed service with the voices dir pinned to *tmp_path*."""
    factory = RecordingFactory()
    return SynthesisService(factory=factory, voices_dir=tmp_path), factory


def test_voice_studio_design_composes_official_split_with_transcript_rule(tmp_path):
    svc, factory = make_studio_service(tmp_path)

    voice_id, sr, wav, timings = svc.voice_studio_design(
        "voice-design", " hello studio ", "Auto", "bright female",
    )

    assert isinstance(voice_id, str) and voice_id.startswith("voice-")
    assert isinstance(sr, int) and sr == 24000
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32 and wav.ndim == 1
    # Official model split: preview on VoiceDesign, prompt items on Base.
    assert factory.alias_calls == ["voice-design", "base"]

    design_call = factory.fakes["voice-design"].calls[-1]
    assert design_call["method"] == "generate_voice_design"
    assert design_call["text"] == ["hello studio"]        # stripped exactly once
    assert design_call["instruct"] == ["bright female"]
    assert design_call["gen_kwargs"] == {"max_new_tokens": 512}

    prompt_call = factory.fakes["base"].calls[-1]
    assert prompt_call["method"] == "create_voice_clone_prompt"
    at_wav, at_sr = prompt_call["ref_audio"]              # OFFICIAL (wav, sr)
    assert at_sr == sr and np.asarray(at_wav).size == wav.size
    # Transcript rule: ref_text IS the design text (never re-typed).
    assert prompt_call["ref_text"] == "hello studio"

    # Three-phase evidence discipline: design and prompt timed SEPARATELY.
    assert timings["design_s"] > 0.0 and timings["prompt_s"] > 0.0


def test_voice_studio_design_validates_before_any_model_touch(tmp_path):
    svc, factory = make_studio_service(tmp_path)
    with pytest.raises(ValueError, match="Text is required"):
        svc.voice_studio_design("voice-design", "  ", "Auto", "bright female")
    with pytest.raises(ValueError, match="Voice design instruction is required"):
        svc.voice_studio_design("voice-design", "words", "Auto", "   ")
    assert factory.alias_calls == []                      # not even loaded


def test_voice_studio_design_routes_off_mismatched_radio(tmp_path):
    svc, factory = make_studio_service(tmp_path)
    _vid, _sr, _wav, _timings = svc.voice_studio_design(
        "base", "hello studio", "Auto", "bright female",
    )
    assert factory.alias_calls == ["voice-design", "base"]  # voice_design kind wins


def test_voice_studio_save_persists_official_payload_and_lists(tmp_path):
    import torch

    svc, _factory = make_studio_service(tmp_path)
    voice_id, _sr, _wav, _t = svc.voice_studio_design(
        "voice-design", "hello studio", "Auto", "bright female",
    )
    assert svc.voice_studio_list() == []                  # nothing saved yet

    path = svc.voice_studio_save(voice_id, " my voice ")

    assert str(path).endswith("my voice.pt") and (tmp_path / "my voice.pt").is_file()
    assert svc.voice_studio_list() == ["my voice"]

    # The .pt payload keeps the OFFICIAL item schema plus the meta sidecar.
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    assert set(payload["items"][0].keys()) == {
        "ref_code", "ref_spk_embedding", "x_vector_only_mode",
        "icl_mode", "ref_text",
    }
    assert payload["items"][0]["ref_text"] == "hello studio"
    # Language is the RESOLVED official raw identifier ("Auto" -> "auto"),
    # exactly like every other service generation method.
    assert payload["voice_meta"] == {
        "description": "bright female", "language": "auto",
        "ref_text": "hello studio",
    }


def test_voice_studio_save_rejects_bad_names_and_unknown_ids(tmp_path):
    svc, _factory = make_studio_service(tmp_path)
    voice_id, *_ = svc.voice_studio_design(
        "voice-design", "hello studio", "Auto", "bright female",
    )

    with pytest.raises(ValueError, match="Unknown studio voice id") as exc:
        svc.voice_studio_save("voice-999", "whatever")
    assert CHINESE.search(str(exc.value))
    with pytest.raises(ValueError, match="plain file name"):
        svc.voice_studio_save(voice_id, "a/b")
    with pytest.raises(ValueError, match="plain file name"):
        svc.voice_studio_save(voice_id, "..")
    with pytest.raises(ValueError, match="Voice name is required"):
        svc.voice_studio_save(voice_id, "   ")
    assert svc.voice_studio_list() == []                  # nothing was written


def test_voice_studio_generate_loads_saved_voice_and_reuses_on_base(tmp_path):
    svc, factory = make_studio_service(tmp_path)
    voice_id, *_ = svc.voice_studio_design(
        "voice-design", "hello studio", "Auto", "bright female",
    )
    svc.voice_studio_save(voice_id, "my_voice")

    sr, wav, generate_s = svc.voice_studio_generate(
        "base", "my_voice", " new sentence ", "Auto",
    )
    assert isinstance(sr, int) and sr == 24000
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert isinstance(generate_s, float) and generate_s > 0.0  # reuse timed apart

    model = factory.fakes["base"]
    call = model.calls[-1]
    assert call["method"] == "generate_voice_clone"
    assert call["text"] == ["new sentence"]
    assert call["voice_clone_prompt"] is not None          # reloaded items
    assert call["gen_kwargs"] == {"max_new_tokens": 512}
    # The designed voice generalises: a SECOND, different sentence too.
    _sr2, wav2, gen2 = svc.voice_studio_generate(
        "base", "my_voice", "another different sentence", "Auto",
    )
    assert gen2 > 0.0 and wav2.size > 0

    with pytest.raises(ValueError, match="not found") as exc:
        svc.voice_studio_generate("base", "ghost", "words", "Auto")
    assert CHINESE.search(str(exc.value))


def test_voice_studio_generate_routes_off_mismatched_radio(tmp_path):
    svc, factory = make_studio_service(tmp_path)
    voice_id, *_ = svc.voice_studio_design(
        "voice-design", "hello studio", "Auto", "bright female",
    )
    svc.voice_studio_save(voice_id, "my_voice")

    # A mismatched (voice-design) sidebar pick still routes the reuse phase
    # to the base capability -- and the LRU serves the ALREADY-resident base
    # model without a reload (no new factory call).
    n_factory_before = len(factory.alias_calls)
    sr, wav, _gen_s = svc.voice_studio_generate(
        "voice-design", "my_voice", "words", "Auto",
    )
    assert sr == 24000 and wav.dtype == np.float32
    assert len(factory.alias_calls) == n_factory_before     # cached, not reloaded
    assert svc.current_alias == "base"                      # base kind won
    assert factory.fakes["base"].calls[-1]["method"] == "generate_voice_clone"


def test_voice_studio_store_caps_fifo_and_voices_dir_defaults(tmp_path):
    svc, _factory = make_studio_service(tmp_path)
    ids = [
        svc.voice_studio_design("voice-design", f"text {i}", "Auto", "d")[0]
        for i in range(svc.studio_cap + 1)
    ]
    with pytest.raises(ValueError, match="Unknown studio voice id"):
        svc.voice_studio_save(ids[0], "evicted")          # oldest was evicted
    svc.voice_studio_save(ids[-1], "kept")                # newest still there
    assert svc.voice_studio_list() == ["kept"]

    # Lazy default: a service built without voices_dir resolves to
    # <cwd>/voices on first touch; LISTING is side-effect free (no mkdir --
    # only an actual save ever creates the directory, on demand).
    bare = SynthesisService(factory=lambda a: FakeTTSModel())
    assert bare.voices_dir == Path.cwd() / "voices"
    before = set(Path.cwd().glob("voices"))
    assert isinstance(bare.voice_studio_list(), list)
    assert set(Path.cwd().glob("voices")) == before
