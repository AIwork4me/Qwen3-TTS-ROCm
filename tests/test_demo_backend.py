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
from pathlib import Path

import numpy as np
import pytest

from qwen3_tts_rocm import loader
from qwen3_tts_rocm.demo import (
    HistoryStore,
    SynthesisService,
    display_map,
    format_error,
)
from qwen3_tts_rocm.demo.backend import DEFAULT_GEN_KWARGS, _normalize_gen_kwargs
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
    assert made[0].encode_inputs == [(tone, tone_sr)]

    svc.codec_roundtrip((tone_sr, tone))       # second call reuses the tokenizer
    assert len(made) == 1


def test_codec_roundtrip_rejects_malformed_input_pairs():
    svc, _, made = make_tokenized_service()
    for bad in (None, (24000,), ("not-int", np.zeros(4)), (-5, np.zeros(4)),
                (24000, None)):
        with pytest.raises(ValueError, match=r"\(sr, wav\)"):
            svc.codec_roundtrip(bad)
    assert made == []                          # never constructed a tokenizer


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
