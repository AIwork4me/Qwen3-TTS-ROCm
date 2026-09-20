# tests/test_demo_ui.py
"""Task 16: enhanced five-tab bilingual Gradio application (demo/ui.py + cli_demo).

Hermetic UI-layer coverage, zero GPU / zero weights / browser-free:

* **UI shape**        -- ``build_ui(service, header_info)`` constructs a
  ``gr.Blocks`` carrying the five tabs (Reference clone / Speaker preset /
  Voice Design / Codec / History), the official disclaimer
  footer verbatim and the single-GPU serialization notice (排队串行生成说明).
* **Callback smoke**  -- ``build_callbacks(service)`` exposes module-level,
  Blocks-free callables that the tests invoke end-to-end against a
  :class:`~qwen3_tts_rocm.testing.FakeTTSModel`-backed service (the pattern
  fixed by the task brief: business logic lives behind thin UI wrappers).
* **Save/Load voice** -- ``SynthesisService.load_voice_file`` reconstructs
  prompt items EXACTLY like the official ``qwen_tts.cli.demo.load_prompt_and_gen``
  payload schema, round-tripped through ``torch.save({"items": [...]})``.

Ground truth = installed ``.venv/.../qwen_tts/cli/demo.py`` (theme/css/header/
tabs/_wav_to_gradio_audio/status string/disclaimer).  Skips politely when
gradio itself is unavailable so a bare CI runner can still run the rest.

Step-1 gate::

    pytest tests/test_demo_ui.py -v          # whole file red before impl
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gradio")


from qwen3_tts_rocm.demo import SynthesisService
from qwen3_tts_rocm.testing import FakeTTSModel, make_tone

CHINESE = re.compile("[\u4e00-\u9fff]")


# ---------------------------------------------------------------------------
# Test doubles injected through the service's factory hooks (same style as
# tests/test_demo_backend.py)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Official-surface stand-in for ``Qwen3TTSTokenizer``."""

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
        holder = type("Encoded", (), {})()
        holder.audio_codes = [np.zeros((12, 4), dtype=np.int64)]
        return holder

    def decode(self, encoded):
        wav, sr = make_tone(seconds=0.25)
        return ([wav], sr)


def make_service() -> SynthesisService:
    """A fresh fake-backed service (fresh fake per alias, like real reloads)."""
    return SynthesisService(
        factory=lambda alias: FakeTTSModel(),
        tokenizer_factory=FakeTokenizer,
    )


class RecordingFactory:
    """``alias -> FakeTTSModel`` factory; remembers which aliases were built."""

    def __init__(self) -> None:
        self.alias_calls: list[str] = []
        self.fakes: dict[str, FakeTTSModel] = {}

    def __call__(self, alias: str) -> FakeTTSModel:
        self.alias_calls.append(str(alias))
        self.fakes[str(alias)] = FakeTTSModel()
        return self.fakes[str(alias)]


def make_recording_callbacks() -> tuple[dict, RecordingFactory]:
    """Callbacks wired to a fake-backed service whose factory loads are logged.

    B-4 routing evidence: the tests assert WHICH alias the tab's generate
    callback actually sent to the model factory, not just the status text.
    """
    from qwen3_tts_rocm.demo.ui import build_callbacks

    factory = RecordingFactory()
    service = SynthesisService(factory=factory, tokenizer_factory=FakeTokenizer)
    return build_callbacks(service), factory


def tone_pair(seconds: float = 0.5) -> tuple[int, np.ndarray]:
    wav, sr = make_tone(seconds=seconds)
    return sr, wav


@pytest.fixture()
def cbs():
    """Callback dict built against ONE fake-backed service, Blocks-free."""
    from qwen3_tts_rocm.demo.ui import build_callbacks

    return build_callbacks(make_service())


@pytest.fixture()
def app_blocks():
    """One headlessly constructed Blocks app for shape assertions."""
    from qwen3_tts_rocm.demo.ui import build_ui

    return build_ui(make_service(), {"alias": "custom-voice"})


# ---------------------------------------------------------------------------
# UI shape: headless Blocks construction (task-brief Step 1 sketch)
# ---------------------------------------------------------------------------


def test_build_ui_constructs_five_tabs(app_blocks):
    ids = {c.label for c in app_blocks.blocks.values() if hasattr(c, "label")}
    assert any(l and "Reference" in l for l in ids)  # clone tab
    assert any(l and "Speaker" in l for l in ids)  # preset tab
    assert any(l and "Voice Design" in l for l in ids)  # design tab
    assert any(l and "Codec" in l for l in ids)  # codec tab
    assert any(l and "History" in l for l in ids)  # history tab


def test_build_ui_contains_history_disclaimer_and_queue_note():
    from qwen3_tts_rocm.demo.ui import CONCURRENCY_NOTE, DISCLAIMER, build_ui

    app = build_ui(make_service(), {"alias": "custom-voice"})
    texts = []
    for comp in app.blocks.values():
        for attr in ("value", "label"):
            v = getattr(comp, attr, None)
            if isinstance(v, str):
                texts.append(v)
    blob = "\n".join(texts)
    assert any("History" in t or "历史" in t for t in texts)  # history area
    assert DISCLAIMER.strip().splitlines()[1][:40] in blob  # EN disclaimer body
    assert "免责声明" in blob  # zh disclaimer heading
    assert "single-GPU" in CONCURRENCY_NOTE and "排队串行生成" in CONCURRENCY_NOTE
    assert "排队串行生成" in blob  # queue note shown in UI
    assert app.title and "Qwen3 TTS" in app.title  # curl-smoke anchor


def test_wav_to_gradio_audio_matches_official_sr_first_shape():
    from qwen3_tts_rocm.demo.ui import _wav_to_gradio_audio

    wav = np.zeros(8, dtype=np.float64)
    sr, out = _wav_to_gradio_audio(wav, 24000)
    assert sr == 24000 and out.dtype == np.float32 and out.shape == (8,)


# ---------------------------------------------------------------------------
# SynthesisService.load_voice_file: official load_prompt_and_gen reconstruction
# ---------------------------------------------------------------------------


def test_load_voice_file_roundtrip_official_payload_schema(tmp_path: Path):
    import torch

    svc = make_service()
    items = svc.clone_prompt_from_ref("base", ref_audio=tone_pair(0.4), ref_text=" reference words ")
    payload = {
        "items": [
            {
                "ref_code": it.ref_code.tolist(),
                "ref_spk_embedding": it.ref_spk_embedding.tolist(),
                "x_vector_only_mode": it.x_vector_only_mode,
                "icl_mode": it.icl_mode,
                "ref_text": None,
            }
            for it in items
        ]
    }
    path = tmp_path / "voice.pt"
    torch.save(payload, str(path))

    rebuilt = svc.load_voice_file(str(path))
    assert isinstance(rebuilt, list) and len(rebuilt) == len(items)
    a, b = items[0], rebuilt[0]
    assert bool(b.x_vector_only_mode) == bool(a.x_vector_only_mode)
    assert bool(b.icl_mode) == bool(a.icl_mode)
    assert b.ref_spk_embedding.shape == tuple(np.asarray(a.ref_spk_embedding).shape)
    assert b.ref_code.shape == tuple(np.asarray(a.ref_code).shape)


def test_load_voice_file_rejects_bad_payloads_bilingually(tmp_path: Path):
    import torch

    svc = make_service()

    bad_empty = tmp_path / "empty_items.pt"
    torch.save({"items": []}, str(bad_empty))
    with pytest.raises(ValueError, match="Empty voice items") as exc:
        svc.load_voice_file(str(bad_empty))
    assert CHINESE.search(str(exc.value))

    bad_no_items = tmp_path / "missing_key.pt"
    torch.save({"something_else": 1}, str(bad_no_items))
    with pytest.raises(ValueError, match="Invalid file format"):
        svc.load_voice_file(str(bad_no_items))

    bad_no_spk = tmp_path / "missing_spk.pt"
    torch.save({"items": [{"ref_code": [[1, 2]]}]}, str(bad_no_spk))
    with pytest.raises(ValueError, match="Missing ref_spk_embedding"):
        svc.load_voice_file(str(bad_no_spk))


def test_save_then_load_then_generate_end_to_end(tmp_path: Path):
    from qwen3_tts_rocm.demo.ui import build_callbacks

    svc = make_service()
    cb = build_callbacks(svc)
    saved_file, status = cb["save_voice"]("base", tone_pair(0.4), " hello reference ", xvec_only=False)
    assert saved_file and Path(str(saved_file)).exists()
    assert status.startswith("Finished.")

    audio, status2 = cb["load_voice_gen"](
        "base", saved_file, " new target words ", "Auto", {"max_new_tokens": 512}
    )
    assert status2.startswith("Finished.") and audio is not None
    sr_out, wav_out = audio
    assert isinstance(sr_out, int) and wav_out.dtype == np.float32
    calls = svc.get("base").calls
    assert calls[-1]["method"] == "generate_voice_clone"
    assert calls[-1]["voice_clone_prompt"] is not None  # reconstructed items


# ---------------------------------------------------------------------------
# Generation callbacks (thin wrappers over the Task 15 service)
# ---------------------------------------------------------------------------


def test_callback_custom_voice_happy_path_and_status_string(cbs):
    audio, status = cbs["custom_voice"]("custom-voice", " Hello world ", "Chinese", "Ryan", "", {})
    assert status == "Finished. (生成完成)"
    sr, wav = audio
    assert isinstance(sr, int) and sr == 24000 and wav.ndim == 1


def test_callback_custom_voice_blank_text_returns_bilingual_status_not_raise(cbs):
    audio, status = cbs["custom_voice"]("custom-voice", "  ", "Auto", "", "", {})
    assert audio is None and "Text is required" in status
    assert CHINESE.search(status)


def test_callback_voice_design_happy_path(cbs):
    audio, status = cbs["voice_design"]("voice-design", " once upon a time ", "Auto", " calm whisper ", {})
    assert status == "Finished. (生成完成)"
    assert audio is not None


def test_callback_voice_design_missing_instruction_reports_error(cbs):
    audio, status = cbs["voice_design"]("voice-design", "text", "Auto", "  ", {})
    assert audio is None and "Voice design instruction is required" in status


def test_callback_voice_clone_xvec_only_happy_path(cbs):
    """B-4 behavior change: the clone tab needs the base capability, so a
    mismatched sidebar pick auto-routes to the default base alias and the
    success status carries the auto-switch notice (was exact "Finished." only)."""
    audio, status = cbs["voice_clone"]("custom-voice", " target words ", "Auto", tone_pair(0.4), "", True, {})
    assert status == "已自动切换模型至 base (auto-switched model for this tab) · Finished. (生成完成)"
    assert audio is not None


def test_callback_voice_clone_requires_ref_audio_like_official(cbs):
    audio, status = cbs["voice_clone"]("custom-voice", "target", "Auto", None, "transcript", False, {})
    assert audio is None and "Reference audio is required" in status


def test_generation_kwargs_collector_drops_empties_keeps_values(cbs):
    kw = cbs["build_gen_kwargs"](None, None, None, None, None, None, None, None)
    assert kw == {}  # all defaults -> service 512
    kw = cbs["build_gen_kwargs"](1024, 0.7, 40, 0.9, 1.1, None, None, None)
    assert kw == {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    kw = cbs["build_gen_kwargs"](None, 0.5, 0, 0.0, None, 30, 0.95, 0.6)
    assert kw == {
        "temperature": 0.5,
        "top_k": 0,
        "top_p": 0.0,
        "subtalker_top_k": 30,
        "subtalker_top_p": 0.95,
        "subtalker_temperature": 0.6,
    }  # falsy-but-set values kept


def test_generation_honors_advanced_kwargs_state(cbs):
    _audio, status = cbs["custom_voice"](
        "custom-voice", "Hello there", "Auto", "Ryan", "", {"max_new_tokens": 2048}
    )
    assert status == "Finished. (生成完成)"


# ---------------------------------------------------------------------------
# UX-fix B-4: per-tab automatic model routing.  The sidebar Model Switcher is
# global but each tab calls a fixed backend capability, so every generation
# callback resolves its alias through SynthesisService.resolve_alias FIRST and
# reports an auto-switch notice on the success path (error paths keep
# format_error verbatim).  Matching sidebar picks stay notice-free -- pinned by
# the exact "Finished. (生成完成)" assertions in the tests above.
# ---------------------------------------------------------------------------


def test_preset_speakers_tab_routes_off_voice_design_radio():
    """The audited failure: sidebar=VoiceDesign on Tab (2) used to raise the
    official "voice_design ... does not support generate_custom_voice" error;
    now the tab routes itself to custom-voice and says so in the status."""
    cb, factory = make_recording_callbacks()
    audio, status = cb["custom_voice"]("voice-design", " Hello there ", "Auto", "Ryan", "", {})
    assert factory.alias_calls == ["custom-voice"]      # routed, not mis-called
    assert status == "已自动切换模型至 custom-voice (auto-switched model for this tab) · Finished. (生成完成)"
    assert audio is not None


def test_voice_design_tab_routes_off_custom_voice_radio():
    cb, factory = make_recording_callbacks()
    audio, status = cb["voice_design"]("custom-voice", " once upon a time ", "Auto", " calm whisper ", {})
    assert factory.alias_calls == ["voice-design"]
    assert "已自动切换模型至 voice-design" in status and "Finished" in status
    assert audio is not None


def test_voice_clone_tab_routes_off_mismatched_radio():
    cb, factory = make_recording_callbacks()
    audio, status = cb["voice_clone"](
        "voice-design", " target words ", "Auto", tone_pair(0.4), " ref words ", False, {}
    )
    assert factory.alias_calls == ["base"]
    assert "已自动切换模型至 base" in status and "Finished" in status
    sr_out, wav_out = audio
    assert isinstance(sr_out, int) and wav_out.dtype == np.float32


def test_save_voice_subtab_routes_off_mismatched_radio():
    cb, factory = make_recording_callbacks()
    out_path, status = cb["save_voice"]("voice-design", tone_pair(0.4), " hello ref ", xvec_only=False)
    assert factory.alias_calls == ["base"]              # prompt-from-ref is base-kind
    assert "已自动切换模型至 base" in status
    assert out_path and Path(str(out_path)).exists()


def test_load_voice_gen_subtab_routes_off_mismatched_radio(tmp_path: Path):
    import torch

    cb, factory = make_recording_callbacks()
    payload = {"items": [{"ref_code": [[1, 2]], "ref_spk_embedding": [[0.1, 0.2]]}]}
    voice_file = tmp_path / "voice.pt"
    torch.save(payload, str(voice_file))

    audio, status = cb["load_voice_gen"]("voice-design", str(voice_file), " new words ", "Auto", {})
    assert factory.alias_calls == ["base"]              # load-voice-generate is base-kind
    assert "已自动切换模型至 base" in status and audio is not None


def test_callbacks_tolerate_stale_page_alias_after_server_restart():
    """DESYNC case: the browser kept a pre-restart radio value that the running
    registry no longer knows -- the tab still routes to its default, notice on."""
    cb, factory = make_recording_callbacks()
    audio, status = cb["custom_voice"]("pre-restart-alias", " hello ", "Auto", "Ryan", "", {})
    assert factory.alias_calls == ["custom-voice"]
    assert "已自动切换模型至 custom-voice" in status and audio is not None


def test_model_switcher_label_documents_per_tab_auto_routing(app_blocks):
    """B-4 sidebar copy: both languages name the per-tab capability matching."""
    labels = [c.label for c in app_blocks.blocks.values() if getattr(c, "label", None)]
    switcher = [label for label in labels if "Model Switcher" in str(label)]
    assert len(switcher) == 1
    assert "各页签生成时自动匹配能力" in switcher[0]
    assert "each tab auto-selects the matching capability" in switcher[0]


# ---------------------------------------------------------------------------
# Codec tab
# ---------------------------------------------------------------------------


def test_callback_codec_roundtrip_meta_and_downloadable_wav(cbs):
    audio, meta_text, dl = cbs["codec"](tone_pair(0.5))
    assert audio is not None
    sr_out, _wav_out = audio
    assert isinstance(sr_out, int) and sr_out == 24000
    assert "24000" in meta_text and "1920" in meta_text  # rates visible
    assert dl is not None and Path(dl).exists()


def test_callback_codec_rejects_malformed_audio_gracefully(cbs):
    audio, meta_text, dl = cbs["codec"](None)
    assert audio is None and dl is None
    assert meta_text  # error text returned


# ---------------------------------------------------------------------------
# Model switcher: lazy load status with path + VRAM via mem_get_info
# ---------------------------------------------------------------------------


def test_switch_model_status_carries_alias_and_memory_line(cbs):
    status = cbs["switch_model"]("custom-voice")
    assert "custom-voice" in status
    assert CHINESE.search(status)  # bilingual status line


def test_status_line_is_cpu_safe_without_cuda():
    from qwen3_tts_rocm.demo.ui import status_line

    line = status_line(make_service(), "custom-voice")
    assert isinstance(line, str) and "custom-voice" in line  # never raises headless


# ---------------------------------------------------------------------------
# History area: refresh / play / download / delete current-session items
# ---------------------------------------------------------------------------


def test_history_lifecycle_refresh_play_download_delete(cbs, tmp_path: Path):
    assert cbs["history_refresh"]() == []  # empty initially

    _, status = cbs["custom_voice"]("custom-voice", " remembered clip ", "Auto", "Ryan", "", {})
    assert status == "Finished. (生成完成)"

    rows = cbs["history_refresh"]()
    assert len(rows) == 1 and int(rows[0][0]) >= 1  # [id, label, dur, ...]

    hid = int(rows[0][0])
    played = cbs["history_play"](hid)
    assert played is not None and played[0] == 24000  # (sr, wav) preview

    dl_path = cbs["history_download"](hid)
    assert Path(dl_path).exists() and Path(dl_path).stat().st_size > 0

    rows_after = cbs["history_delete"](hid)
    assert rows_after == []
    with pytest.raises(ValueError):
        cbs["history_play"](hid)


def test_history_grows_across_two_generations_fifo_order(cbs):
    cbs["custom_voice"]("custom-voice", "first clip", "Auto", "Ryan", "", {})
    cbs["voice_design"]("voice-design", "second clip", "Auto", "calm", {})
    rows = cbs["history_refresh"]()
    assert len(rows) == 2
    assert int(rows[0][0]) > int(rows[1][0])  # newest first
    assert "VoiceDesign" in rows[0][1]  # label names the origin


# ---------------------------------------------------------------------------
# CLI surface (parser-level; loader/network untouched)
# ---------------------------------------------------------------------------


def test_cli_parser_surface_compatible_with_upstream_plus_models_dir():
    from qwen3_tts_rocm.cli_demo import build_parser

    ns = build_parser().parse_args(
        [
            "--alias",
            "custom-voice",
            "--port",
            "8123",
            "--ip",
            "127.0.0.1",
            "--dtype",
            "float16",
            "--concurrency",
            "4",
            "--models-dir",
            "/tmp/modelz",
        ]
    )
    assert ns.alias == "custom-voice" and ns.port == 8123 and ns.ip == "127.0.0.1"
    assert ns.dtype == "float16"
    assert ns.concurrency == 4  # upstream default was 16
    assert ns.flash_attn is False  # ROCm-safe default
    assert ns.models_dir == "/tmp/modelz"


def test_cli_defaults_match_concurrency_ruling_and_env_override_opt_in():
    from qwen3_tts_rocm.cli_demo import build_parser

    ns = build_parser().parse_args(["custom-voice"])  # positional alias accepted
    assert ns.port == 8000 and ns.ip == "0.0.0.0"
    assert ns.concurrency == 1  # single-GPU queue ruling
    assert ns.share is False and ns.ssl_verify is True
    assert ns.checkpoint is None and ns.checkpoint_pos == "custom-voice"


def test_cli_resolve_target_prefers_checkpoint_over_alias():
    from qwen3_tts_rocm import cli_demo

    def parse(*argv: str):
        return cli_demo.build_parser().parse_args(list(argv))

    ref, _alias_or_none, mode = cli_demo.resolve_target(parse("--checkpoint", "/data/my-model"))
    assert mode == "pinned" and str(ref) == "/data/my-model"

    _r2, alias2, mode2 = cli_demo.resolve_target(parse("--alias", "base-0.6b"))
    assert mode2 == "registry" and alias2 == "base-0.6b"

    _r3, alias3, mode3 = cli_demo.resolve_target(parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"))
    assert mode3 == "registry" and alias3 == "custom-voice-0.6b"


# ---------------------------------------------------------------------------
# UX-fix U3a / A-9: --help must render BooleanOptionalAction flags exactly once
# ---------------------------------------------------------------------------


def test_cli_help_renders_boolean_optional_flags_once():
    """The negative spelling used to render doubled in usage AND options
    (``--flash-attn/--no-flash-attn | --no-flash-attn/--no-flash-attn``):
    argparse's BooleanOptionalAction appended a literal
    ``--no-flash-attn/--no-flash-attn`` variant of the cosmetic slash pair.
    Now no flag string appears twice within one invocation rendering."""
    from qwen3_tts_rocm.cli_demo import build_parser

    help_text = build_parser().format_help()
    for doubled, negative in (
        ("--no-flash-attn/--no-flash-attn", "--no-flash-attn"),
        ("--no-share/--no-share", "--no-share"),
        ("--no-ssl-verify/--no-ssl-verify", "--no-ssl-verify"),
    ):
        assert doubled not in help_text  # no doubled pair rendering
        assert help_text.count(negative) <= 2  # once in usage + once in options


def test_cli_boolean_optional_flags_still_parse_both_spellings():
    """Formatting fix must not disturb the parse surface: both the positive
    and the negative spelling of every BooleanOptionalAction flag keep working
    (and keep their ROCm-safe defaults)."""
    from qwen3_tts_rocm.cli_demo import build_parser

    parse = build_parser().parse_args

    ns = parse(["--no-flash-attn", "--share"])
    assert ns.flash_attn is False and ns.share is True

    ns = parse(["--flash-attn", "--no-share", "--no-ssl-verify"])
    assert ns.flash_attn is True and ns.share is False and ns.ssl_verify is False

    ns = parse(["--ssl-verify"])
    assert ns.ssl_verify is True  # default stays enabled


# ---------------------------------------------------------------------------
# Fix round 1 regressions
# ---------------------------------------------------------------------------


def test_load_voice_file_required_guard_before_factory_touch():
    """SPEC-1: backend owns the official 'Voice file is required' guard."""
    factory_calls: list[str] = []

    def counting_factory(alias: str) -> FakeTTSModel:
        factory_calls.append(str(alias))
        return FakeTTSModel()

    svc = SynthesisService(factory=counting_factory, tokenizer_factory=FakeTokenizer)
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="Voice file is required") as exc:
            svc.load_voice_file(bad)
        assert CHINESE.search(str(exc.value))  # bilingual like upstream

    assert factory_calls == []  # never touched the model
    assert svc.current_alias is None and svc.cached_aliases() == ()
    assert svc.history.list() == []


def test_load_voice_gen_delegates_missing_file_to_backend(cbs):
    """SPEC-1: ui callback carries no validation of its own anymore."""
    _audio, status = cbs["load_voice_gen"]("base", None, "target", "Auto", {})
    assert "Voice file is required" in status
    assert CHINESE.search(status)


def test_speaker_dropdown_accepts_custom_value_pre_seed():
    """SPEC-2: cold-start REST/typing path must pass validation (same fix as
    the Language dropdowns); also pins ALL language dropdowns keep the flag."""
    from qwen3_tts_rocm.demo.ui import build_ui

    app = build_ui(make_service(), {"alias": "custom-voice"})
    labeled = [c for c in app.blocks.values() if getattr(c, "label", None) in ("Speaker (说话人)",)]
    assert len(labeled) == 1
    assert labeled[0].allow_custom_value is True

    lang_dd = [c for c in app.blocks.values() if getattr(c, "label", None) == "Language (语种)"]
    assert len(lang_dd) == 6  # clone/save-load/cv/vd + studio design/reuse
    assert all(dd.allow_custom_value for dd in lang_dd)


def test_callback_accepts_unseeded_speaker_display_via_fallback(cbs):
    """allow_custom_value end-to-end: 'Ryan' is NOT among the seed choices,
    yet the backend fallback resolves it and generation succeeds."""
    audio, status = cbs["custom_voice"]("custom-voice", "Hello custom", "Auto", "Ryan", "", {})
    assert status == "Finished. (生成完成)" and audio is not None


def test_cached_aliases_public_view_is_tolerant():
    """MINOR-3: public accessor replaces private _cache reach-through."""
    svc = make_service()
    assert svc.cached_aliases() == ()  # nothing resident yet

    svc.get("base")
    svc.get("base")  # cached hit, still one
    assert svc.cached_aliases() == ("base",)

    bare = SynthesisService.__new__(SynthesisService)  # __init__ skipped
    assert bare.cached_aliases() == ()  # tolerant of doubles


def test_cli_bare_launch_defaults_to_custom_voice_and_serves(monkeypatch):
    """Bare `docker run ...` (no args) must START SERVING, not print help.

    The untouched upstream CLI keeps its help-and-exit behavior; our enhanced
    entry point defaults --alias to custom-voice for out-of-box UX.
    """
    import types as _types

    import qwen3_tts_rocm.cli_demo as cd
    import qwen3_tts_rocm.demo.ui as ui_mod

    seen: dict = {}

    def fake_build_ui(service, header_info):
        seen["header"] = header_info
        return _types.SimpleNamespace(
            queue=lambda **_kw: fake_build_ui,  # chainable
            launch=lambda **launch_kwargs: seen.setdefault("launch", launch_kwargs),
        )

    monkeypatch.setattr(cd, "resolve_target", lambda args: ("/fake/ref", "custom-voice", "registry"))
    monkeypatch.setattr(cd, "_build_service", lambda *a, **k: _types.SimpleNamespace(unload_all=lambda: None))
    monkeypatch.setattr(ui_mod, "build_ui", fake_build_ui)
    monkeypatch.setattr(ui_mod, "launch_visual_kwargs", dict)

    rc = cd.main([])  # bare launch

    assert rc == 0
    assert seen["header"]["alias"] == "custom-voice"
    assert seen["launch"]["server_port"] == 8000


def test_cli_startup_banner_is_human_one_liner(monkeypatch, capsys):
    """UX-fix U2: banner names the alias, device and local URL in one line."""
    import types as _types

    import qwen3_tts_rocm.cli_demo as cd
    import qwen3_tts_rocm.demo.ui as ui_mod

    def fake_build_ui(service, header_info):
        return _types.SimpleNamespace(
            queue=lambda **_kw: fake_build_ui,
            launch=lambda **_kw: None,
        )

    monkeypatch.setattr(cd, "resolve_target", lambda args: ("/fake/ref", "base", "registry"))
    monkeypatch.setattr(cd, "_build_service", lambda *a, **k: _types.SimpleNamespace(unload_all=lambda: None))
    monkeypatch.setattr(ui_mod, "build_ui", fake_build_ui)
    monkeypatch.setattr(ui_mod, "launch_visual_kwargs", dict)

    rc = cd.main(["--alias", "base", "--device", "cpu", "--port", "8123"])
    out = capsys.readouterr().out

    assert rc == 0
    banner = [ln for ln in out.splitlines() if ln.startswith("[qwen3-tts-rocm]")]
    assert len(banner) == 1  # exactly one human line by default
    assert "模型 Model: base" in banner[0]
    assert "设备 device: cpu" in banner[0]
    assert "就绪后打开 open: http://localhost:8123" in banner[0]
    assert "mode=" not in out  # jargon dropped from the default banner


def test_cli_debug_env_keeps_old_target_line(monkeypatch, capsys):
    """QWEN3_TTS_ROCM_DEBUG=1 reprints the raw target/mode line for support."""
    import types as _types

    import qwen3_tts_rocm.cli_demo as cd
    import qwen3_tts_rocm.demo.ui as ui_mod

    def fake_build_ui(service, header_info):
        return _types.SimpleNamespace(
            queue=lambda **_kw: fake_build_ui,
            launch=lambda **_kw: None,
        )

    monkeypatch.setattr(cd, "resolve_target", lambda args: ("/fake/ref", "base", "registry"))
    monkeypatch.setattr(cd, "_build_service", lambda *a, **k: _types.SimpleNamespace(unload_all=lambda: None))
    monkeypatch.setattr(ui_mod, "build_ui", fake_build_ui)
    monkeypatch.setattr(ui_mod, "launch_visual_kwargs", dict)
    monkeypatch.setenv("QWEN3_TTS_ROCM_DEBUG", "1")

    rc = cd.main(["--alias", "base"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "target='/fake/ref'" in out and "mode=registry" in out  # raw debug line kept


def test_cli_port_busy_prints_bilingual_hint_and_returns_2(monkeypatch, capsys):
    """UX-fix U2: gradio 'no empty port' OSError -> friendly hint + rc 2, no traceback."""
    import types as _types

    import qwen3_tts_rocm.cli_demo as cd
    import qwen3_tts_rocm.demo.ui as ui_mod

    unloaded: list[str] = []

    def fake_build_ui(service, header_info):
        def launch(**_kw):
            raise OSError("Cannot find empty port in range: 8000-8000")

        return _types.SimpleNamespace(
            queue=lambda **_kw: fake_build_ui,
            launch=launch,
        )

    monkeypatch.setattr(cd, "resolve_target", lambda args: ("/fake/ref", "custom-voice", "registry"))
    monkeypatch.setattr(
        cd, "_build_service",
        lambda *a, **k: _types.SimpleNamespace(unload_all=lambda: unloaded.append("x")),
    )
    monkeypatch.setattr(ui_mod, "build_ui", fake_build_ui)
    monkeypatch.setattr(ui_mod, "launch_visual_kwargs", dict)

    rc = cd.main(["--port", "8000"])
    out = capsys.readouterr().out

    assert rc == 2
    assert "ERROR:" in out
    assert "端口 8000 被占用" in out and "port busy" in out
    assert "--port 8001" in out  # concrete retry suggestion
    assert unloaded == ["x"]  # finally: unload_all still released the model


# ---------------------------------------------------------------------------
# Regression (2026-08 first-user journey, F4): the post-generation chain must
# refresh the sidebar status READ-ONLY.  It used to chain cb["switch_model"],
# force-loading the sidebar pick after EVERY generate click — even a failed one
# on a tab whose capability differs from the pick — surprise-loading ~4-5 GiB
# and, with the size-1 LRU, evicting the model the generation just used.
# ---------------------------------------------------------------------------


def test_generation_chains_do_not_register_switch_model(app_blocks):
    """Exactly ONE switch_model registration may exist in the Blocks graph —
    the sidebar radio's own change handler.  None of the four generation
    click chains (preset / design / clone / load-voice) may force-load."""
    entries = list(app_blocks.fns.values())
    names = [getattr(dep.fn, "__name__", "") for dep in entries]
    assert names.count("switch_model") == 1
    # The survivor is the eager sidebar-load (radio change), registered before
    # every generation handler.
    assert names.index("switch_model") < names.index("_gen_cv")


def test_generation_chain_followers_are_status_line_lambdas(app_blocks):
    """Each generation click chain still refreshes the status box: the entry
    right after every generation handler is the read-only status_line lambda,
    never the service.get-backed switch_model closure."""
    entries = list(app_blocks.fns.values())
    names = [getattr(dep.fn, "__name__", "") for dep in entries]
    for gen in ("_gen_cv", "_gen_vd", "_gen_clone", "_gen_loaded"):
        follower = entries[names.index(gen) + 1]
        assert getattr(follower.fn, "__name__", "") != "switch_model"
        assert "status_line" in str(getattr(follower.fn, "__closure__", None) or ()) or \
            getattr(follower.fn, "__qualname__", "") == "build_callbacks.<locals>.<lambda>"


# ---------------------------------------------------------------------------
# ⑥ Voice Studio (音色工坊): describe -> preview -> save -> reuse with NO
# download/re-upload hop.  The studio callbacks are Blocks-free like every
# other key in build_callbacks; the UI-shape tests pin that the tab wires
# them and that the flow crosses no gr.File boundary.
# ---------------------------------------------------------------------------


def make_studio_service(tmp_path: Path) -> SynthesisService:
    return SynthesisService(
        factory=lambda alias: FakeTTSModel(),
        tokenizer_factory=FakeTokenizer,
        voices_dir=tmp_path,
    )


def test_build_ui_constructs_voice_studio_tab(app_blocks):
    """⑥ appears after ⑤ History with the bilingual studio labels."""
    ids = [str(c.label) for c in app_blocks.blocks.values() if hasattr(c, "label")]
    assert any("Voice Studio" in l and "音色工坊" in l for l in ids)
    assert any("Voice Description" in l for l in ids)
    assert any("Saved Voice" in l for l in ids)
    assert any("Design Timings" in l for l in ids)


def test_build_ui_voice_studio_adds_no_file_components(app_blocks):
    """The no-hop guarantee, structurally: the studio flow crosses NO file
    boundary -- the four pre-existing gr.File components (save-voice output,
    prompt upload, codec download, history download) are still exactly four,
    so the studio save/generate path runs purely through the in-session id
    and the saved-name dropdown."""
    files = [c for c in app_blocks.blocks.values() if type(c).__name__ == "File"]
    assert len(files) == 4


def test_callback_voice_studio_design_save_generate_roundtrip(tmp_path: Path):
    from qwen3_tts_rocm.demo.ui import build_callbacks

    svc = make_studio_service(tmp_path)
    cb = build_callbacks(svc)

    audio, status, voice_id, timing_text = cb["voice_studio_design"](
        "voice-design", " hello studio ", "Auto", "bright female", {}
    )
    assert status == "Finished. (生成完成)"
    assert voice_id.startswith("voice-")
    sr, wav = audio
    assert isinstance(sr, int) and wav.dtype == np.float32
    assert "design" in timing_text and "prompt" in timing_text  # phases apart

    save_status, dropdown_update = cb["voice_studio_save"](voice_id, " my_voice ")
    assert save_status.startswith("Saved") and "已保存" in save_status
    assert dropdown_update.get("choices") == ["my_voice"]
    assert cb["voice_studio_list"]() == ["my_voice"]

    audio2, status2, gen_timing = cb["voice_studio_generate"](
        "base", "my_voice", " new sentence ", "Auto", {}
    )
    assert status2 == "Finished. (生成完成)" and audio2 is not None
    assert gen_timing.endswith("s")                        # reuse timed apart
    calls = svc.get("base").calls
    assert calls[-1]["method"] == "generate_voice_clone"
    assert calls[-1]["voice_clone_prompt"] is not None


def test_callback_voice_studio_design_reports_error_not_raise(tmp_path: Path):
    from qwen3_tts_rocm.demo.ui import build_callbacks

    cb = build_callbacks(make_studio_service(tmp_path))
    audio, status, voice_id, timing = cb["voice_studio_design"](
        "voice-design", "   ", "Auto", "bright female", {}
    )
    assert audio is None and voice_id == "" and timing == ""
    assert "Text is required" in status and CHINESE.search(status)

    save_status, upd = cb["voice_studio_save"]("voice-404", "name")
    assert "Unknown studio voice id" in save_status and CHINESE.search(save_status)
    assert not upd.get("choices")                          # dropdown untouched


def test_callback_voice_studio_generate_missing_voice_reports_error(tmp_path: Path):
    from qwen3_tts_rocm.demo.ui import build_callbacks

    cb = build_callbacks(make_studio_service(tmp_path))
    audio, status, gen_timing = cb["voice_studio_generate"](
        "base", "ghost", "words", "Auto", {}
    )
    assert audio is None and gen_timing == ""
    assert "not found" in status and CHINESE.search(status)


def test_voice_studio_design_tab_routes_off_mismatched_radio(tmp_path: Path):
    """B-4 routing for the studio: a base sidebar pick auto-switches to the
    tab's own VoiceDesign capability for the design phase."""
    factory = RecordingFactory()
    service = SynthesisService(factory=factory, tokenizer_factory=FakeTokenizer,
                               voices_dir=tmp_path)
    from qwen3_tts_rocm.demo.ui import build_callbacks

    cb = build_callbacks(service)
    _audio, status, _vid, _t = cb["voice_studio_design"](
        "base", "hello studio", "Auto", "bright female", {}
    )
    assert factory.alias_calls == ["voice-design", "base"]
    assert "已自动切换模型至 voice-design" in status and "Finished" in status


def test_voice_studio_generation_records_into_history(tmp_path: Path):
    from qwen3_tts_rocm.demo.ui import build_callbacks

    cb = build_callbacks(make_studio_service(tmp_path))
    cb["voice_studio_design"]("voice-design", "hello studio", "Auto", "bright female", {})
    rows = cb["history_refresh"]()
    assert len(rows) == 1 and "VoiceStudio-design" in rows[0][1]
