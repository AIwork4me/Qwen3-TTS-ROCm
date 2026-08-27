# tests/test_demo_ui.py
"""Task 16: enhanced four-tab bilingual Gradio application (demo/ui.py + cli_demo).

Hermetic UI-layer coverage, zero GPU / zero weights / browser-free:

* **UI shape**        -- ``build_ui(service, header_info)`` constructs a
  ``gr.Blocks`` carrying the four product tabs (Reference clone / Speaker
  preset / Voice Design / Codec), the History area, the official disclaimer
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


def test_build_ui_constructs_four_tabs(app_blocks):
    ids = {c.label for c in app_blocks.blocks.values() if hasattr(c, "label")}
    assert any(l and "Reference" in l for l in ids)  # clone tab
    assert any(l and "Speaker" in l for l in ids)  # preset tab
    assert any(l and "Voice Design" in l for l in ids)  # design tab
    assert any(l and "Codec" in l for l in ids)  # codec tab


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
    audio, status = cbs["voice_clone"]("custom-voice", " target words ", "Auto", tone_pair(0.4), "", True, {})
    assert status == "Finished. (生成完成)" and audio is not None


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
