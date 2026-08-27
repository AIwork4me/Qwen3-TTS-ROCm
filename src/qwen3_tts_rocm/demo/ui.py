"""Enhanced five-tab bilingual Gradio application (增强版双语文字演示界面).

Layer contract (Task 16)
------------------------
* **Upstream visual ground truth** -- theme ``gr.themes.Soft`` with Source Sans
  Pro, full-width CSS, ``# Qwen3 TTS Demo`` header markdown with Checkpoint +
  Model Type lines, SR-first :func:`_wav_to_gradio_audio`, the
  ``Finished. (生成完成)`` status string and the complete bilingual disclaimer
  footer are reused VERBATIM from the installed official
  ``qwen_tts/cli/demo.py``.
* **Thin wrappers only** -- every business decision lives in
  :mod:`qwen3_tts_rocm.demo.backend` (Task 15).  This module exposes
  :func:`build_callbacks`, a service-bound dict of Blocks-free callables that
  pytest drives end-to-end without a browser; :func:`build_ui` merely lays out
  components and wires those very functions onto events.
* **Single-GPU queue ruling** -- the app targets gfx1151-style unified-memory
  APUs where exactly ONE TTS model stays resident, so deployments run the
  queue at ``default_concurrency_limit=1`` (see the visible footer note).
* **Hermetic imports** -- like the backend layer, no torch/qwen_tts import at
  module scope; both appear lazily inside the few callbacks that touch files.

Tabs (design doc §5.5): Voice Clone (reference audio, incl. the official
Save/Load Voice sub-tab), Preset Speakers, Voice Design, Codec roundtrip, and
the generation History area (play / download / delete for the CURRENT
session's clips; waveforms live ONLY inside the service's HistoryStore).
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict
from typing import Any

import gradio as gr
import numpy as np

from .. import models
from .backend import display_map, format_error

__all__ = [
    "CONCURRENCY_NOTE",
    "DEFAULT_ALIAS",
    "DEFAULT_TITLE",
    "DISCLAIMER",
    "build_callbacks",
    "build_ui",
    "launch_visual_kwargs",
    "status_line",
]

#: Default model of the switcher radio (custom-voice per Task brief).
DEFAULT_ALIAS = "custom-voice"

#: Page <title>; the curl smoke test anchors on the literal "Qwen3 TTS".
DEFAULT_TITLE = "Qwen3 TTS Demo"

#: Official theme/css, character-for-character.
_CSS = ".gradio-container {max-width: none !important;}"

_STATUS_FINISHED = "Finished. (生成完成)"


def _auto_switch_status(alias: str) -> str:
    """B-4 success status when a tab had to route away from the sidebar pick.

    The sidebar Model Switcher is global while each tab calls a fixed backend
    capability; when the two disagree the callback resolves through
    :meth:`~qwen3_tts_rocm.demo.backend.SynthesisService.resolve_alias` and
    this notice replaces the plain ``Finished`` line so the silent correction
    is visible (error paths keep :func:`format_error` verbatim).
    """
    return f"已自动切换模型至 {alias} (auto-switched model for this tab) · {_STATUS_FINISHED}"

#: Bilingual label table for the model-switcher radio (all usable TTS aliases;
#: the speech tokenizer entry of the registry is intentionally NOT generatable).
_ALIAS_LABELS: dict[str, str] = {
    "custom-voice": "CustomVoice 预设音色 1.7B",
    "voice-design": "VoiceDesign 音色设计 1.7B",
    "base": "Base 克隆基座 1.7B",
    "custom-voice-0.6b": "CustomVoice 预设音色 0.6B",
    "base-0.6b": "Base 克隆基座 0.6B",
}

#: Visible queue note required by the concurrency ruling (footer area).
CONCURRENCY_NOTE = (
    "**Queue note (排队串行生成说明)**\n"
    "This deployment runs the Gradio queue at concurrency 1 over the "
    "single-GPU unified-memory device: synthesis requests are serialized one "
    "at a time (所有生成请求经单 GPU 统一内存队列按顺序一次一个执行，请在结果出现后"
    "再提交下一条)。"
)

#: Official disclaimer footer, copied verbatim from qwen_tts/cli/demo.py.
DISCLAIMER = """**Disclaimer (免责声明)**  
- The audio is automatically generated/synthesized by an AI model solely to demonstrate the model’s capabilities; it may be inaccurate or inappropriate, does not represent the views of the developer/operator, and does not constitute professional advice. You are solely responsible for evaluating, using, distributing, or relying on this audio; to the maximum extent permitted by applicable law, the developer/operator disclaims liability for any direct, indirect, incidental, or consequential damages arising from the use of or inability to use the audio, except where liability cannot be excluded by law. Do not use this service to intentionally generate or replicate unlawful, harmful, defamatory, fraudulent, deepfake, or privacy/publicity/copyright/trademark‑infringing content; if a user prompts, supplies materials, or otherwise facilitates any illegal or infringing conduct, the user bears all legal consequences and the developer/operator is not responsible.
- 音频由人工智能模型自动生成/合成，仅用于体验与展示模型效果，可能存在不准确或不当之处；其内容不代表开发者/运营方立场，亦不构成任何专业建议。用户应自行评估并承担使用、传播或依赖该音频所产生的一切风险与责任；在适用法律允许的最大范围内，开发者/运营方不对因使用或无法使用本音频造成的任何直接、间接、附带或后果性损失承担责任（法律另有强制规定的除外）。严禁利用本服务故意引导生成或复制违法、有害、诽谤、欺诈、深度伪造、侵犯隐私/肖像/著作权/商标等内容；如用户通过提示词、素材或其他方式实施或促成任何违法或侵权行为，相关法律后果由用户自行承担，与开发者/运营方无关。"""

_REF_TEXT_NOTE = "Required if not set use x-vector only (不勾选use x-vector only时必填)."


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Official helper verbatim: ``(sr, float32 wav)`` for Audio(type=numpy)."""
    return int(sr), np.asarray(wav, dtype=np.float32)


def _history(service) -> Any:
    """HistoryStore accessor tolerant of ``__new__``-built test doubles."""
    store = getattr(service, "history", None)
    if store is None:  # doubles constructed without __init__
        from .backend import HistoryStore

        store = HistoryStore()
        service.history = store
    return store


def _loaded_model(service, alias: str) -> Any:
    """Resident model for *alias*, or ``None`` when not loaded.

    Goes through the PUBLIC :meth:`SynthesisService.cached_aliases` membership
    check first; only then touches the cache dict for the instance itself,
    with a getattr fallback for ``__new__``-built test doubles.
    """
    try:
        if str(alias) not in set(service.cached_aliases()):
            return None
    except AttributeError:  # double without __init__
        pass
    return getattr(service, "_cache", {}).get(str(alias))


def _resolved_path_text(alias: str) -> str:
    """Human-readable load path for *alias* (never raises)."""
    try:
        return str(models.resolve_path(alias))
    except Exception:  # noqa: BLE001 - status must not crash UI
        return "(unknown location / 未知路径)"


def _vram_text() -> str:
    """torch.cuda.mem_get_info one-liner; CPU-safe fallback, never raises."""
    try:
        import torch  # lazy; CI runners may have no torch

        if not torch.cuda.is_available():
            return "VRAM/GTT: unavailable (无可见 GPU 设备)"
        free, total = torch.cuda.mem_get_info()
        return f"VRAM/GTT: used {max(total - free, 0) / 2**30:.1f} GiB · total {total / 2**30:.1f} GiB"
    except Exception:  # noqa: BLE001 - probing never blocks UI
        return "VRAM/GTT: unavailable (无法读取显存)"


def status_line(service, alias: str) -> str:
    """Sidebar status: alias, resolved load path and live memory usage."""
    loaded = str(alias) in set(service.cached_aliases())
    mark = "loaded (已驻留)" if loaded else "not loaded yet (未加载，首次使用时懒加载)"
    return f"[{alias}] {mark}\npath: {_resolved_path_text(alias)}\n{_vram_text()}"


def _write_temp_wav(sr: int, wav: np.ndarray, prefix: str) -> str:
    """Materialise a serving copy of an in-memory waveform (.wav temp file)."""
    import soundfile as sf

    fd, out_path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(fd)
    sf.write(out_path, np.asarray(wav, dtype=np.float32), int(sr))
    return out_path


def _audio_pair_or_none(audio_value: Any) -> tuple[int, np.ndarray] | None:
    """Accept Gradio Audio(type=numpy) values as an ``(sr, wav)`` pair."""
    if isinstance(audio_value, (tuple, list)) and len(audio_value) == 2:
        try:
            return int(audio_value[0]), np.asarray(audio_value[1])
        except (TypeError, ValueError):
            return None
    return None


def build_gen_kwargs(
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    subtalker_top_k,
    subtalker_top_p,
    subtalker_temperature,
) -> dict[str, Any]:
    """Collapse the eight Advanced accordion fields into a kwargs dict.

    Empty means default (service-side latency guardrail applies); set values --
    including falsy ones such as ``top_k=0`` -- win, mirroring the official CLI
    convention where unset flags are absent rather than ``None``.
    """
    values = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "subtalker_top_k": subtalker_top_k,
        "subtalker_top_p": subtalker_top_p,
        "subtalker_temperature": subtalker_temperature,
    }
    return {k: v for k, v in values.items() if v is not None}


def build_callbacks(service) -> dict[str, Any]:
    """Return the Blocks-free callback dict wired to *service*.

    Keys: ``custom_voice``, ``voice_design``, ``voice_clone``, ``save_voice``,
    ``load_voice_gen``, ``codec``, ``switch_model``, ``status_line``,
    ``refresh_choices``, ``build_gen_kwargs``, ``history_refresh`` /
    ``history_play`` / ``history_download`` / ``history_delete`` /
    ``history_id_at``.  Every callable is intentionally thin: validate nothing,
    decide nothing -- delegate to the backend and normalise failures through
    :func:`~qwen3_tts_rocm.demo.backend.format_error`.  The one decision each
    generation callback makes (UX-fix B-4) is ALSO backend-owned: it resolves
    the global sidebar alias against its tab's required capability via
    ``service.resolve_alias`` first, appending the bilingual auto-switch
    notice to the success status when routing had to move off the pick.
    """
    store = _history(service)

    # -- generation ---------------------------------------------------------

    def _record(label: str, sr: int, wav: Any) -> None:
        """Drop one finished clip into the shared history (best-effort)."""
        try:
            store.add(label, int(sr), wav)
        except Exception:  # noqa: BLE001,S110 - history is optional
            pass

    def run_custom_voice(alias, text, language_display, speaker_display, instruct, gen_kwargs):
        try:
            used_alias, switched = service.resolve_alias("custom_voice", alias)
            sr, wav = service.custom_voice(
                used_alias,
                text,
                language_display=language_display,
                speaker_display=speaker_display,
                instruct=instruct,
                gen_kwargs=dict(gen_kwargs or {}),
            )
            _record(f"CustomVoice [{used_alias}] {text[:20]}", sr, wav)
            status = _auto_switch_status(used_alias) if switched else _STATUS_FINISHED
            return _wav_to_gradio_audio(wav, sr), status
        except Exception as exc:  # noqa: BLE001 - surfaced in the status box
            return None, format_error(exc)

    def run_voice_design(alias, text, language_display, instruct, gen_kwargs):
        try:
            used_alias, switched = service.resolve_alias("voice_design", alias)
            sr, wav = service.voice_design(
                used_alias,
                text,
                language_display=language_display,
                instruct=instruct,
                gen_kwargs=dict(gen_kwargs or {}),
            )
            _record(f"VoiceDesign [{used_alias}] {text[:20]}", sr, wav)
            status = _auto_switch_status(used_alias) if switched else _STATUS_FINISHED
            return _wav_to_gradio_audio(wav, sr), status
        except Exception as exc:  # noqa: BLE001
            return None, format_error(exc)

    def run_voice_clone(alias, text, language_display, ref_audio_value, ref_text, xvec_only, gen_kwargs):
        try:
            used_alias, switched = service.resolve_alias("base", alias)
            pair = _audio_pair_or_none(ref_audio_value)
            sr, wav = service.voice_clone(
                used_alias,
                text,
                language_display=language_display,
                ref_audio=pair,
                ref_text=ref_text,
                xvec_only=bool(xvec_only),
                gen_kwargs=dict(gen_kwargs or {}),
            )
            _record(f"VoiceClone [{used_alias}] {text[:20]}", sr, wav)
            status = _auto_switch_status(used_alias) if switched else _STATUS_FINISHED
            return _wav_to_gradio_audio(wav, sr), status
        except Exception as exc:  # noqa: BLE001
            return None, format_error(exc)

    def save_voice(alias, ref_audio_value, ref_text, xvec_only):
        """Official save_prompt: items -> torch.save {"items": [...]}.pt."""
        try:
            import torch  # lazy

            used_alias, switched = service.resolve_alias("base", alias)
            pair = _audio_pair_or_none(ref_audio_value)
            items = service.clone_prompt_from_ref(
                used_alias, ref_audio=pair, ref_text=ref_text, xvec_only=bool(xvec_only)
            )
            fd, out_path = tempfile.mkstemp(prefix="qwen3_tts_voice_", suffix=".pt")
            os.close(fd)
            torch.save(_savable_payload(items), out_path)
            return out_path, (_auto_switch_status(used_alias) if switched else _STATUS_FINISHED)
        except Exception as exc:  # noqa: BLE001
            return None, format_error(exc)

    def load_voice_gen(alias, file_obj, text, language_display, gen_kwargs):
        """Official load_prompt_and_gen through backend.load_voice_file."""
        try:
            used_alias, switched = service.resolve_alias("base", alias)
            # Backend owns validation too: a missing/unresolvable file raises
            # the official "Voice file is required" ValueError inside
            # service.load_voice_file and lands here like any other failure.
            items = service.load_voice_file(file_obj)
            sr, wav = service.voice_clone_with_prompt(
                used_alias,
                text,
                language_display=language_display,
                items=items,
                gen_kwargs=dict(gen_kwargs or {}),
            )
            _record(f"VoiceClone-loaded [{used_alias}] {text[:20]}", sr, wav)
            status = _auto_switch_status(used_alias) if switched else _STATUS_FINISHED
            return _wav_to_gradio_audio(wav, sr), status
        except Exception as exc:  # noqa: BLE001
            return None, (
                "Failed to read or use voice file. Check file format/content.\n"
                "(读取或使用音色文件失败，请检查文件格式或内容)\n" + format_error(exc)
            )

    def run_codec(audio_value):
        """Encode->decode roundtrip; meta text + downloadable wav out."""
        try:
            pair = _audio_pair_or_none(audio_value)
            sr_in = int(pair[0]) if pair else -1
            n_in = int(np.asarray(pair[1]).size) if pair else 0
            sr_out, wav_out, meta = service.codec_roundtrip(pair)
            meta_text = _format_meta(meta, sr_in, n_in, sr_out, wav_out.size)
            return (
                _wav_to_gradio_audio(wav_out, sr_out),
                meta_text,
                _write_temp_wav(sr_out, wav_out, "qwen3_tts_codec_out_"),
            )
        except Exception as exc:  # noqa: BLE001
            return None, format_error(exc), None

    # -- sidebar --------------------------------------------------------------

    def switch_model(alias):
        """Explicitly (lazily) load the picked alias; bilingual status back."""
        try:
            service.get(str(alias))
        except Exception as exc:  # noqa: BLE001 - shown, server stays up
            return f"[{alias}] load failed (加载失败)\n{format_error(exc)}\n{_vram_text()}"
        return status_line(service, alias)

    def refresh_choices(alias):
        """Latest display choices from the LOADED model only (no force load).

        Returns ``(languages_or_None, speakers_or_None)``; ``None`` leaves the
        dropdown untouched when the model lacks that getter or isn't loaded.
        """
        model = _loaded_model(service, alias)
        if model is None:
            return None, None
        langs = spks = None
        getter = getattr(model, "get_supported_languages", None)
        if callable(getter):
            langs, _ = display_map(list(getter()))
        getter = getattr(model, "get_supported_speakers", None)
        if callable(getter):
            spks, _ = display_map(list(getter()))
        return langs, spks

    # -- history --------------------------------------------------------------

    def history_refresh() -> list[list]:
        rows = []
        for item in store.list():
            stamp = time.strftime("%H:%M:%S", time.localtime(item["created_at"]))
            origin = "VoiceDesign (音色设计)" if "VoiceDesign" in item["label"] else item["label"]
            rows.append([item["id"], origin, f"{item['duration_s']:.2f}", stamp])
        return rows

    def history_play(history_id: int):
        """Strict preview: unknown/evicted ids raise the backend's ValueError."""
        sr, wav = store.item(int(history_id))
        return _wav_to_gradio_audio(wav, sr)

    def history_download(history_id: int):
        sr, wav = store.item(int(history_id))  # ValueError -> caller surfaces
        return _write_temp_wav(sr, wav, "qwen3_tts_history_")

    def history_delete(history_id: int) -> list[list]:
        store.remove(int(history_id))
        return history_refresh()

    def history_id_at(row_index) -> int | None:
        """Map a clicked Dataframe row position to its history id."""
        rows = store.list()
        try:
            return int(rows[int(row_index)]["id"])
        except Exception:  # noqa: BLE001 - table changed meanwhile
            return None

    return {
        "custom_voice": run_custom_voice,
        "voice_design": run_voice_design,
        "voice_clone": run_voice_clone,
        "save_voice": save_voice,
        "load_voice_gen": load_voice_gen,
        "codec": run_codec,
        "switch_model": switch_model,
        "status_line": lambda alias: status_line(service, alias),
        "refresh_choices": refresh_choices,
        "build_gen_kwargs": build_gen_kwargs,
        "history_refresh": history_refresh,
        "history_play": history_play,
        "history_download": history_download,
        "history_delete": history_delete,
        "history_id_at": history_id_at,
    }


def _format_meta(meta: dict[str, Any], sr_in: int, n_in: int, sr_out: int, n_out: int) -> str:
    """Codec tab meta text block: rates + codes/steps-per-second info."""
    downsample = meta.get("encode_downsample")
    lines = [
        f"model_type={meta.get('model_type', '?')}",
        (
            f"input {sr_in} Hz ({n_in / max(sr_in, 1):.2f}s) -> "
            f"output {sr_out} Hz ({n_out / max(sr_out, 1):.2f}s)"
        ),
        f"encode_downsample={downsample} · decode_upsample={meta.get('decode_upsample', '?')}",
    ]
    if isinstance(downsample, (int, float)) and downsample:
        lines.append(f"codes ≈ {float(sr_in) / float(downsample):.2f} tokens/steps per second (编码步速)")
    if "codes_shape" in meta:
        lines.append(f"codes_shape={tuple(meta['codes_shape'])}")
    ratio = (n_in / max(sr_in, 1)) > 0 and (n_in / max(sr_in, 1))
    if ratio:
        speed = (n_out / max(sr_out, 1)) / ratio
        lines.append(f"roundtrip ≈ {speed:.2f}× realtime")
    return "\n".join(lines)


def launch_visual_kwargs() -> dict[str, Any]:
    """Visual kwargs for ``Blocks.launch()`` under Gradio 6 (theme + css).

    The official demo look -- ``gr.themes.Soft`` with Source Sans Pro and the
    full-width container CSS -- lives here because Gradio 6 moved these
    parameters from the ``Blocks(...)`` constructor into ``launch()``.
    """
    return {
        "theme": gr.themes.Soft(
            font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
        ),
        "css": _CSS,
    }


def _savable_payload(items: list) -> dict[str, Any]:
    """Official ``{"items": [asdict ...]}`` schema, loadable by both demos.

    Real models put torch tensors inside their prompt items -- those are the
    official payload types, so they are forwarded VERBATIM (identical bytes to
    what ``qwen_tts/cli/demo.py`` saves).  Plain array-likes (test doubles,
    third-party callers) become builtin lists, which survive
    ``torch.load(..., weights_only=True)`` on every torch version and which
    the official reconstruction consumes equally well via ``torch.tensor``.
    """
    rows = []
    for it in items:
        row = asdict(it)
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
        rows.append(row)
    return {"items": rows}


def _header_markdown(header_info: dict[str, Any] | None) -> str:
    """Official header block with Checkpoint + Model Type lines."""
    info = dict(header_info or {})
    checkpoint = str(info.get("checkpoint") or info.get("alias") or DEFAULT_ALIAS)
    model_type = str(info.get("model_type") or "(auto-detected after first load)")
    return f"# Qwen3 TTS Demo\n**Checkpoint:** `{checkpoint}`  \n**Model Type:** `{model_type}`  \n"


def build_ui(service, port_header_info: dict[str, Any] | None = None) -> gr.Blocks:
    """Assemble the enhanced five-tab demo application around *service*."""
    header_info = dict(port_header_info or {})
    default_alias = str(header_info.get("alias") or DEFAULT_ALIAS)

    # Service-bound thin callbacks (same objects pytest drives headlessly).
    cb = build_callbacks(service)

    alias_items = [(label, alias) for alias, label in _ALIAS_LABELS.items() if alias in models.REPOS]
    radio_value = next((a for _l, a in alias_items if a == default_alias), alias_items[0][1])

    kw_fields_spec = (
        ("Max New Tokens (最大新 token 数，空=默认512)", None),
        ("Temperature 温度 (空=默认)", None),
        ("Top-K (空=默认)", None),
        ("Top-P (空=默认)", None),
        ("Repetition Penalty (重复惩罚，空=默认)", None),
        ("Subtalker Top-K (子码本Top-K，仅v2分词器，空=默认)", None),
        ("Subtalker Top-P (子码本Top-P，仅v2分词器，空=默认)", None),
        ("Subtalker Temperature (子码本温度，仅v2分词器，空=默认)", None),
    )

    # Gradio 6 moved theme/css from Blocks(...) into launch(); cli_demo.py
    # forwards these so the official Soft/Source-Sans-Pro look is preserved.
    with gr.Blocks(title=DEFAULT_TITLE) as demo:
        # ---------------- global sidebar -----------------------------------
        with gr.Sidebar() if hasattr(gr, "Sidebar") else gr.Column(scale=1) as _sidebar:
            model_radio = gr.Radio(
                label=(
                    "Model Switcher (模型切换器 · 各页签生成时自动匹配能力 / "
                    "each tab auto-selects the matching capability)"
                ),
                choices=alias_items,
                value=radio_value,
                interactive=True,
            )
            model_status = gr.Textbox(
                label="Load Status / VRAM (加载状态·显存)",
                value=cb["status_line"](radio_value),
                lines=4,
                interactive=False,
            )
            gr.Markdown("_Model metadata appears after first load (首次加载后显示模型元数据)_")
            with gr.Accordion("Advanced Sampling Parameters (高级采样参数)", open=False):
                kw_fields = [
                    gr.Number(label=label, value=value, precision=None) for label, value in kw_fields_spec
                ]
            gen_kwargs_state = gr.State({})
            for field in kw_fields:
                field.change(cb["build_gen_kwargs"], inputs=kw_fields, outputs=gen_kwargs_state)

        # ---------------- main column --------------------------------------
        with gr.Column(scale=4):
            gr.Markdown(_header_markdown(header_info))
            with gr.Tabs():
                # ============ ① Voice Clone ================================
                with gr.Tab("① Voice Clone (语音克隆)"), gr.Tabs():
                    with gr.Tab("Clone & Generate (克隆并合成)"), gr.Row():
                        with gr.Column(scale=2):
                            ref_audio = gr.Audio(
                                label="Reference Audio (参考音频)",
                                sources=["upload", "microphone"],
                                type="numpy",
                            )
                            ref_text = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder=_REF_TEXT_NOTE,
                            )
                            xvec_only = gr.Checkbox(
                                label=(
                                    "Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)"
                                ),
                                value=False,
                            )
                            btn_clone = gr.Button(
                                "Generate (生成)",
                                variant="primary",
                            )
                        with gr.Column(scale=2):
                            clone_target = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            clone_lang = gr.Dropdown(
                                label="Language (语种)",
                                choices=["Auto"],
                                value="Auto",
                                interactive=True,
                                allow_custom_value=True,  # pre-load typing/REST
                            )
                        with gr.Column(scale=3):
                            clone_audio = gr.Audio(label="Output Audio (合成结果)")
                            clone_status = gr.Textbox(label="Status (状态)", lines=2)
                    with gr.Tab("Save / Load Voice (保存/加载音色)"), gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Save Voice (保存音色)
Upload reference audio and text, choose use x-vector only or not, then save a reusable voice prompt file.  
(上传参考音频和参考文本，选择是否使用 use x-vector only 模式后保存为可复用的音色文件)
"""
                            )
                            ref_audio_s = gr.Audio(
                                label="Reference Audio (参考音频)",
                                sources=["upload", "microphone"],
                                type="numpy",
                            )
                            ref_text_s = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder=_REF_TEXT_NOTE,
                            )
                            xvec_only_s = gr.Checkbox(
                                label=(
                                    "Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)"
                                ),
                                value=False,
                            )
                            save_btn = gr.Button("Save Voice File (保存音色文件)", variant="primary")
                            prompt_file_out = gr.File(label="Voice File (音色文件)")

                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Load Voice & Generate (加载音色并合成)
Upload a previously saved voice file, then synthesize new text.  
(上传已保存提示文件后，输入新文本进行合成)
"""
                            )
                            prompt_file_in = gr.File(
                                label="Upload Prompt File (上传提示文件)", file_types=[".pt"]
                            )
                            load_target = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            load_lang = gr.Dropdown(
                                label="Language (语种)",
                                choices=["Auto"],
                                value="Auto",
                                interactive=True,
                                allow_custom_value=True,  # pre-load typing/REST
                            )
                            load_btn = gr.Button(
                                "Generate (生成)",
                                variant="primary",
                            )

                        with gr.Column(scale=3):
                            load_audio = gr.Audio(label="Output Audio (合成结果)")
                            load_status = gr.Textbox(label="Status (状态)", lines=3)

                # ============ ② Preset Speakers ============================
                with gr.Tab("② Preset Speakers (预设音色)"), gr.Row():
                    with gr.Column(scale=2):
                        cv_text = gr.Textbox(
                            label="Text (待合成文本)",
                            lines=4,
                            placeholder="Enter text to synthesize (输入要合成的文本).",
                        )
                        with gr.Row():
                            cv_lang = gr.Dropdown(
                                label="Language (语种)",
                                choices=["Auto"],
                                value="Auto",
                                interactive=True,
                                allow_custom_value=True,  # pre-load typing/REST
                            )
                            cv_spk = gr.Dropdown(
                                label="Speaker (说话人)",
                                choices=["Vivian"],
                                value="Vivian",
                                interactive=True,
                                allow_custom_value=True,  # pre-load typing/REST
                            )
                        cv_instruct = gr.Textbox(
                            label="Instruction (Optional) (控制指令，可不输入)",
                            lines=2,
                            placeholder="e.g. Say it in a very angry tone (例如：用特别伤心的语气说).",
                        )
                        btn_cv = gr.Button(
                            "Generate (生成)",
                            variant="primary",
                        )
                    with gr.Column(scale=3):
                        cv_audio = gr.Audio(label="Output Audio (合成结果)")
                        cv_status = gr.Textbox(label="Status (状态)", lines=2)

                # ============ ③ Voice Design ===============================
                with gr.Tab("③ Voice Design (音色设计)"), gr.Row():
                    with gr.Column(scale=2):
                        vd_text = gr.Textbox(
                            label="Text (待合成文本)",
                            lines=4,
                            value=(
                                "It's in the top drawer... wait, it's empty? "
                                "No way, that's impossible! I'm sure I put it there!"
                            ),
                        )
                        vd_lang = gr.Dropdown(
                            label="Language (语种)",
                            choices=["Auto"],
                            value="Auto",
                            interactive=True,
                            allow_custom_value=True,  # pre-load typing/REST
                        )
                        vd_instruct = gr.Textbox(
                            label="Voice Design Instruction (音色描述)",
                            lines=3,
                            value=(
                                "Speak in an incredulous tone, but with a hint of "
                                "panic beginning to creep into your voice."
                            ),
                        )
                        btn_vd = gr.Button(
                            "Generate (生成)",
                            variant="primary",
                        )
                    with gr.Column(scale=3):
                        vd_audio = gr.Audio(label="Output Audio (合成结果)")
                        vd_status = gr.Textbox(label="Status (状态)", lines=2)

                # ============ ④ Codec ======================================
                with gr.Tab("④ Codec (编解码器)"):
                    gr.Markdown(
                        "Roundtrip arbitrary audio through the official 12Hz speech "
                        "tokenizer: encode -> rate/steps metadata -> decode playback.\n"
                        "(任意外部音频经官方 12Hz 语音分词器编解码往返：先展示码率信息，再试听还原音频)"
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            codec_input = gr.Audio(
                                label="Codec Input Audio (输入音频，支持上传/麦克风)",
                                sources=["upload", "microphone"],
                                type="numpy",
                            )
                            btn_codec = gr.Button(
                                "Encode & Decode (编解码往返)",
                                variant="primary",
                            )
                        with gr.Column(scale=3):
                            codec_meta = gr.Textbox(
                                label="Codec Metadata (码率信息)", lines=6, interactive=False
                            )
                            codec_audio = gr.Audio(label="Decoded Audio (解码还原)")
                            codec_file = gr.File(label="Download WAV (下载)")

                # ============ ⑤ History ====================================
                with gr.Tab("⑤ History (合成历史)"):
                    history_table = gr.Dataframe(
                        headers=["ID", "Label (来源)", "Seconds (时长)", "When"],
                        datatype=["number", "str", "str", "str"],
                        value=cb["history_refresh"](),
                        interactive=False,
                        label="Generation History (当前会话的合成记录，最新在前)",
                    )
                    selected_id = gr.State(None)
                    with gr.Row():
                        with gr.Column():
                            hist_audio = gr.Audio(label="Preview (试听)")
                        with gr.Column():
                            hist_download = gr.File(label="Download WAV (下载 wav)")
                    del_btn = gr.Button("Delete Selected (删除所选)", variant="stop")

        # ---------------- event wiring ---------------------------------------
        def _choice_updates(langs, spks):
            lang_update = gr.update(choices=langs, value=langs[0]) if langs else gr.update()
            spk_update = (
                gr.update(choices=spks, value="Vivian" if "Vivian" in spks else spks[0])
                if spks
                else gr.update()
            )
            return lang_update, spk_update

        def _sync_full(alias):
            """Both dropdown refreshes for the Preset Speakers tab."""
            langs, spks = cb["refresh_choices"](alias)
            if langs is None and spks is None:
                return gr.update(), gr.update()
            return _choice_updates(langs, spks)

        def _sync_lang_only(alias):
            """Language-only refresh (Voice Design / Clone tabs)."""
            langs, _spks = cb["refresh_choices"](alias)
            if langs is None:
                return gr.update()
            return _choice_updates(langs, None)[0]

        def _post_generation_chain(click_event, *, sync_lang, sync_speakers):
            """Common tails: side-status + VRAM, fresh choices, history rows."""
            click_event.then(cb["switch_model"], inputs=[model_radio], outputs=[model_status])
            if sync_lang is not None:
                click_event.then(_sync_lang_only, inputs=[model_radio], outputs=[sync_lang])
            if sync_speakers:
                click_event.then(_sync_full, inputs=[model_radio], outputs=[cv_lang, cv_spk])
            click_event.then(lambda: cb["history_refresh"](), None, outputs=[history_table])

        def _history_select(evt: gr.SelectData):
            hid = cb["history_id_at"](evt.index[0] if evt.index else evt.index)
            played = downloaded = None
            if hid is not None:
                try:
                    played = cb["history_play"](hid)
                    downloaded = cb["history_download"](hid)
                except Exception:  # noqa: BLE001 - evicted mid-click
                    played = downloaded = None
            return {"id": hid}, played, downloaded

        model_radio.change(cb["switch_model"], inputs=[model_radio], outputs=[model_status]).then(
            _sync_full, inputs=[model_radio], outputs=[cv_lang, cv_spk]
        )
        demo.load(cb["status_line"], inputs=[model_radio], outputs=[model_status])

        kwargs_inputs = [*kw_fields]

        # ---- ② custom voice -------------------------------------------------
        def _gen_cv(radio_v, text_v, lang_v, spk_v, instruct_v, *_raw_kw):
            return cb["custom_voice"](
                radio_v, text_v, lang_v, spk_v, instruct_v, cb["build_gen_kwargs"](*_raw_kw)
            )

        _post_generation_chain(
            btn_cv.click(
                _gen_cv,
                inputs=[model_radio, cv_text, cv_lang, cv_spk, cv_instruct, *kwargs_inputs],
                outputs=[cv_audio, cv_status],
                api_name="generate_custom_voice",
            ),
            sync_lang=None,
            sync_speakers=True,
        )

        # ---- ③ voice design -------------------------------------------------
        def _gen_vd(radio_v, text_v, lang_v, instruct_v, *_raw_kw):
            return cb["voice_design"](radio_v, text_v, lang_v, instruct_v, cb["build_gen_kwargs"](*_raw_kw))

        _post_generation_chain(
            btn_vd.click(
                _gen_vd,
                inputs=[model_radio, vd_text, vd_lang, vd_instruct, *kwargs_inputs],
                outputs=[vd_audio, vd_status],
                api_name="generate_voice_design",
            ),
            sync_lang=vd_lang,
            sync_speakers=False,
        )

        # ---- ① clone & generate ---------------------------------------------
        def _gen_clone(radio_v, target_v, lang_v, ref_aud_v, ref_txt_v, xvec_v, *_raw_kw):
            return cb["voice_clone"](
                radio_v, target_v, lang_v, ref_aud_v, ref_txt_v, xvec_v, cb["build_gen_kwargs"](*_raw_kw)
            )

        _post_generation_chain(
            btn_clone.click(
                _gen_clone,
                inputs=[
                    model_radio,
                    clone_target,
                    clone_lang,
                    ref_audio,
                    ref_text,
                    xvec_only,
                    *kwargs_inputs,
                ],
                outputs=[clone_audio, clone_status],
                api_name="generate_voice_clone",
            ),
            sync_lang=clone_lang,
            sync_speakers=False,
        )

        # ---- ① save / load voice --------------------------------------------
        save_btn.click(
            cb["save_voice"],
            inputs=[model_radio, ref_audio_s, ref_text_s, xvec_only_s],
            outputs=[prompt_file_out, load_status],
            api_name="save_voice",
        )

        def _gen_loaded(radio_v, file_v, target_v, lang_v, *_raw_kw):
            return cb["load_voice_gen"](radio_v, file_v, target_v, lang_v, cb["build_gen_kwargs"](*_raw_kw))

        _post_generation_chain(
            load_btn.click(
                _gen_loaded,
                inputs=[model_radio, prompt_file_in, load_target, load_lang, *kwargs_inputs],
                outputs=[load_audio, load_status],
                api_name="load_voice_generate",
            ),
            sync_lang=load_lang,
            sync_speakers=False,
        )

        # ---- ④ codec ----------------------------------------------------------
        btn_codec.click(
            cb["codec"],
            inputs=[codec_input],
            outputs=[codec_audio, codec_meta, codec_file],
            api_name="codec_roundtrip",
        )

        # ---- ⑤ history interactions -------------------------------------------
        history_table.select(_history_select, inputs=None, outputs=[selected_id, hist_audio, hist_download])
        del_btn.click(
            lambda hid: cb["history_delete"](hid) if hid is not None else cb["history_refresh"](),
            inputs=[selected_id],
            outputs=[history_table],
        ).then(lambda: (None, None), None, outputs=[hist_audio, hist_download])

        # ---------------- footer -----------------------------------------------
        gr.Markdown(CONCURRENCY_NOTE)
        gr.Markdown(DISCLAIMER)

    return demo
