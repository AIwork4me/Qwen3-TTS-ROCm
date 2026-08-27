"""Headless Gradio-demo backend: business logic without a browser (无界面演示后端).

This layer is what :mod:`qwen3_tts_rocm.cli_demo` (and later the HTML/Gradio
thin-wrappers) call; every decision lives here so it stays unit-testable with
:class:`~qwen3_tts_rocm.testing.FakeTTSModel` injected through factory hooks --
no weights, no GPU, no browser needed.

Design contract
---------------
* **Official objects only**: the service never wraps generation results beyond
  normalising them into ``(sample_rate:int, waveform: np.ndarray[float32])``
  tuples; everything it calls on a model/tokenizer is the published official
  surface (verified in Tasks 10-14).
* **Latency guardrail (latency guardrail)**: degenerate long generation measured
  **23 minutes** on this gfx1151 host at the official default
  ``max_new_tokens=2048`` (Task 12 evidence), so every synthesis method ships
  ``gen_kwargs`` pre-filled with ``{"max_new_tokens": 512}``; caller values win
  (see :func:`_normalize_gen_kwargs`).
* **Validation before model touch**: empty text / missing speaker / missing
  reference audio etc. raise ``ValueError`` *before* any model load or call,
  mirroring the official ``qwen_tts/cli/demo.py`` handlers verbatim (same
  bilingual strings, same check order -- text first, then speaker/design/
  reference audio, then reference text).
* **LRU of size ONE**: exactly one TTS model stays loaded; loading another
  alias unloads the previous one (:func:`qwen3_tts_rocm.loader.unload`), so
  unified-memory APUs never juggle two models.  The speech tokenizer is NOT a
  ``Qwen3TTSModel``: it is constructed lazily once and cached separately from
  the model LRU.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np

from .. import env, loader, models

__all__ = [
    "DEFAULT_GEN_KWARGS",
    "HistoryStore",
    "SynthesisService",
    "display_map",
    "format_error",
    "lookup_display",
]


# ---------------------------------------------------------------------------
# Latency guardrail defaults
# ---------------------------------------------------------------------------

#: Sampling kwargs merged under every synthesis call unless overridden.  512
#: caps pathological generations (~23 min measured at the official 2048 cap on
#: gfx1151); a caller passing its own ``max_new_tokens`` always wins.
DEFAULT_GEN_KWARGS: dict[str, Any] = {"max_new_tokens": 512}


def _normalize_gen_kwargs(user_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge :data:`DEFAULT_GEN_KWARGS` under the caller's overrides.

    ``None``-valued entries are dropped (the official CLI convention: an unset
    flag is absent, never ``None``), and explicit user values win over the
    guardrail default::

        _normalize_gen_kwargs({"max_new_tokens": 2048})  # -> {"max_new_tokens": 2048}
        _normalize_gen_kwargs(None)                      # -> {"max_new_tokens": 512}
    """
    merged = dict(DEFAULT_GEN_KWARGS)
    if user_kwargs:
        merged.update({k: v for k, v in dict(user_kwargs).items() if v is not None})
    return merged


# ---------------------------------------------------------------------------
# Choice display helpers (verbatim behaviour of the official qwen_tts demo)
# ---------------------------------------------------------------------------


def _title_case_display(value: str) -> str:
    """Official title-case display string: underscores become spaces."""
    text = (value or "").strip().replace("_", " ")
    return " ".join(word[:1].upper() + word[1:] if word else "" for word in text.split())


def display_map(values: list[str] | tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """Official demo convention: lowercase/model ids -> pretty display choices.

    Returns ``(display_list, mapping)`` where ``mapping`` goes back from each
    display string to the ORIGINAL raw value (e.g. ``"Ono Anna" -> "ono_anna"``,
    ``"Chinese" -> "chinese"``).  Empty input yields ``([], {})`` like the
    upstream ``_build_choices_and_map``.
    """
    if not values:
        return [], {}
    display = [_title_case_display(v) for v in values]
    mapping = {d: r for d, r in zip(display, values)}
    return display, mapping


def lookup_display(mapping: Mapping[str, str], display_value: str | None,
                   default: str) -> str:
    """Resolve a UI display string back to its raw value, case-insensitively.

    The official languages come back LOWERCASE from ``get_supported_languages``
    while users pick/display "Chinese"; an exact hit wins, otherwise keys are
    matched case-insensitively (then Unicode-casefolded), else *default*.
    """
    if display_value is None:
        return default
    key = str(display_value).strip()
    if not key:
        return default
    if key in mapping:
        return mapping[key]
    folded = {k.casefold(): v for k, v in mapping.items()}
    return folded.get(key.casefold(), default)


def format_error(exc: BaseException) -> str:
    """Bilingual one-line status string for the demo UI's error box."""
    tail = "（请检查输入或查看终端日志 / check input or see terminal log）"
    return f"{type(exc).__name__}: {exc}{tail}"


# ---------------------------------------------------------------------------
# Official-demo validation strings (ground truth: qwen_tts/cli/demo.py)
# ---------------------------------------------------------------------------

_TEXT_REQUIRED = "Text is required (必须填写文本)."
_SPEAKER_REQUIRED = "Speaker is required (必须选择说话人)."
_DESIGN_REQUIRED = "Voice design instruction is required (必须填写音色描述)."
_TARGET_TEXT_REQUIRED = "Target text is required (必须填写待合成文本)."
_REF_AUDIO_REQUIRED = "Reference audio is required (必须上传参考音频)."
_REF_TEXT_REQUIRED = (
    "Reference text is required when use x-vector only is NOT enabled.\n"
    "(未勾选 use x-vector only 时，必须提供参考音频文本；否则请勾选 "
    "use x-vector only，但效果会变差.)"
)
_ITEMS_EMPTY = "Empty voice items (音色为空)."
_VOICE_FILE_INVALID = "Invalid file format (文件格式不正确)."
_ITEM_FORMAT_INVALID = "Invalid item format in file (文件内部格式错误)."
_SPK_MISSING = "Missing ref_spk_embedding (缺少说话人向量)."

_CODEC_PAIR_REQUIRED = (
    "Audio input must be a (sr, wav) pair (音频输入必须是 (采样率, 波形) 二元组)"
)


def _require_filled(value: object, error: str) -> str:
    """Raise *error* (ValueError) unless *value* is a non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    return value.strip()


# ---------------------------------------------------------------------------
# SynthesisService
# ---------------------------------------------------------------------------


class SynthesisService:
    """Headless synthesis facade over ONE lazily-loaded official TTS model.

    Args:
        factory: ``alias -> model`` callable; defaults to loading through
            :func:`qwen3_tts_rocm.loader.load` (ROCm smart defaults).  Tests
            inject e.g. a cached-:class:`FakeTTSModel` factory instead.
        tokenizer_factory: zero-arg callable building the speech tokenizer used
            by :meth:`codec_roundtrip`; defaults to the official
            ``Qwen3TTSTokenizer.from_pretrained(models.local_dir("tokenizer"),
            device_map=pick_device(), dtype=bfloat16)`` built lazily ONCE.
    """

    def __init__(self, factory=None, tokenizer_factory=None) -> None:
        self._factory = factory if factory is not None else loader.load
        self._tokenizer_factory = (
            tokenizer_factory if tokenizer_factory is not None
            else self._default_tokenizer_factory
        )
        # LRU slot of size ONE: alias -> officially loaded model object.
        self._cache: dict[str, Any] = {}
        #: Alias whose model is currently loaded (``None`` when unloaded).
        self.current_alias: str | None = None
        # Speech tokenizer: a separate lazy singleton, outside the model LRU.
        self._tokenizer: Any = None
        #: Shared generation history for the UI layer (合成历史, wavs live here
        #: ONLY -- the UI never keeps its own copy of a waveform on disk).
        self.history = HistoryStore()

    # -- model LRU ----------------------------------------------------------

    @staticmethod
    def _default_tokenizer_factory():
        """Build the official 12Hz speech tokenizer with ROCm-smart settings.

        Imported lazily: merely creating a :class:`SynthesisService` (and the
        whole import path around it) must stay free of multi-GB deps so tests
        can inject doubles first.
        """
        import torch  # lazy: ambient ROCm wheel
        from qwen_tts import Qwen3TTSTokenizer

        return Qwen3TTSTokenizer.from_pretrained(
            str(models.local_dir("tokenizer")),
            device_map=env.pick_device("auto"),
            dtype=torch.bfloat16,
        )

    def get(self, alias: str):
        """Return the loaded model for *alias*, unloading any previous one.

        Loading happens lazily through *factory* on first touch; requesting a
        different alias evicts (via :func:`loader.unload`) and replaces the
        single cached entry, keeping memory flat on unified-memory APUs.
        """
        alias = str(alias)
        if alias in self._cache:
            self.current_alias = alias
            return self._cache[alias]
        self.unload_all()
        model = self._factory(alias)
        self._cache[alias] = model
        self.current_alias = alias
        return model

    def unload_all(self) -> None:
        """Best-effort teardown of every cached model (尽力释放，绝不抛错)."""
        for model in self._cache.values():
            loader.unload(model)
        self._cache.clear()
        self.current_alias = None

    # -- choice plumbing ------------------------------------------------------

    def _language_choices(self, model) -> dict[str, str]:
        """Display->raw language map from the OFFICIAL lowercase getter."""
        _, lang_map = display_map(list(model.get_supported_languages()))
        return lang_map

    def _speaker_choices(self, model) -> dict[str, str]:
        _, spk_map = display_map(list(model.get_supported_speakers()))
        return spk_map

    # -- synthesis ------------------------------------------------------------

    def custom_voice(self, alias: str, text: str, language_display: str = "Auto",
                     speaker_display: str = "Vivian", instruct: str | None = None,
                     gen_kwargs: Mapping[str, Any] | None = None,
                    ) -> tuple[int, np.ndarray]:
        """Custom-voice generation; validates like the official demo UI."""
        stripped = _require_filled(text, _TEXT_REQUIRED)
        _require_filled(speaker_display, _SPEAKER_REQUIRED)

        model = self.get(alias)
        language = lookup_display(self._language_choices(model),
                                  language_display, "Auto")
        speaker = lookup_display(self._speaker_choices(model),
                                 speaker_display, speaker_display.strip())
        instruction = ((instruct or "").strip() or None)  # official blank rule
        wavs, sr = model.generate_custom_voice(
            text=stripped,
            language=language,
            speaker=speaker,
            instruct=instruction,
            **_normalize_gen_kwargs(gen_kwargs),
        )
        return int(sr), _as_float32(wavs[0])

    def voice_design(self, alias: str, text: str, language_display: str = "Auto",
                     instruct: str | None = None,
                     gen_kwargs: Mapping[str, Any] | None = None,
                    ) -> tuple[int, np.ndarray]:
        """Natural-language voice-design generation (instruction mandatory)."""
        stripped = _require_filled(text, _TEXT_REQUIRED)
        design = _require_filled(instruct, _DESIGN_REQUIRED)

        model = self.get(alias)
        language = lookup_display(self._language_choices(model),
                                  language_display, "Auto")
        wavs, sr = model.generate_voice_design(
            text=stripped,
            language=language,
            instruct=design,
            **_normalize_gen_kwargs(gen_kwargs),
        )
        return int(sr), _as_float32(wavs[0])

    def voice_clone(self, alias: str, text: str, language_display: str = "Auto",
                    ref_audio: tuple[int, Any] | None = None,
                    ref_text: str | None = None, xvec_only: bool = False,
                    gen_kwargs: Mapping[str, Any] | None = None,
                   ) -> tuple[int, np.ndarray]:
        """Zero-shot cloning from a Gradio-style ``(sr, wav)`` reference clip.

        Validation order mirrors ``run_voice_clone``: target text, then
        reference audio (a usable ``(sr, wav)`` pair is mandatory here), then
        reference text unless ``xvec_only``.  The reference reaches the model
        in the OFFICIAL ``(wav, sr)`` argument order.
        """
        stripped = _require_filled(text, _TARGET_TEXT_REQUIRED)
        # OFFICIAL (wav, sr) argument order regardless of input tuple order.
        ref_wav, ref_sr = _coerce_ref_pair(ref_audio, _REF_AUDIO_REQUIRED)
        cleaned_ref_text = (ref_text or "").strip() or None
        if not xvec_only and not cleaned_ref_text:
            raise ValueError(_REF_TEXT_REQUIRED)

        model = self.get(alias)
        language = lookup_display(self._language_choices(model),
                                  language_display, "Auto")
        wavs, sr = model.generate_voice_clone(
            text=stripped,
            language=language,
            ref_audio=(ref_wav, ref_sr),
            ref_text=cleaned_ref_text,
            x_vector_only_mode=bool(xvec_only),
            **_normalize_gen_kwargs(gen_kwargs),
        )
        return int(sr), _as_float32(wavs[0])

    def clone_prompt_from_ref(self, alias: str,
                              ref_audio: tuple[int, Any] | None = None,
                              ref_text: str | None = None,
                              xvec_only: bool = False) -> list:
        """Reusable clone prompt items (official ``create_voice_clone_prompt``)."""
        ref_wav, ref_sr = _coerce_ref_pair(ref_audio, _REF_AUDIO_REQUIRED)
        cleaned_ref_text = (ref_text or "").strip() or None
        if not xvec_only and not cleaned_ref_text:
            raise ValueError(_REF_TEXT_REQUIRED)

        model = self.get(alias)
        return model.create_voice_clone_prompt(
            ref_audio=(ref_wav, ref_sr),
            ref_text=cleaned_ref_text,
            x_vector_only_mode=bool(xvec_only),
        )

    def voice_clone_with_prompt(self, alias: str, text: str,
                                language_display: str = "Auto",
                                items: list | None = None,
                                gen_kwargs: Mapping[str, Any] | None = None,
                               ) -> tuple[int, np.ndarray]:
        """Clone using previously-built prompt items (save/load voice tab)."""
        if not items:
            raise ValueError(_ITEMS_EMPTY)
        stripped = _require_filled(text, _TARGET_TEXT_REQUIRED)

        model = self.get(alias)
        language = lookup_display(self._language_choices(model),
                                  language_display, "Auto")
        wavs, sr = model.generate_voice_clone(
            text=stripped,
            language=language,
            voice_clone_prompt=items,               # official passes the list itself
            **_normalize_gen_kwargs(gen_kwargs),
        )
        return int(sr), _as_float32(wavs[0])

    def load_voice_file(self, path: str | Any) -> list:
        """Reconstruct clone-prompt items from an official .pt voice file.

        Port of ``qwen_tts/cli/demo.py::load_prompt_and_gen``'s reconstruction
        block VERBATIM (same torch.load options, same field defaults -- note
        ``icl_mode`` falls back to "not x-vector-only" exactly like upstream) --
        the UI layer reuses THIS seam instead of duplicating it.  Errors carry
        the official bilingual strings so the status box reads like the
        official demo on a malformed file.
        """
        import torch  # lazy: ambient wheel from earlier tasks
        from qwen_tts import VoiceClonePromptItem

        path = getattr(path, "name", None) or getattr(path, "path", None) or str(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "items" not in payload:
            raise ValueError(_VOICE_FILE_INVALID)

        items_raw = payload["items"]
        if not isinstance(items_raw, list) or len(items_raw) == 0:
            raise ValueError(_ITEMS_EMPTY)

        items: list = []
        for d in items_raw:
            if not isinstance(d, dict):
                # ValueError by OFFICIAL PARITY: qwen_tts/cli/demo.py's
                # load_prompt_and_gen raises ValueError on malformed items.
                raise ValueError(_ITEM_FORMAT_INVALID)  # noqa: TRY004
            ref_code = d.get("ref_code", None)
            if ref_code is not None and not torch.is_tensor(ref_code):
                ref_code = torch.tensor(ref_code)
            ref_spk = d.get("ref_spk_embedding", None)
            if ref_spk is None:
                raise ValueError(_SPK_MISSING)
            if not torch.is_tensor(ref_spk):
                ref_spk = torch.tensor(ref_spk)
            items.append(
                VoiceClonePromptItem(
                    ref_code=ref_code,
                    ref_spk_embedding=ref_spk,
                    x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                    icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                    ref_text=d.get("ref_text", None),
                )
            )
        return items

    # -- codec roundtrip --------------------------------------------------------

    def codec_roundtrip(self, wav_tuple: tuple[int, Any]) -> tuple[int, np.ndarray, dict]:
        """Encode -> decode one clip through the official speech tokenizer.

        *wav_tuple* is a ``(sr, wav)`` pair (Gradio Audio component shape).
        The tokenizer builds lazily once (cached independently of the model
        LRU).  Returns ``(output_sr, decoded_waveform[float32], meta)``; *meta*
        carries ``model_type`` plus the four rate getters and, best-effort, the
        ``codes_shape`` extracted duck-typed from the encoder output (mirrors
        the Task 14 approach while staying inside this package).
        """
        # _coerce_pair returns OFFICIAL (wav, sr) order regardless of input shape.
        wav_in, sr_in = _coerce_pair(wav_tuple, _CODEC_PAIR_REQUIRED)
        tokenizer = self._get_tokenizer()

        encoded = tokenizer.encode(wav_in, sr=int(sr_in))
        decoded = tokenizer.decode(encoded)

        meta = self._extract_meta(tokenizer, encoded)
        out_sr, audio = _extract_decode(decoded)
        return int(out_sr), np.asarray(audio, dtype=np.float32), meta

    def _get_tokenizer(self):
        """Lazily build (once) and cache the speech tokenizer double/factory."""
        if self._tokenizer is None:
            self._tokenizer = self._tokenizer_factory()
        return self._tokenizer

    @staticmethod
    def _extract_meta(tokenizer, encoded) -> dict[str, Any]:
        """Official metadata getters (Task 14 style); omit anything absent."""
        meta: dict[str, Any] = {}
        getters = (
            ("model_type", "get_model_type"),
            ("input_sample_rate", "get_input_sample_rate"),
            ("output_sample_rate", "get_output_sample_rate"),
            ("encode_downsample", "get_encode_downsample_rate"),
            ("decode_upsample", "get_decode_upsample_rate"),
        )
        for key, getter in getters:
            fn = getattr(tokenizer, getter, None)
            if callable(fn):
                try:
                    meta[key] = fn()
                except Exception:  # noqa: BLE001,S112 - meta is best-effort only
                    continue
        codes = getattr(encoded, "audio_codes", None)
        if codes is not None:
            try:
                first = codes[0] if isinstance(codes, (list, tuple)) else codes
                meta["codes_shape"] = tuple(int(dim) for dim in first.shape)
            except Exception:  # noqa: BLE001,S110 - unknown code geometry: skip
                pass
        return meta


def _as_float32(waveform: Any) -> np.ndarray:
    """Transport a generation result into a plain float32 ndarray (零拷贝当可能).

    Handles the official wrapper's ndarray returns natively; cpu-torch tensors
    are bridged defensively WITHOUT touching their API beyond transport.
    """
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    arr = np.asarray(waveform)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _coerce_pair(pair: Any, error_message: str) -> tuple[np.ndarray, int]:
    """Validate/normalise a ``(sr, wav)`` audio pair; raises ValueError otherwise.

    Returns ``(wav[float32], sr[int])`` with stereo averaged down to mono
    (same channel rule as the official ``_normalize_audio``)."""
    reason = f"unsupported audio pair {type(pair).__name__}"
    sr = -1
    ok = False
    if isinstance(pair, (tuple, list)) and len(pair) == 2:
        raw_sr, raw_wav = pair
        if not isinstance(raw_sr, bool):
            try:
                sr = int(raw_sr)
                reason = f"bad sample rate {raw_sr!r}"
            except (TypeError, ValueError):
                reason = f"non-integer sample rate {raw_sr!r}"
            wav_arr = None if raw_wav is None else np.asarray(raw_wav)
            if sr > 0 and wav_arr is not None and wav_arr.size and wav_arr.ndim >= 1:
                ok = True
            elif wav_arr is None or not wav_arr.size:
                reason = "empty waveform"
    else:
        reason = f"expected a 2-element (sr, wav) tuple, got {type(pair).__name__}"
    if not ok:
        raise ValueError(f"{error_message}; got {reason}")
    if wav_arr.ndim > 1:
        wav_arr = np.mean(wav_arr, axis=-1)
    return _as_float32(wav_arr), sr


def _coerce_ref_pair(pair: Any, official_error: str) -> tuple[np.ndarray, int]:
    """:func:`_coerce_pair`, but EVERY failure reports the official UI string."""
    try:
        return _coerce_pair(pair, official_error)
    except ValueError as exc:
        raise ValueError(f"{official_error}[{exc}]") from exc


def _extract_decode(decoded: Any) -> tuple[int, np.ndarray]:
    """Pull ``(sample_rate, audio)`` from any official decode shape (Task 14 arms)."""
    if isinstance(decoded, tuple) and len(decoded) == 2:
        wavs, sr = decoded
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        return int(sr), np.asarray(audio)

    if isinstance(decoded, dict):
        audio = next((decoded[k] for k in ("audio", "wavs", "audio_values", "wav")
                      if k in decoded), None)
        sr = next((decoded[k] for k in ("sample_rate", "sampling_rate", "sr")
                   if k in decoded), None)
        if audio is not None and sr is not None:
            first = audio[0] if isinstance(audio, (list, tuple)) else audio
            return int(sr), np.asarray(first)
        raise TypeError(f"decode dict lacks audio/sample-rate keys: {sorted(decoded)}")

    get_audio = getattr(decoded, "get_audio", None)
    if callable(get_audio):
        sr = getattr(decoded, "sample_rate", getattr(decoded, "sr", None))
        if sr is not None:
            return int(sr), np.asarray(get_audio())
        raise TypeError("get_audio-style decode object carries no sample rate")

    raise TypeError(f"unrecognized decode surface: {type(decoded).__name__}")


# ---------------------------------------------------------------------------
# HistoryStore
# ---------------------------------------------------------------------------


class HistoryStore:
    """FIFO generation history with a hard 30-entry cap (合成历史记录).

    Thread-safe enough for the demo UI queue: mutations happen under a lock;
    ids are monotonically increasing so they sort chronologically.  Waveforms
    are stored by reference (single-process UI owns them; do not mutate).
    """

    #: Maximum stored clips; adding past it evicts the OLDEST entry first.
    cap = 30

    def __init__(self, cap: int | None = None) -> None:
        self._cap = self.cap if cap is None else max(int(cap), 0)
        self._lock = threading.Lock()
        self._items: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._next_id = 0

    def add(self, label: str, sr: int, wav: Any) -> int:
        """Store one clip; returns its new id, evicting FIFO past the cap."""
        arr = _as_float32(wav)
        entry = {
            "label": str(label),
            "created_at": time.time(),
            "sr": int(sr),
            "wav": arr,
            "duration_s": round(len(arr) / int(sr), 3) if int(sr) > 0 else 0.0,
        }
        with self._lock:
            self._next_id += 1
            history_id = self._next_id
            self._items[history_id] = entry
            while len(self._items) > self._cap:
                self._items.popitem(last=False)          # FIFO eviction
            return history_id

    def list(self) -> list[dict[str, Any]]:
        """Summary dicts (newest first): id/label/created_at/duration_s."""
        with self._lock:
            summaries = [
                {"id": hid, "label": e["label"], "created_at": e["created_at"],
                 "duration_s": e["duration_s"]}
                for hid, e in reversed(list(self._items.items()))
            ]
        return summaries

    def item(self, history_id: int) -> tuple[int, np.ndarray]:
        """Fetch one clip's ``(sr, wav)``; unknown ids raise ValueError."""
        with self._lock:
            if int(history_id) not in self._items:
                raise ValueError(
                    f"unknown history id {history_id} (未知的历史记录编号: {history_id})"
                )
            entry = self._items[int(history_id)]
        return entry["sr"], entry["wav"]

    def remove(self, history_id: int) -> None:
        """Drop one clip; unknown ids raise ValueError ( bilingual diagnostic )."""
        with self._lock:
            if int(history_id) not in self._items:
                raise ValueError(
                    f"unknown history id {history_id} (未知的历史记录编号: {history_id})"
                )
            del self._items[int(history_id)]
