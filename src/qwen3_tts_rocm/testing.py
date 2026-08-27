"""Shared GPU-test toolkit for qwen3-tts-rocm (shipped so OSS users reuse it too).

This module is deliberately part of the *package*, not just the test suite:
every helper here is useful to anyone scripting against a loaded official
model, and Tasks 10-17 of this repository build their smoke/behaviour suites
on top of it.

Public API (fixed contract -- do not rename):

* :func:`make_tone`       -- deterministic clean sinusoid -> ``(wav, sr)``.
* :func:`write_wav`       -- write a float waveform to disk (mkdir + soundfile).
* :func:`assert_wav_sane` -- one-call sanity gate with bilingual diagnostics.
* :class:`FakeTTSModel`   -- instant, dependency-free double of the official
  ``qwen_tts.Qwen3TTSModel`` inference surface, used by pipeline tests that
  must not pay multi-GB weight loads.

Conventions
-----------
The Qwen3-TTS speech tokenizer always decodes at **24000 Hz** (spike evidence:
``models/Qwen3-TTS-Tokenizer-12Hz/config.json`` -> ``output_sample_rate``),
so that is the default sample rate for every helper here and the value
:func:`assert_wav_sane` checks against via ``sr_expected``.

All diagnostics are bilingual (English | Chinese) per project style.

FakeTTSModel determinism scheme
-------------------------------
Audio is synthesised as an amplitude-bounded sine whose frequency and phase are
derived from a blake2b digest over ``(method, sample_index, text, language,
speaker/instruct, x_vector_only_mode, prompt fingerprint, sorted gen_kwargs)``.
blake2b is stable across interpreter processes (unlike builtin ``hash()``), so
the same invocation yields byte-identical audio today and tomorrow, while ANY
change to text / instruction / speaker / sampling kwargs reliably produces a
different array.  Length is constant (constructor ``seconds``), which keeps
arrays element-wise comparable across calls -- downstream "two instructions
differ" assertions subtract waveforms directly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FakeTTSModel", "assert_wav_sane", "make_tone", "write_wav"]

#: Qwen3-TTS tokenizer decode rate; every generated waveform ships at this Hz.
_SAMPLE_RATE = 24000

#: Hard clipping ceiling accepted by :func:`assert_wav_sane` (loose vs 1.0).
_PEAK_LIMIT = 1.5

#: The nine speaker ids shipped in the official CustomVoice config
#  (``talker_config.spk_id``, verified on-disk); ``get_supported_speakers``
#  returns them sorted/lowercase exactly like the official wrapper.
_DEFAULT_SPEAKERS = (
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
)

#: Official languages (``talker_config.codec_language_id`` minus ``*_dialect``)
#: plus the implicit ``auto``; validation is case-insensitive like the official
#: implementation ("Chinese" == "chinese").
_DEFAULT_LANGUAGES = (
    "auto", "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
)


def _bilingual(en: str, zh: str) -> AssertionError:
    """Build an AssertionError carrying both messages (双语诊断信息)."""
    return AssertionError(f"{en} ({zh})")


# ---------------------------------------------------------------------------
# Waveform helpers
# ---------------------------------------------------------------------------


def make_tone(seconds: float = 1.0, freq: float = 220.0,
              sr: float = 24000) -> tuple[np.ndarray, int]:
    """Return ``(wav, sr)`` for a short clean test tone.

    A single sine at *freq* Hz with 0.8 amplitude and tiny edge fades (click
    suppression when written to disk).  float32 output, length
    ``round(seconds * sr)`` >= 1 sample.

    ``freq`` must be finite and > 0: a zero (or negative) frequency yields an
    all-zero waveform that would only trip :func:`assert_wav_sane`'s silence
    gate later, so it is rejected at the source with a bilingual ValueError.
    """
    seconds = float(seconds)
    freq = float(freq)
    sample_rate = int(sr)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"seconds must be finite and > 0, got {seconds!r} (时长必须为正数)")
    if not (sample_rate > 0):
        raise ValueError(f"sr must be > 0, got {sr!r} (采样率必须为正数)")
    if not np.isfinite(freq) or freq <= 0:
        raise ValueError(f"freq must be > 0 and finite, got {freq!r} (频率必须为正数)")

    n = max(round(seconds * sample_rate), 1)
    t = np.arange(n, dtype=np.float64) / sample_rate
    wav = 0.8 * np.sin(2.0 * np.pi * freq * t)

    fade = min(64, n // 10)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        wav[:fade] *= ramp
        wav[-fade:] *= ramp[::-1]
    return wav.astype(np.float32), sample_rate


def write_wav(path: str | Path, wav: Iterable[float], sr: int) -> Path:
    """Write *wav* to *path* as WAV (parent dirs created as needed).

    Uses ``soundfile`` (already installed alongside the official runtime); the
    path is returned so calls chain nicely inside tmp_path-based fixtures.
    """
    try:
        import soundfile as sf
    except Exception as exc:  # actionable instead of a bare trace
        raise RuntimeError(
            "soundfile is required to write wav files "
            "(写出 wav 需要 soundfile): pip install soundfile"
        ) from exc

    target = Path(path)
    if str(target.parent) not in ("", ".") and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), np.asarray(wav), int(sr))
    return target


def assert_wav_sane(wav: Any, sr_expected: int | None = None,
                    min_energy_rms: float | None = 1e-4) -> bool:
    """Assert a waveform passes TTS-output sanity checks; AssertionError otherwise.

    Rules (each violation raises with a bilingual message):
      * input converts to an ndarray whose ORIGINAL dtype kind is ``f``
        (casting to float happens internally for the math only -- an int16
        buffer from a wav reader must be cast explicitly by the caller);
      * non-empty and entirely finite (no NaN / +-Inf);
      * abs peak <= 1.5 (loose clipping guard around the nominal [-1, 1] range);
      * rms > ``min_energy_rms`` when that threshold is > 0 (catches silence);
      * when given, ``sr_expected`` must equal the Qwen3-TTS tokenizer decode
        rate 24000 -- call it as
        ``wavs, sr = model.generate(...); assert_wav_sane(wavs[0], sr)``
        so any pipeline emitting at the wrong rate fails loudly right here.

    Returns True on success (handy inside expression-style asserts).
    """
    try:
        arr = np.asarray(wav)
    except Exception as exc:  # report coercion failures bilingually
        raise _bilingual(
            f"wav must be array-like, got unconvertible object {type(wav).__name__}: {exc!r}",
            f"波形无法转换为数组（类型 {type(wav).__name__}）",
        ) from exc

    # int-dtype audio buffers fail here BY DESIGN: silence-vs-value bugs hide in
    # integer scaling; callers cast explicitly (wav.astype(np.float32)) first.
    if arr.dtype.kind != "f":
        raise _bilingual(
            f"wav must have a floating-point dtype before sanity checks, got "
            f"dtype {arr.dtype}; cast explicitly at the source",
            f"波形必须是浮点数组才能做健全性检查，实际为 {arr.dtype}，请先显式转换类型",
        )

    flat = arr.ravel().astype(np.float64, copy=False)
    if flat.size == 0:
        raise _bilingual("wav is empty (zero samples)", "波形为空数组（样本数为 0）")

    n_bad = int((~np.isfinite(flat)).sum())
    if n_bad:
        raise _bilingual(
            f"wav contains {n_bad} non-finite sample(s) (NaN or Inf)",
            f"波形包含 {n_bad} 个非有限值（NaN 或 Inf）",
        )

    peak = float(np.abs(flat).max())
    if not peak <= _PEAK_LIMIT:
        raise _bilingual(
            f"clipped waveform: abs peak {peak:.6g} exceeds {_PEAK_LIMIT}",
            f"波形削波：绝对峰值 {peak:.6g} 超过上限 {_PEAK_LIMIT}",
        )

    if min_energy_rms is not None and min_energy_rms > 0:
        rms = float(np.sqrt(np.mean(np.square(flat))))
        if not rms > min_energy_rms:
            raise _bilingual(
                f"wav looks silent: rms {rms:.6g} <= min_energy_rms "
                f"{min_energy_rms:.6g} (generation produced no audible energy)",
                f"波形疑似静音：RMS 能量 {rms:.6g} 未超过阈值 {min_energy_rms:.6g}"
                "（生成结果没有可听能量）",
            )

    if sr_expected is not None and int(sr_expected) != _SAMPLE_RATE:
        raise _bilingual(
            f"sample rate mismatch: got sr={sr_expected}, but the Qwen3-TTS "
            f"tokenizer decode path always outputs {_SAMPLE_RATE} Hz; check "
            "where this waveform came from",
            f"采样率不匹配：实际 {sr_expected} Hz，而 Qwen3-TTS 分词器解码固定输出 "
            f"{_SAMPLE_RATE} Hz，请排查该波形的来源",
        )
    return True


# ---------------------------------------------------------------------------
# FakeTTSModel: official-API-compatible test double (零依赖、可复现)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceClonePromptItem:
    """Mirror of the official prompt item's observable fields."""

    ref_code: Any                      # int codec codes (fake: synthetic ints)
    ref_spk_embedding: Any             # speaker embedding vector (fake: float32)
    x_vector_only_mode: bool           # True -> embedding-only clone mode
    icl_mode: bool                     # True -> ref_text/ref_code conditioning


class FakeTTSModel:
    """Minimal, deterministic stand-in for the official Qwen3TTSModel.

    Mirrors the shapes of the four official inference entry points used by the
    suites of Tasks 10-17 (keyword-callable exactly like the real thing), plus
    the two metadata getters.  Audio generation costs microseconds and no GPU;
    every invocation is recorded into :attr:`calls` for spying::

        fake = FakeTTSModel()
        wavs, sr = fake.generate_custom_voice(text="Hi", language="English",
                                              speaker="Ryan")
        assert fake.calls[0]["speaker"] == ["Ryan"]
        assert_wav_sane(wavs[0], sr)

    Determinism/differentiation contract: identical kwargs -> byte-identical
    output (across calls AND across instances AND processes); changing ANY of
    text/language/speaker/instruct/gen-kwargs changes the waveform.  See the
    module docstring for the seeding scheme.
    """

    #: Public nested alias so consumers can spell ``FakeTTSModel.VoiceClonePromptItem``.
    VoiceClonePromptItem = VoiceClonePromptItem

    def __init__(self, sr: int = 24000,
                 speakers: Iterable[str] | None = None,
                 languages: Iterable[str] | None = None,
                 seconds: float = 0.25) -> None:
        self.sr = int(sr)
        self.seconds = float(seconds)
        self.speakers = tuple(_DEFAULT_SPEAKERS if speakers is None else speakers)
        self.languages = tuple(_DEFAULT_LANGUAGES if languages is None else languages)
        #: Recorded invocations: list of dicts with a "method" key plus the
        #: normalised per-sample arguments and merged gen_kwargs, oldest first.
        self.calls: list[dict[str, Any]] = []

    # -- official metadata surface ------------------------------------------

    def get_supported_speakers(self) -> list[str]:
        """Sorted lowercase speaker names, like the official wrapper."""
        return sorted(self.speakers)

    def get_supported_languages(self) -> list[str]:
        """Sorted lowercase language names ('auto' included), like official."""
        return sorted(self.languages)

    # -- official generate surface -------------------------------------------

    def generate_custom_voice(self, text, language: str = "Auto",
                              speaker=None, instruct=None, **gen_kwargs):
        """Official shape: ``(list[np.ndarray], sr)``, one wav per text item."""
        texts = self._ensure_list(text)
        languages = self._broadcast(language, len(texts))
        # Recording keeps "no instruction"/"omitted speaker" as None verbatim,
        # while synthesis needs a per-sample value -> parallel item lists.
        speakers = self._per_sample(speaker, len(texts), "speaker", "说话人")
        speaker_items = (speakers if isinstance(speakers, list)
                         else [None] * len(texts))
        instructs = self._per_sample(instruct, len(texts), "instruct", "指令")
        instruct_items = (instructs if isinstance(instructs, list)
                          else [None] * len(texts))
        self._validate_languages(languages)
        self._validate_speakers([s for s in speaker_items if s is not None])

        wavs, gen_repr = [], self._gen_repr(gen_kwargs)
        for i, txt in enumerate(texts):
            wavs.append(self._synth("generate_custom_voice", i, txt,
                                    languages[i], speaker_items[i],
                                    instruct_items[i], gen_repr))

        self.calls.append({
            "method": "generate_custom_voice",
            "text": texts,
            "language": languages,
            "speaker": speakers,
            "instruct": instructs,
            "gen_kwargs": dict(gen_kwargs),
        })
        return wavs, self.sr

    def generate_voice_design(self, text, language: str = "Auto",
                              instruct=None, **gen_kwargs):
        """Official shape: natural-language style instruction controls timbre."""
        texts = self._ensure_list(text)
        languages = self._broadcast(language, len(texts))
        instructs = self._per_sample(instruct, len(texts), "instruct", "指令")
        instruct_items = (instructs if isinstance(instructs, list)
                          else [None] * len(texts))
        self._validate_languages(languages)

        wavs, gen_repr = [], self._gen_repr(gen_kwargs)
        for i, txt in enumerate(texts):
            wavs.append(self._synth("generate_voice_design", i, txt,
                                    languages[i], None, instruct_items[i],
                                    gen_repr))

        self.calls.append({
            "method": "generate_voice_design",
            "text": texts,
            "language": languages,
            "instruct": instructs,
            "gen_kwargs": dict(gen_kwargs),
        })
        return wavs, self.sr

    def generate_voice_clone(self, text, language: str = "Auto",
                             ref_audio=None, ref_text=None,
                             voice_clone_prompt=None, x_vector_only_mode=False,
                             **gen_kwargs):
        """Official shape: clone via (ref_audio+ref_text[, mode]) OR a prompt list."""
        texts = self._ensure_list(text)
        languages = self._broadcast(language, len(texts))
        self._validate_languages(languages)

        has_prompt = voice_clone_prompt is not None
        if not has_prompt and ref_audio is None:
            raise ValueError(
                "voice cloning needs either voice_clone_prompt or ref_audio "
                "(克隆必须提供 reference 音频或预先构建的 voice_clone_prompt)"
            )
        if not has_prompt and not x_vector_only_mode and not ref_text:
            raise ValueError(
                "ref_text is required unless x_vector_only_mode=True "
                "(ICL 模式下必须提供 ref_text 参考文本)"
            )

        wavs, gen_repr = [], self._gen_repr(gen_kwargs)
        fingerprint = (
            x_vector_only_mode,
            self._prompt_fingerprint(voice_clone_prompt),
            repr(ref_audio),
            repr(ref_text),
        )
        for i, txt in enumerate(texts):
            wavs.append(self._synth("generate_voice_clone", i, txt,
                                    languages[i], None, None, gen_repr,
                                    fingerprint))

        self.calls.append({
            "method": "generate_voice_clone",
            "text": texts,
            "language": languages,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "voice_clone_prompt": voice_clone_prompt,
            "x_vector_only_mode": bool(x_vector_only_mode),
            "gen_kwargs": dict(gen_kwargs),
        })
        return wavs, self.sr

    def create_voice_clone_prompt(self, ref_audio, ref_text=None,
                                  x_vector_only_mode=False):
        """Official shape: build prompt items; requires ref_text in ICL mode.

        Spy fidelity: each item's RNG is seeded over the full input key
        ``(method, audio repr, ref_text, x_vector_only_mode)``, so identical
        audio with a different transcript (or mode) yields *different* items
        while identical invocations stay byte-reproducible.  The
        :attr:`calls` record keeps the RAW ``x_vector_only_mode`` value (a
        list stays a list); the returned items keep the official ``bool()``
        semantics on their ``x_vector_only_mode`` field.
        """
        audios = self._ensure_list(ref_audio)
        texts = ([*(t for t in ref_text)] if isinstance(ref_text, list)
                 else [ref_text] * len(audios))
        flags = ([bool(v) for v in x_vector_only_mode]
                 if isinstance(x_vector_only_mode, (list, tuple))
                 else [bool(x_vector_only_mode)] * len(audios))
        if len(texts) != len(audios) or len(flags) != len(audios):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(audios)}, "
                f"ref_text={len(texts)}, x_vector_only_mode={len(flags)} "
                "(批量大小不一致)"
            )

        items: list[VoiceClonePromptItem] = []
        for audio, txt, xvec in zip(audios, texts, flags):
            if not xvec and not txt:
                raise ValueError(
                    "ref_text is required unless x_vector_only_mode=True "
                    "(ICL 模式下必须提供 ref_text 参考文本)"
                )
            rng = np.random.default_rng(self._seed("create_voice_clone_prompt",
                                                   repr(audio), txt, xvec))
            items.append(VoiceClonePromptItem(
                ref_code=rng.integers(0, 16000, size=24, dtype=np.int64),
                ref_spk_embedding=rng.standard_normal(192).astype(np.float32),
                x_vector_only_mode=xvec,
                icl_mode=not xvec,
            ))

        self.calls.append({
            "method": "create_voice_clone_prompt",
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            # RAW argument preserved for spying (list stays a list); the
            # per-item bool() projection lives on the items themselves.
            "x_vector_only_mode": x_vector_only_mode,
        })
        return items

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _seed(*parts: Any) -> int:
        """Stable cross-process digest -> RNG seed (builtin hash() is NOT stable)."""
        hasher = hashlib.blake2b(digest_size=8)
        for part in parts:
            hasher.update(repr(part).encode("utf-8"))
            hasher.update(b"\x1f")
        return int.from_bytes(hasher.digest(), "big")

    @classmethod
    def _gen_repr(cls, gen_kwargs: dict[str, Any]) -> tuple:
        """Canonical, order-insensitive projection of sampling kwargs."""
        return tuple(sorted((str(k), repr(v)) for k, v in gen_kwargs.items()))

    @staticmethod
    def _prompt_fingerprint(prompt) -> str | None:
        """Stable identity for a prompt (or its items' observable fields)."""
        if prompt is None:
            return None
        items = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        return repr([
            (
                getattr(it, "x_vector_only_mode", None),
                getattr(it, "icl_mode", None),
                repr(getattr(it, "ref_code", None)),
                repr(getattr(it, "ref_spk_embedding", None)),
            )
            for it in items
        ])

    def _synth(self, method: str, index: int, text: Any, language: Any,
               speaker: Any, instruct: Any, gen_repr: tuple,
               *extra: Any) -> np.ndarray:
        """Deterministic bounded sine seeded by the full invocation key."""
        rng = np.random.default_rng(
            self._seed(method, index, text, language, speaker, instruct,
                       gen_repr, extra))
        freq = float(rng.uniform(120.0, 960.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        n = max(round(self.sr * self.seconds), 1)
        t = np.arange(n, dtype=np.float64) / self.sr
        wav = 0.6 * np.sin(2.0 * np.pi * freq * t + phase)
        fade = min(32, n // 10)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
            wav[:fade] *= ramp
            wav[-fade:] *= ramp[::-1]
        return wav.astype(np.float32)

    @staticmethod
    def _ensure_list(value: Any) -> list:
        """Official-style scalar->list promotion (scalars stay objects otherwise)."""
        return value if isinstance(value, list) else [value]

    @classmethod
    def _maybe_list(cls, value: Any, count: int) -> Any:
        """Broadcast scalars to per-sample lists but preserve None verbatim."""
        if value is None:
            return None
        values = cls._ensure_list(value)
        return values * count if len(values) == 1 and count > 1 else values

    @classmethod
    def _per_sample(cls, value: Any, count: int,
                    field: str, field_zh: str) -> Any:
        """:meth:`_maybe_list` plus a batch-size guard for explicit lists.

        Scalars broadcast to *count* entries; an explicit list must already
        carry one entry per text, otherwise the mismatch is a caller bug and
        raises a bilingual ValueError (never a silent broadcast or a raw
        IndexError further down the loop).
        """
        if value is None:
            return None
        if isinstance(value, list) and len(value) != count:
            raise ValueError(
                f"batch size mismatch: {count} texts vs {len(value)} {field} "
                f"(批量大小不一致：{count} 条文本对 {len(value)} 个{field_zh})"
            )
        return cls._maybe_list(value, count)

    @classmethod
    def _broadcast(cls, value: Any, count: int) -> list:
        """language always lands as a per-sample list ('Auto'/None -> auto)."""
        if value is None:
            value = "Auto"
        return cls._maybe_list(value, count)

    def _validate_speakers(self, speakers: Iterable[Any]) -> None:
        known = {str(s).lower() for s in self.speakers}
        for speaker in speakers:
            if str(speaker).lower() not in known:
                raise ValueError(
                    f"unsupported speaker {speaker!r} (不支持的说话人)；"
                    f"supported speakers: {sorted(known)}"
                )

    def _validate_languages(self, languages: Iterable[Any]) -> None:
        known = {str(l).lower() for l in self.languages}
        for language in languages:
            if str(language).lower() not in known:
                raise ValueError(
                    f"unsupported language {language!r} (不支持的语言)；"
                    f"supported languages: {sorted(known)}"
                )
