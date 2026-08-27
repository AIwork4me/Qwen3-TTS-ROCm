# tests/test_voice_design.py
"""Task 12: GPU integration -- generate_voice_design suite (real weights).

Mirrors the Task 11 matrix against alias ``voice-design`` (official
VoiceDesign model, 1.7B bf16 + sdpa, loaded via
:func:`qwen3_tts_rocm.loader.load`): single sane render, batch of two,
two contrasting design instructions, and unsupported-language validation.

NOTE on ``instruct`` semantics: VoiceDesign instructions are NATURAL-LANGUAGE
TIMBRE/STYLE DESCRIPTIONS (e.g. age/gender/voice-quality/pacing cues such as
"年轻女性，声音清亮，语速轻快"), tokenised into the model's instruction channel by
the official wrapper -- they are not control codes, so any sensible Chinese or
English description is valid input.

Call convention: keyword-first everywhere.  The installed official wrapper's
positional order is ``generate_voice_design(text, instruct, language, ...)``
(different from earlier brief sketches); keyword names are identical, so every
call below uses keywords only.  Unsupported language validation surfaces the
official ``ValueError`` from ``_validate_languages`` ("Unsupported languages:
['Klingon'] ...").

Robustness policy (binding, same tolerance as Task 11): generation is
stochastic; every test's primary net is
:func:`qwen3_tts_rocm.testing.assert_wav_sane`; difference assertions are
robust-only (shape mismatch proves divergence; equal shapes compared min-length
trimmed).  ``torch.manual_seed`` before each comparison member reduces rerun
variance but is NOT part of the official kwargs.  Flaky-after-two-tuning-
attempts policy: downgrade to sanity-only and record in the task report.
"""

from __future__ import annotations

import pytest
from conftest import _distinct, _reseed, _timed_generate

from qwen3_tts_rocm import loader, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

TEXT = "今天的天气真不错，适合去公园散步。"

#: Natural-language timbre descriptions -- young bright/female vs old hoarse/male.
_INSTRUCT_YOUNG_FEMALE = "年轻女性，声音清亮，语速轻快"
_INSTRUCT_OLD_MALE = "老年男性，声音沙哑，语速缓慢"


@pytest.fixture(scope="module")
def vd_model(gpu):
    """Load the VoiceDesign model once for this module; unload on teardown."""
    m = loader.load("voice-design")
    yield m
    loader.unload(m)


def test_single_sane(vd_model):
    """One text with one design instruction renders exactly one sane waveform."""
    wavs, sr = _timed_generate(
        vd_model, "generate_voice_design",
        text=TEXT, instruct=_INSTRUCT_YOUNG_FEMALE, language="Auto",
    )
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr)


def test_batch_of_two_sane(vd_model):
    """A two-item text list broadcasts language/instruct and yields two sane wavs."""
    wavs, sr = _timed_generate(
        vd_model,
        "generate_voice_design",
        text=[TEXT, TEXT],
        instruct=_INSTRUCT_YOUNG_FEMALE,
        language="Auto",
    )
    assert len(wavs) == 2
    for w in wavs:
        testing.assert_wav_sane(w, sr_expected=sr)


def test_two_instructions_both_sane_and_distinct(vd_model):
    """Contrasting timbre descriptions each yield sane AND distinguishable audio."""
    _reseed()
    a, _ = _timed_generate(vd_model, "generate_voice_design", text=TEXT,
                           instruct=_INSTRUCT_YOUNG_FEMALE, language="Auto")
    _reseed()
    b, _ = _timed_generate(vd_model, "generate_voice_design", text=TEXT,
                           instruct=_INSTRUCT_OLD_MALE, language="Auto")
    testing.assert_wav_sane(a[0])
    testing.assert_wav_sane(b[0])
    # Robust distinction: shape mismatch proves divergence outright; equal
    # shapes must differ in at least one sample (prob-0 coincidence otherwise).
    assert _distinct(a[0], b[0])


def test_unsupported_language_raises_valueerror(vd_model):
    """A nonsensical language surfaces the official ValueError gracefully."""
    with pytest.raises(ValueError):
        vd_model.generate_voice_design(text=TEXT, instruct=_INSTRUCT_YOUNG_FEMALE,
                                       language="Klingon")
