# tests/test_generate_custom_voice_06b.py
"""Task 1A: GPU integration -- 0.6B CustomVoice E2E parity (real weights).

Upstream documents NO instruction control for 0.6B CustomVoice (1.7B has it).
The instruct test pins the INSTALLED package's actual behavior as discovered in
the Task 0 ground-truth audit; docs and UI communicate that boundary.
Keyword-first calls; robust-only distinction; assert_wav_sane primary net.

Pinned observed behavior (Task 0 ground-truth audit, 2026-09-20): instruct is
ACCEPTED BUT SILENTLY IGNORED on 0.6B -- upstream documents no instruction
control for this checkpoint, and the installed qwen-tts 0.1.1 wrapper nulls
the parameter for ``tts_model_size in "0b6"`` before use
(qwen_tts/inference/qwen3_tts_model.py:799-800, no raise / no warning), so
``generate_custom_voice(..., instruct=...)`` completes and renders normally
while the instruction has no effect.  ``test_instruct_behavior`` pins exactly
that accept-and-ignore branch; it must NOT assert any instruct-driven change.

Runs the official CustomVoice model (0.6B, bf16 + sdpa) loaded through
:func:`qwen3_tts_rocm.loader.load` on the Radeon 8060S (gfx1151) and asserts
official-API behavior end to end: metadata surface, single-sample generation,
batch broadcasting, the instruct boundary above, and sampling-kwarg
passthrough.  Exploratory-integration TDD: genuine failures here are treated
as product bugs and debugged systematically -- never loosened silently.

Call convention: keyword-first everywhere (same wrapper as the 1.7B suite; see
tests/test_generate_custom_voice.py for the positional-order ground truth).

Latency guardrail (binding program constraint): every generation passes the
official ``max_new_tokens`` kwarg bound at 512, matching the 1.7B suites and
the benchmark methodology.

Robustness policy (binding, same as the 1.7B suite): generation is
stochastic.  Every test's primary net is
:func:`qwen3_tts_rocm.testing.assert_wav_sane`; difference assertions are
robust-only (differing shapes already prove divergence; equal shapes are
compared min-length trimmed).  ``torch.manual_seed`` is set before each
generation in a comparison pair via conftest ``_reseed``.
"""

from __future__ import annotations

import pytest
from conftest import _reseed, _timed_generate

from qwen3_tts_rocm import loader, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

TEXT = "今天的天气真不错，适合去公园散步。"

#: Style instruction reused from the 1.7B suite; on 0.6B it is accepted and
#: silently dropped by the installed wrapper (see module docstring).
_INSTRUCT_CALM = "用低沉缓慢的语气说"

#: Official sampling-kwarg latency cap applied to EVERY render in this suite
#: (binding program constraint; mirrors the 1.7B suites and the benchmark).
MAX_NEW_TOKENS = 512


@pytest.fixture(scope="module")
def cv06_model(gpu):
    """Load the 0.6B CustomVoice model once for this module; unload on teardown."""
    m = loader.load("custom-voice-0.6b")
    yield m
    loader.unload(m)


def test_single(cv06_model):
    """Metadata is populated and one text renders one sane waveform."""
    spks = cv06_model.get_supported_speakers()
    langs = cv06_model.get_supported_languages()
    assert len(spks) >= 9 and langs
    assert "auto" in [str(lang).lower() for lang in langs]
    wavs, sr = _timed_generate(cv06_model, "generate_custom_voice",
                               text=TEXT, language="Auto", speaker=spks[0],
                               max_new_tokens=MAX_NEW_TOKENS)
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr, min_energy_rms=1e-3)


def test_batch(cv06_model):
    """A two-item text list broadcasts language/speaker and yields two sane wavs."""
    wavs, sr = _timed_generate(
        cv06_model, "generate_custom_voice",
        text=[TEXT, TEXT], language="Auto",
        speaker=cv06_model.get_supported_speakers()[0],
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert len(wavs) == 2
    for w in wavs:
        testing.assert_wav_sane(w, sr_expected=sr)


def test_instruct_behavior(cv06_model):
    """instruct is ACCEPTED but SILENTLY IGNORED on 0.6B -- pinned, not assumed.

    Task 0 ground-truth audit finding (qwen-tts 0.1.1,
    ``qwen3_tts_model.py:799-800``): for ``tts_model_size in "0b6"`` the
    wrapper overwrites ``instruct = None`` before use -- no raise, no warning.
    The observed branch is therefore accept-and-ignore: the call must complete
    and produce a sane waveform.  Asserting any instruct-driven output change
    would be wrong on this checkpoint (that control exists on 1.7B only).
    """
    spk = cv06_model.get_supported_speakers()[0]
    _reseed()
    wavs, sr = _timed_generate(cv06_model, "generate_custom_voice",
                               text=TEXT, language="Auto", speaker=spk,
                               instruct=_INSTRUCT_CALM,
                               max_new_tokens=MAX_NEW_TOKENS)
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr, min_energy_rms=1e-3)


def test_kwargs_passthrough(cv06_model):
    """temperature flows through to the official sampler (0.6B talker).

    Temperature extremes produce different token streams, so shape OR value
    inequality keeps the assertion robust while both renders must still pass
    the sanity gate independently.
    """
    spk = cv06_model.get_supported_speakers()[0]
    _reseed()
    lo, lo_sr = _timed_generate(cv06_model, "generate_custom_voice", text=TEXT,
                                language="Auto", speaker=spk,
                                temperature=0.01, max_new_tokens=MAX_NEW_TOKENS)
    _reseed()
    hi, hi_sr = _timed_generate(cv06_model, "generate_custom_voice", text=TEXT,
                                language="Auto", speaker=spk,
                                temperature=1.9, max_new_tokens=MAX_NEW_TOKENS)
    testing.assert_wav_sane(lo[0], sr_expected=lo_sr)
    testing.assert_wav_sane(hi[0], sr_expected=hi_sr)
    assert lo[0].shape != hi[0].shape or float(abs(lo[0] - hi[0]).max()) > 0
