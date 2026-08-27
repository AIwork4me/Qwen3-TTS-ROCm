# tests/test_generate_custom_voice.py
"""Task 11: GPU integration -- generate_custom_voice full matrix (real weights).

Runs the official CustomVoice model (1.7B, bf16 + sdpa) loaded through
:func:`qwen3_tts_rocm.loader.load` on the Radeon 8060S (gfx1151) and asserts
official-API behavior end to end: metadata surface, single-sample generation,
batch broadcasting, instruction control, and sampling-kwarg passthrough.
This is exploratory-integration TDD: genuine failures here are treated as
product bugs and debugged systematically -- never loosened silently.

Call convention: keyword-first everywhere.  The installed official wrapper's
positional order is ``generate_custom_voice(text, speaker, language, ...)``
(different from earlier brief sketches); the keyword names are identical, so
every call below uses keywords only.

Known-plan deviation (justified): the brief sketched
``assert ... and "Auto" in langs`` for :meth:`test_single`, but the official
wrapper lowercases every entry of ``get_supported_languages()``
(``set(str(x).lower() ...) -> sorted(...)``), so the shipped list contains
``"auto"``.  The assertion is made case-insensitive to match official behavior
instead of hard-coding a wrong expectation.

Robustness policy (binding): generation is stochastic.  Every test's primary
net is :func:`qwen3_tts_rocm.testing.assert_wav_sane`; difference assertions
are robust-only (differing shapes already prove divergence; equal shapes are
compared min-length trimmed).  ``torch.manual_seed`` is set before each
generation in a comparison pair because temperature / instruct changes the
token stream anyway -- seeds reduce rerun variance but are NOT part of the
official kwargs.  Should any distinction assert prove flaky across two tuning
attempts, it is downgraded to sanity-only with the decision recorded in the
task report.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from qwen3_tts_rocm import loader, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

TEXT = "今天的天气真不错，适合去公园散步。"

_INSTRUCT_CALM = "用低沉缓慢的语气说"
_INSTRUCT_CHEERFUL = "用非常欢快的语气说"


def _reseed(seed: int = 1234) -> None:
    """Reseed torch RNG before stochastic pairs (not an official kwarg -- see docstring)."""
    import torch

    torch.manual_seed(seed)


def _timed_generate(model, **kwargs):
    """One keyword-first generate_custom_voice call; prints an evidence timing line."""
    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(**kwargs)
    print(
        f"[timing] generate_custom_voice n_text={len(wavs)} "
        f"took={time.perf_counter() - t0:.1f}s"
    )
    return wavs, sr


def _distinct(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff two waveforms are demonstrably different generations.

    Different shapes already prove divergent token streams; when shapes match,
    compare element-wise over the min-length trimmed arrays (identical here).
    """
    if a.shape != b.shape:
        return True
    n = min(a.shape[-1], b.shape[-1])
    return float(np.abs(a[..., :n] - b[..., :n]).max()) > 0.0


@pytest.fixture(scope="module")
def cv_model(gpu):
    """Load the CustomVoice model once for this module; unload on teardown."""
    m = loader.load("custom-voice")
    yield m
    loader.unload(m)


def test_single(cv_model):
    """Metadata is populated and one text renders one sane waveform."""
    spks = cv_model.get_supported_speakers()
    langs = cv_model.get_supported_languages()
    assert len(spks) >= 9 and langs
    assert "auto" in [str(lang).lower() for lang in langs]
    wavs, sr = _timed_generate(cv_model, text=TEXT, language="Auto", speaker=spks[0])
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr, min_energy_rms=1e-3)


def test_batch(cv_model):
    """A two-item text list broadcasts language/speaker and yields two sane wavs."""
    wavs, sr = _timed_generate(
        cv_model,
        text=[TEXT, TEXT],
        language="Auto",
        speaker=cv_model.get_supported_speakers()[0],
    )
    assert len(wavs) == 2
    for w in wavs:
        testing.assert_wav_sane(w, sr_expected=sr)


def test_instruct_changes_output(cv_model):
    """Two different instructions each produce a sane waveform.

    Coordinator ruling: the brief's tautology
    ``assert abs(float(a[0].std()-b[0].std())) >= 0`` was deleted.  The
    substance is the two independent sanity gates below; additionally a robust
    distinction check runs only where it cannot false-fail (see _distinct).
    """
    spk = cv_model.get_supported_speakers()[0]
    _reseed()
    a, _ = _timed_generate(cv_model, text=TEXT, language="Auto", speaker=spk,
                           instruct=_INSTRUCT_CALM)
    _reseed()
    b, _ = _timed_generate(cv_model, text=TEXT, language="Auto", speaker=spk,
                           instruct=_INSTRUCT_CHEERFUL)
    testing.assert_wav_sane(a[0])
    testing.assert_wav_sane(b[0])
    # Robust distinction: shape mismatch proves divergence outright; equal
    # shapes must differ in at least one sample (prob-0 coincidence otherwise).
    assert _distinct(a[0], b[0])


def test_sampling_kwarg_passthrough_effect(cv_model):
    """temperature flows through to the official sampler (brief NOTE applied).

    Temperature extremes produce different token streams, so shape OR value
    inequality keeps the assertion robust while both renders must still pass
    the sanity gate independently.
    """
    spk = cv_model.get_supported_speakers()[0]
    _reseed()
    lo, lo_sr = _timed_generate(cv_model, text=TEXT, language="Auto",
                                speaker=spk, temperature=0.01)
    _reseed()
    hi, hi_sr = _timed_generate(cv_model, text=TEXT, language="Auto",
                                speaker=spk, temperature=1.9)
    testing.assert_wav_sane(lo[0], sr_expected=lo_sr)
    testing.assert_wav_sane(hi[0], sr_expected=hi_sr)
    assert lo[0].shape != hi[0].shape or float(abs(lo[0] - hi[0]).max()) > 0
