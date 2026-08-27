"""Task 9: test-infrastructure toolkit (TDD step 1 -- these tests are RED first).

Covers ``qwen3_tts_rocm.testing`` (make_tone / write_wav / assert_wav_sane /
FakeTTSModel) and the shared fixtures in ``tests/conftest.py`` (``gpu``,
``get_model``, ``model_alias``, ``models_ready``).  Everything here is
CPU-hermetic except the single ``@pytest.mark.gpu`` smoke at the bottom, so the
step-4 gate is::

    pytest tests/test_testing_utils.py -m "not gpu"
"""

import re

import numpy as np
import pytest

from qwen3_tts_rocm import models as models_mod
from qwen3_tts_rocm.testing import (
    FakeTTSModel,
    assert_wav_sane,
    make_tone,
    write_wav,
)

CHINESE = re.compile("[\u4e00-\u9fff]")  # any CJK char: diagnostics must be bilingual


# ---------------------------------------------------------------------------
# make_tone + assert_wav_sane
# ---------------------------------------------------------------------------


def test_make_tone_default_returns_valid_pair():
    wav, sr = make_tone()
    assert sr == 24000
    assert isinstance(wav, np.ndarray) and wav.dtype.kind == "f"
    assert wav.ndim == 1 and wav.shape[0] == int(24000 * 1.0)
    assert_wav_sane(wav, sr_expected=sr)  # must pass its own sanity gate


def test_make_tone_custom_freq_seconds_shape():
    wav, sr = make_tone(seconds=0.5, freq=440.0, sr=16000)
    assert sr == 16000 and wav.shape[0] == 8000
    # dominant frequency must actually be the requested one
    spectrum = np.abs(np.fft.rfft(wav))
    peak_bin = int(np.argmax(spectrum))
    # bin width = sr / n_samples = 2 Hz -> 440 Hz sits near bin 220
    assert abs(peak_bin - 440.0 * wav.shape[0] / sr) < 3


def test_assert_wav_sane_accepts_float64_and_list_input():
    assert_wav_sane(np.sin(np.linspace(0, 3.3, 512)) * 0.5)  # float64 ok
    assert_wav_sane([0.1, -0.2, 0.3], sr_expected=None)  # array-like coerced


def test_zero_array_fails_as_silent():
    with pytest.raises(AssertionError) as ei:
        assert_wav_sane(np.zeros(1000, dtype=np.float32))
    assert CHINESE.search(str(ei.value)) and "silent" in str(ei.value).lower()


def test_int_dtype_rejected():
    with pytest.raises(AssertionError) as ei:
        assert_wav_sane(np.arange(1000, dtype=np.int16))
    msg = str(ei.value).lower()
    assert "float" in msg and CHINESE.search(str(ei.value))


def test_sr_mismatch_raises():
    wav, sr = make_tone()
    assert sr == 24000
    with pytest.raises(AssertionError) as ei:
        assert_wav_sane(wav, sr_expected=44100)
    assert "44100" in str(ei.value)


def test_sr_expected_equal_passes():
    wav, sr = make_tone()
    assert_wav_sane(wav, sr_expected=sr)
    assert_wav_sane(wav)  # sr check is optional


def test_nan_and_inf_rejected():
    bad = np.ones(256, dtype=np.float32)
    bad[7] = np.nan
    with pytest.raises(AssertionError, match="finite"):
        assert_wav_sane(bad)
    bad[7] = np.inf
    with pytest.raises(AssertionError, match="finite"):
        assert_wav_sane(bad)


def test_clipping_peak_above_limit_rejected():
    loud = np.full(1000, 2.0, dtype=np.float32)
    with pytest.raises(AssertionError) as ei:
        assert_wav_sane(loud)
    assert "peak" in str(ei.value).lower()


def test_too_quiet_below_min_energy_rejected():
    faint = np.full(1000, 5e-6, dtype=np.float32)
    with pytest.raises(AssertionError):
        assert_wav_sane(faint, min_energy_rms=1e-4)


def test_min_energy_check_disabled_when_zero():
    assert_wav_sane(np.zeros(8, dtype=np.float32), min_energy_rms=0)


def test_empty_array_rejected():
    with pytest.raises(AssertionError):
        assert_wav_sane(np.array([], dtype=np.float32))


def test_non_array_none_rejected():
    with pytest.raises(AssertionError):
        assert_wav_sane(None)


# ---------------------------------------------------------------------------
# write_wav
# ---------------------------------------------------------------------------


def test_write_wav_roundtrip(tmp_path):
    sf = pytest.importorskip("soundfile")
    wav, sr = make_tone(seconds=0.25, freq=330.0)
    path = tmp_path / "sub" / "tone.wav"
    write_wav(path, wav, sr)  # must create parent dirs itself
    read, read_sr = sf.read(str(path), dtype="float32")
    assert read_sr == sr
    # PCM_16 quantisation tolerance: sample-wise agreement within 2/32768
    assert np.max(np.abs(read - wav)) < 2.0 / 32768


# ---------------------------------------------------------------------------
# FakeTTSModel
# ---------------------------------------------------------------------------


@pytest.fixture
def fake():
    return FakeTTSModel()


def test_fake_defaults_nine_official_speakers_and_languages_incl_auto(fake):
    speakers = fake.get_supported_speakers()
    languages = fake.get_supported_languages()
    assert isinstance(speakers, list) and len(speakers) == 9
    # the nine names shipped in the official CustomVoice config spk_id map
    for expected in ("serena", "vivian", "uncle_fu", "ryan", "aiden",
                     "ono_anna", "sohee", "eric", "dylan"):
        assert expected in [s.lower() for s in speakers]
    assert isinstance(languages, list)
    assert "auto" in [l.lower() for l in languages]
    assert "chinese" in [l.lower() for l in languages]


def test_generate_custom_voice_structure(fake):
    wavs, sr = fake.generate_custom_voice(
        text="Hello world", language="English", speaker="Ryan")
    assert sr == 24000
    assert isinstance(wavs, list) and len(wavs) == 1
    wav = wavs[0]
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert wav.ndim == 1 and wav.shape[0] > 0
    assert_wav_sane(wav, sr_expected=sr)


def test_generate_batch_text_list(fake):
    wavs, sr = fake.generate_custom_voice(
        text=["a", "b", "c"], language="Auto", speaker="serena")
    assert len(wavs) == 3 and all(isinstance(w, np.ndarray) for w in wavs)
    assert_wav_sane(wavs[2], sr_expected=sr)


def test_same_text_deterministic_across_calls_and_instances():
    a1 = FakeTTSModel().generate_voice_design(text="same", language="Auto",
                                             instruct="calm")
    a2 = FakeTTSModel().generate_voice_design(text="same", language="Auto",
                                             instruct="calm")
    assert a1[1] == a2[1] == 24000
    assert np.array_equal(a1[0][0], a2[0][0])  # byte-identical by design


def test_distinct_texts_produce_distinct_arrays(fake):
    wa, _ = fake.generate_custom_voice(text="first", language="Auto",
                                       speaker="eric")
    wb, _ = fake.generate_custom_voice(text="second", language="Auto",
                                       speaker="eric")
    assert not np.array_equal(wa[0], wb[0])


def test_two_instructions_reliably_distinguishable_task11_style(fake):
    """Task 11 asserts style changes alter audio; seed scheme must guarantee it."""
    calm, _ = fake.generate_voice_design(text="你好", language="Chinese",
                                         instruct="平静地")
    excited, _ = fake.generate_voice_design(text="你好", language="Chinese",
                                            instruct="兴奋地大喊")
    assert not np.array_equal(calm, excited)


def test_generation_kwargs_change_output_but_keep_structure(fake):
    low, sr_low = fake.generate_voice_design(text="t", language="Auto",
                                             instruct="i", temperature=0.1)
    high, sr_high = fake.generate_voice_design(text="t", language="Auto",
                                               instruct="i", temperature=0.9)
    assert sr_low == sr_high == 24000
    assert low[0].shape == high[0].shape  # structure stable across calls
    assert not np.array_equal(low[0], high[0])
    assert low[0].dtype == high[0].dtype == np.float32


def test_unsupported_speaker_mirrors_official_valueerror(fake):
    with pytest.raises(ValueError, match="speaker"):
        fake.generate_custom_voice(text="hi", language="Auto", speaker="Mystery")


def test_unsupported_language_mirrors_official_valueerror(fake):
    with pytest.raises(ValueError, match="language"):
        fake.generate_custom_voice(text="hi", language="Klingon", speaker="ryan")


def test_calls_recorded_for_test_spying(fake):
    fake.generate_custom_voice(text="one", language="English", speaker="Aiden",
                               instruct=None, do_sample=True)
    fake.generate_voice_design(text="two", language="Auto", instruct="warm")
    assert len(fake.calls) == 2
    first, second = fake.calls
    assert first["method"] == "generate_custom_voice"
    assert first["text"] == ["one"]
    assert first["language"] == ["English"]
    assert first["speaker"] == ["Aiden"]
    assert first["instruct"] is None
    assert second["method"] == "generate_voice_design"
    assert second["instruct"] == ["warm"]


def test_create_voice_clone_prompt_requires_ref_text_in_icl_mode(fake):
    with pytest.raises(ValueError, match="ref_text"):
        fake.create_voice_clone_prompt(ref_audio="clip.wav")


def test_create_voice_clone_prompt_item_fields_and_clone_generation(fake):
    tone, _sr = make_tone(seconds=0.1, freq=200.0)
    items = fake.create_voice_clone_prompt(ref_audio=(tone, 24000),
                                          ref_text="transcript")
    assert isinstance(items, list) and len(items) == 1
    item = items[0]
    assert item.ref_code is not None and item.ref_spk_embedding is not None
    assert item.x_vector_only_mode is False and item.icl_mode is True

    xvec_items = fake.create_voice_clone_prompt(ref_audio=(tone, 24000),
                                                x_vector_only_mode=True)
    assert xvec_items[0].x_vector_only_mode is True
    assert xvec_items[0].icl_mode is False

    wavs, sr = fake.generate_voice_clone(text="cloned", language="English",
                                         ref_audio=(tone, 24000),
                                         ref_text="transcript")
    assert sr == 24000
    assert_wav_sane(wavs[0], sr_expected=sr)

    via_prompt, sr2 = fake.generate_voice_clone(
        text="cloned", language="English", voice_clone_prompt=xvec_items)
    assert sr2 == 24000 and len(via_prompt) == 1
    methods = {c["method"] for c in fake.calls}
    assert "generate_voice_clone" in methods


@pytest.mark.parametrize("model_alias", ["base"], indirect=True)
def test_model_alias_fixture_param_resolver(model_alias):
    """model_alias resolves request.param verbatim for indirect parametrize."""
    assert model_alias == "base"


def test_get_model_caches_per_alias(monkeypatch, tmp_path, _model_cache):
    """Same alias -> same instance; distinct aliases -> distinct instances;
    missing weights -> skip instead of surprise multi-GB download."""
    from qwen3_tts_rocm import loader

    created = []

    def fake_load(ref, **kw):  # never touches disk or GPU
        created.append(ref)
        return FakeTTSModel()

    monkeypatch.setattr(loader, "load", fake_load)
    monkeypatch.setattr(models_mod, "is_downloaded", lambda r: r != "nope")

    get_model = _get_model_factory()  # same closure conftest hands to tests
    m1 = get_model("custom-voice")
    m2 = get_model("custom-voice")
    m3 = get_model("base-0.6b")
    assert m1 is m2 and m1 is not m3
    assert created == ["custom-voice", "base-0.6b"]

    try:
        with pytest.raises(pytest.skip.Exception):
            get_model("nope")
    finally:
        _model_cache.pop("custom-voice", None)
        _model_cache.pop("base-0.6b", None)


def _get_model_factory():
    """Re-import the factory from conftest so unit tests exercise the real one."""
    import conftest

    return conftest._build_get_model()


def test_models_ready_marks_skip_when_weights_missing(monkeypatch, tmp_path):
    monkeypatch.setenv(models_mod.MODELS_DIR_ENV, str(tmp_path))  # empty root
    mark = conftest_models_ready("base")
    assert mark.args and bool(mark.args[0]) is True  # skipif condition true
    assert "base" in mark.kwargs["reason"] or "base" in str(mark.description)


def test_models_ready_not_skipping_when_downloaded(monkeypatch, tmp_path):
    (tmp_path / "Qwen3-TTS-12Hz-1.7B-Base").mkdir()
    (tmp_path / "Qwen3-TTS-12Hz-1.7B-Base" / "config.json").write_text("{}")
    monkeypatch.setenv(models_mod.MODELS_DIR_ENV, str(tmp_path))
    mark = conftest_models_ready("base")
    assert bool(mark.args[0] if mark.args else False) is False


def conftest_models_ready(*aliases):
    import conftest

    return conftest.models_ready(*aliases)


def test_gpu_requirement_helper_skips_without_rocm(monkeypatch):
    import conftest

    def broken():
        raise RuntimeError("no ROCm torch here")

    monkeypatch.setattr(conftest.env, "require_rocm_torch", broken)
    with pytest.raises(pytest.skip.Exception, match="ROCm"):
        conftest._gpu_available_or_skip()


@pytest.mark.gpu
def test_gpu_fixture_smoke(gpu):
    """Live wiring check: on this host ROCm torch IS present, so the session
    fixture must let gpu-marked tests through. Deselected under -m 'not gpu'."""
    import torch

    assert getattr(getattr(torch, "version", None), "hip", None)
