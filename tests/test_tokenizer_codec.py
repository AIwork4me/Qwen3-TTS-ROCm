# tests/test_tokenizer_codec.py
"""Task 14: GPU integration -- official speech tokenizer codec roundtrip.

Loads the REAL ``Qwen/Qwen3-TTS-Tokenizer-12Hz`` weights (682M, bf16,
``device_map="cuda:0"``) through the OFFICIAL ``Qwen3TTSTokenizer.from_pretrained``
and verifies the encode -> decode loop over the bundled synthetic reference
clip (same asset as Task 13), plus the public metadata-getter surface.

Exploratory-integration TDD on real hardware: genuine failures are product
bugs -- debugged systematically, never loosened silently.

Call-convention notes (ground truth = installed package):
* ``encode(audios=str_path)`` accepts wav paths natively (librosa load);
* spike evidence (``evidence/spike/tokenizer-smoke.txt`` + installed
  ``qwen3_tts_tokenizer.py::decode`` docstring) shows ``decode`` returns the
  plain tuple ``(List[np.ndarray], output_sample_rate)`` -- the duck-typed
  extractor below handles that arm first and keeps tolerant fallbacks
  (object-with-``get_audio``, dict payloads) for other official-style
  surfaces instead of guessing private attributes;
* for the 12Hz V2 tokenizer ``encode(...).audio_codes`` is
  ``List[LongTensor] of shape (codes_len, num_quantizers)`` at ~12 code steps
  per second (config: encode/decode rate 1920 samples per step at 24 kHz).

Assertion policy (binding): metadata getters are pinned only to DOCUMENTED
sanity ranges -- non-empty model-type string, positive-int rates -- and every
actual value is PRINTED to stdout so the teed evidence file records the real
numbers.  No exotic constants are hard-coded; the one cross-check between
observed code length and the getter-reported downsample rate uses runtime
getter values exclusively (never literal numerics).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from qwen3_tts_rocm import testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

#: Bundled reference clip must stay inside this window (task contract).
DURATION_WINDOW_S = (2.0, 4.0)


def _asset_path() -> Path:
    """Bundled demo reference clip resolved through the installed package."""
    from importlib.resources import files

    return Path(str(files("qwen3_tts_rocm") / "demo" / "assets" / "ref_en.wav"))


@pytest.fixture(scope="module")
def tok(gpu):
    """Official speech tokenizer once per module; best-effort teardown."""
    import torch
    from qwen_tts import Qwen3TTSTokenizer

    from qwen3_tts_rocm import loader, models

    t = Qwen3TTSTokenizer.from_pretrained(
        str(models.local_dir("tokenizer")),
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    yield t
    loader.unload(t)


def _extract_audio_and_sr(result) -> tuple[np.ndarray, int]:
    """Pull ``(audio float ndarray, sample_rate int)`` from ANY decode shape.

    Order of arms mirrors observed/likely official surfaces; unknown shapes
    fail loudly (TypeError) rather than being guessed into passing.
    """
    if isinstance(result, tuple) and len(result) == 2:
        wavs, sr = result
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        return np.asarray(audio), int(sr)

    if isinstance(result, dict):
        audio = next(
            (result[k] for k in ("audio", "wavs", "audio_values", "wav")
             if k in result),
            None,
        )
        sr = next(
            (result[k] for k in ("sample_rate", "sampling_rate", "sr")
             if k in result),
            None,
        )
        if audio is not None and sr is not None:
            arr = audio[0] if isinstance(audio, (list, tuple)) else audio
            return np.asarray(arr), int(sr)
        raise TypeError(f"decode dict lacks audio/sample-rate keys: {sorted(result)}")

    get_audio = getattr(result, "get_audio", None)
    if callable(get_audio):
        audio = np.asarray(get_audio())
        sr = getattr(result, "sample_rate", getattr(result, "sr", None))
        if sr is not None:
            return audio, int(sr)
        raise TypeError("get_audio-style decode object carries no sample rate")

    raise TypeError(f"unrecognized decode surface: {type(result).__name__}")


def test_metadata_getters_populated(tok):
    """Model type non-empty str; all four rate getters positive ints.

    Actual values go to stdout verbatim -- they ARE the evidence trail.
    """
    model_type = tok.get_model_type()
    rates = {
        name: getattr(tok, name)()
        for name in (
            "get_input_sample_rate",
            "get_output_sample_rate",
            "get_encode_downsample_rate",
            "get_decode_upsample_rate",
        )
    }
    print(f"[tokenizer] model_type={model_type!r}")
    for name, value in rates.items():
        print(f"[tokenizer] {name}={value}")

    assert isinstance(model_type, str) and model_type
    for name, value in rates.items():
        assert isinstance(value, int) and value > 0, f"{name}={value!r} not +int"


def test_encode_decode_roundtrip_sane_and_duration_bound(tok):
    """wav -> codes -> wav: sane decoded audio, durations within [0.5x, 2x]."""
    import soundfile as sf

    src_path = _asset_path()
    src_wav, src_sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    src_dur = len(src_wav) / src_sr
    assert DURATION_WINDOW_S[0] <= src_dur <= DURATION_WINDOW_S[1], (
        f"reference clip drifted out of its {DURATION_WINDOW_S}s window "
        f"(实际 {src_dur:.3f}s)"
    )

    t_start = time.perf_counter()
    encoded = tok.encode(str(src_path))
    t_encoded = time.perf_counter()

    # Observed code geometry vs getter-reported downsampling (runtime values
    # only -- zero hard-coded numerics; padding may cost +-a few frames).
    codes = encoded.audio_codes[0]
    frame_hop = tok.get_encode_downsample_rate()
    in_rate = tok.get_input_sample_rate()
    n_in_at_model_rate = round(len(src_wav) * in_rate / src_sr)
    expected_frames_lo = max(n_in_at_model_rate // frame_hop - 2, 1)
    expected_frames_hi = -(-n_in_at_model_rate // frame_hop) + 2  # ceil + tol
    codes_len = int(codes.shape[0])
    n_quant = int(codes.shape[-1]) if codes.ndim > 1 else 1
    print(
        f"[codec] input {src_dur:.3f}s @ {src_sr}Hz -> codes_len={codes_len} "
        f"x quantizers={n_quant}; getter hop={frame_hop} samples/step "
        f"({expected_frames_lo}-{expected_frames_hi} steps tolerated); "
        f"steps/s={codes_len / src_dur:.2f}"
    )
    assert expected_frames_lo <= codes_len <= expected_frames_hi

    decoded = tok.decode(encoded)
    t_done = time.perf_counter()
    print(f"[timing] encode took={t_encoded - t_start:.2f}s "
          f"decode took={t_done - t_encoded:.2f}s")

    audio, out_sr = _extract_audio_and_sr(decoded)
    # Cross-consistency: whatever the decode surface reports must agree with
    # the official getter (both runtime values).
    assert out_sr == tok.get_output_sample_rate()
    # Full sanity gate at the reported rate (non-silent finite unclipped 24k).
    assert audio.dtype.kind == "f", f"decoded audio dtype {audio.dtype} not float"
    testing.assert_wav_sane(audio, sr_expected=out_sr)

    # Roundtrip degradation bound: decoded duration inside [0.5x, 2x] source.
    out_dur = len(audio) / out_sr
    ratio = out_dur / src_dur
    print(f"[codec] duration {src_dur:.3f}s -> {out_dur:.3f}s (ratio={ratio:.3f})")
    assert 0.5 <= ratio <= 2.0, (
        f"roundtrip duration drifted: in={src_dur:.3f}s out={out_dur:.3f}s "
        f"(ratio {ratio:.3f} outside [0.5, 2.0])"
    )
