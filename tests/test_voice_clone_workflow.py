# tests/test_voice_clone_workflow.py
"""Task 13: GPU integration -- voice clone workflow suite (real Base weights).

Exercises the FULL clone surface of the official 1.7B Base model on the
Radeon 8060S (gfx1151), loaded via :func:`qwen3_tts_rocm.loader.load`:
direct (ref_audio, ref_text) cloning, x-vector-only cloning, the reusable
``create_voice_clone_prompt`` -> ``generate_voice_clone(voice_clone_prompt=...)``
path, save/load PARITY with the OFFICIAL demo payload format, and batched
cloning.  Exploratory-integration TDD: genuine failures are product bugs and
get debugged systematically -- never loosened silently.

Call convention: keyword-first everywhere.  The installed official wrapper's
positional order is
``generate_voice_clone(text, language, ref_audio, ref_text,
x_vector_only_mode, voice_clone_prompt, non_streaming_mode, ...)``
(brief sketches implied a different order; ground truth is the installed
package).  Keyword names are identical, so every call below uses keywords only.

Latency guardrail (deliberate, documented; NOT an assertion change): every
clone generation passes the OFFICIAL ``max_new_tokens`` kwarg bound at 512.
The first full run hit the official ``generate_config.json`` default of 2048
on one stochastic degenerate loop and that single call cost 1392 s (~23 min:
long-context decode on this iGPU's sdpa path slows badly as context grows).
At 12 Hz, 512 code steps cap any render near ~42 s of audio while keeping
worst-case wall time sane; the sanity assertions themselves are untouched.

Reference audio: bundled synthetic clip ``src/qwen3_tts_rocm/demo/assets/
ref_en.wav`` -- license-clean because it is fully reproducible from
``scripts/make_ref_wav.py`` (formant-harmonic speech-like babble; NOT a plain
sine, NOT a real recording).  KNOWN LIMITATION (accepted per task contract):
clone voice quality on synthetic babble is not perceptually representative of
a real human reference; the correctness gate here is official-API sanity
(non-silent finite float waveforms at 24 kHz), so REF_TEXT is an approximate
transcript rather than an exact one.

Save/load parity source of truth: installed
``.venv/lib/python3.12/site-packages/qwen_tts/cli/demo.py`` -- save side
``payload = {"items": [asdict(it) for it in items]}; torch.save(payload, p)``;
load side ``torch.load(p, weights_only=True)`` plus per-item reconstruction
(re-wrap non-tensor ``ref_code``/``ref_spk_embedding`` via ``torch.tensor``,
defaults derived from ``x_vector_only_mode`` when keys are absent).  The
roundtrip test below replays both sides verbatim and additionally asserts
schema equality against the REAL official dataclass fields.

Robustness policy (binding, same as Tasks 11/12): generation is stochastic;
every test's primary net is :func:`qwen3_tts_rocm.testing.assert_wav_sane`;
distinction assertions are robust-only via conftest ``_distinct``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest
from conftest import _distinct, _reseed, _timed_generate

from qwen3_tts_rocm import loader, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

#: Target texts for cloned generation (English, matching the reference clip's
#: language), short enough to keep every render inside the guardrail.
TEXT = "The quick checks keep this audio stack honest every day."
ALT_TEXT = "Reuse means one prompt can serve many different sentences."

#: Approximate transcript of the bundled ~2.8 s synthetic reference clip
#: (see module docstring limitation note).
REF_TEXT = "This tiny synthetic voice was cloned for automated testing."

#: Official sampling-kwarg latency cap applied to EVERY clone render in this
#: suite (see docstring "Latency guardrail" for the measured justification).
MAX_NEW_TOKENS = 512


def _asset_path() -> Path:
    """Bundled demo asset resolved through the installed package."""
    from importlib.resources import files

    return Path(str(files("qwen3_tts_rocm") / "demo" / "assets" / "ref_en.wav"))


def _ref_audio() -> tuple[np.ndarray, int]:
    """Read the bundled clip as an official ``(waveform float32, sr)`` tuple."""
    import soundfile as sf

    wav, sr = sf.read(str(_asset_path()), dtype="float32", always_2d=False)
    assert wav.ndim == 1 and sr > 0
    return wav.astype(np.float32), int(sr)


@pytest.fixture(scope="module")
def bc_model(gpu):
    """Load the Base model once for this module; unload on teardown."""
    m = loader.load("base")
    yield m
    loader.unload(m)


@pytest.fixture(scope="module")
def bc_prompt_items(bc_model):
    """Reusable clone prompt built ONCE per module (encode + speaker embed)."""
    t0 = time.perf_counter()
    items = bc_model.create_voice_clone_prompt(
        ref_audio=_ref_audio(), ref_text=REF_TEXT
    )
    print(
        f"[timing] create_voice_clone_prompt n_items={len(items)} "
        f"took={time.perf_counter() - t0:.1f}s"
    )
    return items


def test_clone_with_ref_text(bc_model):
    """Direct clone: (wav, sr) tuple ref_audio + transcript -> sane output."""
    wavs, sr = _timed_generate(
        bc_model, "generate_voice_clone",
        text=TEXT, language="Auto",
        ref_audio=_ref_audio(), ref_text=REF_TEXT,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr)
    print(f"[info] clone-with-ref-text output duration={len(wavs[0]) / sr:.2f}s")


def test_clone_x_vector_only(bc_model):
    """Embedding-only clone (no ref_text at all) stays sane though weaker."""
    wavs, sr = _timed_generate(
        bc_model, "generate_voice_clone",
        text=TEXT, language="Auto",
        ref_audio=_ref_audio(),
        x_vector_only_mode=True,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert len(wavs) == 1
    # Weaker quality is accepted for this mode; the net is still the full
    # sanity gate (non-silent, finite, unclipped, 24 kHz).
    testing.assert_wav_sane(wavs[0], sr_expected=sr)


def test_create_prompt_reuse(bc_model, bc_prompt_items):
    """One created prompt serves TWO subsequent generate calls, both sane.

    Also pins the official item invariants observed for ICL-mode prompts:
    ``icl_mode`` flips with ``x_vector_only_mode``, ICL carries ref_code +
    speaker embedding + the reference transcript forward.
    """
    from qwen_tts import VoiceClonePromptItem

    items = bc_prompt_items
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, VoiceClonePromptItem)
    assert item.x_vector_only_mode is False
    assert item.icl_mode is True
    assert item.ref_code is not None
    assert item.ref_spk_embedding is not None
    assert item.ref_text == REF_TEXT

    _reseed()
    first, sr = _timed_generate(
        bc_model, "generate_voice_clone", text=TEXT, language="Auto",
        voice_clone_prompt=items, max_new_tokens=MAX_NEW_TOKENS,
    )
    testing.assert_wav_sane(first[0], sr_expected=sr)

    # Same prompt OBJECTS reused untouched for a second, different sentence:
    # the official API must accept them verbatim (consistency of reuse).
    second, _ = _timed_generate(
        bc_model, "generate_voice_clone", text=ALT_TEXT, language="Auto",
        voice_clone_prompt=items, max_new_tokens=MAX_NEW_TOKENS,
    )
    testing.assert_wav_sane(second[0], sr_expected=sr)
    assert _distinct(first[0], second[0])


def test_save_load_roundtrip_parity(bc_model, bc_prompt_items, tmp_path):
    """Byte-format parity with the OFFICIAL demo payload, then regenerate.

    Save exactly like installed ``qwen_tts/cli/demo.py::save_prompt`` and load
    back with its ``load_prompt_and_gen`` reconstruction logic; the rebuilt
    prompt items must equal the originals field-for-field AND still drive a
    sane generation when handed to ``generate_voice_clone``.
    """
    import torch

    items = bc_prompt_items
    original = items[0]

    # --- official SAVE side -------------------------------------------------
    payload = {"items": [asdict(it) for it in items]}
    path = tmp_path / "voice_clone_prompt.pt"
    torch.save(payload, path)

    # --- official LOAD side (weights_only=True, tensor re-wrap) -------------
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict) and "items" in loaded
    raw_items = loaded["items"]
    assert isinstance(raw_items, list) and len(raw_items) == len(items)

    official_fields = {f.name for f in fields(type(original))}
    rebuilt_items = []
    for d in raw_items:
        assert isinstance(d, dict)
        # Schema parity: the serialized dict exposes EXACTLY the official
        # dataclass field names (ref_code/ref_spk_embedding/x_vector_only_
        # mode/icl_mode/ref_text -- verified against installed source).
        assert set(d.keys()) == official_fields
        assert d["ref_text"] == REF_TEXT

        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        assert ref_spk is not None, "missing ref_spk_embedding"
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        rebuilt_items.append(
            type(original)(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get(
                    "icl_mode", not bool(d.get("x_vector_only_mode", False))
                )),
                ref_text=d.get("ref_text", None),
            )
        )

    # Field-for-field parity between reloaded and freshly created items.
    # Devices legitimately differ: ``create_voice_clone_prompt`` returns GPU
    # tensors while the official demo load pins ``map_location="cpu"`` -- so
    # compare values (and dtypes) device-agnostically instead of cross-device.
    rebuilt = rebuilt_items[0]
    assert rebuilt.ref_code.dtype == original.ref_code.dtype
    assert torch.equal(rebuilt.ref_code.detach().cpu(),
                       original.ref_code.detach().cpu())
    assert rebuilt.ref_spk_embedding.dtype == original.ref_spk_embedding.dtype
    assert torch.equal(rebuilt.ref_spk_embedding.detach().cpu(),
                       original.ref_spk_embedding.detach().cpu())
    assert (rebuilt.x_vector_only_mode, rebuilt.icl_mode, rebuilt.ref_text) == (
        original.x_vector_only_mode, original.icl_mode, original.ref_text
    )

    # A saved-then-loaded prompt must behave like the live one downstream.
    wavs, sr = _timed_generate(
        bc_model, "generate_voice_clone", text=TEXT, language="Auto",
        voice_clone_prompt=rebuilt_items, max_new_tokens=MAX_NEW_TOKENS,
    )
    assert len(wavs) == 1
    testing.assert_wav_sane(wavs[0], sr_expected=sr)


def test_batch_clone(bc_model):
    """A two-item text list with one shared reference yields two sane wavs."""
    wavs, sr = _timed_generate(
        bc_model, "generate_voice_clone",
        text=[TEXT, ALT_TEXT], language="Auto",
        ref_audio=_ref_audio(), ref_text=REF_TEXT,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert len(wavs) == 2
    for w in wavs:
        testing.assert_wav_sane(w, sr_expected=sr)
    # Robust-only distinction (binding policy): identical streams would be a
    # probability-zero coincidence for two different sentences.
    assert _distinct(wavs[0], wavs[1])
