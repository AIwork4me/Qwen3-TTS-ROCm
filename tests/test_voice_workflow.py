# tests/test_voice_workflow.py
"""Task 3 (P0 capability parity): GPU integration -- Voice Design -> reusable
voice workflow suite (real 1.7B VoiceDesign + Base weights).

Exercises :mod:`qwen3_tts_rocm.voice_workflow` end to end on the Radeon 8060S
(gfx1151): one natural-language description -> official ``generate_voice_design``
preview -> official ``create_voice_clone_prompt`` items (transcript rule) ->
official ``generate_voice_clone`` reuse on TWO different sentences -> official
payload save/load roundtrip -> regeneration from the reloaded items.  Every
call composes ONLY the three official APIs on objects returned by
:func:`qwen3_tts_rocm.loader.load` -- no wrapper, no proprietary voice
representation (the persisted ``"items"`` key is byte-format-identical to the
official demo's ``{"items": [asdict(item) ...]}`` payload).

Model split (probe-proven 2026-09-20, same as the official demo's own split):
the official wrapper hard-gates ``generate_voice_design`` on the VoiceDesign
checkpoint and ``create_voice_clone_prompt`` / ``generate_voice_clone`` on the
Base checkpoint, so the module fixtures hold BOTH objects during the single
``design_voice`` composition (peak ~2x 4.4 GiB in GTT, comfortably inside this
APU's unified memory); the ``designed`` fixture swap-unloads VoiceDesign right
afterwards so every reuse phase runs with Base alone resident -- exactly the
one-resident-model convention of the existing suites.

Persistence variant pinned here: ``save_voice`` writes ONE ``.pt`` file whose
``"voice_meta"`` string sidecar SURVIVES ``torch.load(..., weights_only=True)``
on the installed torch 2.12 (dict/str/bool are allowlisted globals) -- no
sibling ``.json`` needed.  Official prompt items carry torch tensors, which
pass through the save verbatim (byte-identical to the official demo's bytes).

Latency guardrail (binding, same as Tasks 11-13): every generation passes the
official ``max_new_tokens`` kwarg bound at 512 (the official 2048 default once
cost ~23 min on a degenerate loop on this iGPU).

Robustness policy (binding): generation is stochastic; every test's primary
net is :func:`qwen3_tts_rocm.testing.assert_wav_sane`; distinction assertions
are robust-only via conftest ``_distinct`` (``_reseed`` before each member).

Transcript rule (invariant, enforced by construction AND here): the
``ref_text`` inside every prompt item is EXACTLY the text that generated the
reference preview audio -- never a re-typed or approximate transcript.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from conftest import _distinct, _reseed

from qwen3_tts_rocm import loader, testing, voice_workflow

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

#: Design text == the preview's transcript == the prompt items' ref_text.
TEXT = "今天的天气真不错，适合去公园散步。"

#: A second, different sentence proving the saved voice generalises.
ALT_TEXT = "晚风轻轻吹过湖面，带来一阵清凉。"

#: Natural-language timbre description (same style cue as the Task 12 suite).
DESCRIPTION = "年轻女性，声音清亮，语速轻快"

#: Official sampling-kwarg latency cap applied to EVERY generation here.
MAX_NEW_TOKENS = 512


@pytest.fixture(scope="module")
def vd_model(gpu):
    """Load the VoiceDesign model once for this module; unload on teardown."""
    m = loader.load("voice-design")
    yield m
    loader.unload(m)


@pytest.fixture(scope="module")
def bc_model(gpu):
    """Load the Base model once for this module; unload on teardown."""
    m = loader.load("base")
    yield m
    loader.unload(m)


@pytest.fixture(scope="module")
def designed(vd_model, bc_model):
    """ONE design_voice pass per module; swap-unload VoiceDesign afterwards.

    ``design_voice`` needs both official objects alive for its two phases
    (generate on VoiceDesign, prompt creation on Base); once the DesignResult
    exists the VoiceDesign weights are released so all later reuse phases run
    with Base alone resident.
    """
    res = voice_workflow.design_voice(
        vd_model,
        prompt_model=bc_model,
        text=TEXT,
        language="Auto",
        description=DESCRIPTION,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    loader.unload(vd_model)  # swap-unload: Base alone remains resident
    return res


def test_design_voice_creates_prompt_and_preview(designed):
    """One description -> sane preview + official prompt items carrying TEXT."""
    from qwen_tts import VoiceClonePromptItem

    res = designed
    assert res.preview is not None
    preview_wav, preview_sr = res.preview
    testing.assert_wav_sane(preview_wav, sr_expected=preview_sr)

    assert len(res.prompt_items) == 1
    item = res.prompt_items[0]
    assert isinstance(item, VoiceClonePromptItem)
    # Transcript rule: ref_text is EXACTLY the text that generated the preview.
    assert item.ref_text == TEXT
    # ICL-mode invariants observed in the Task 13 suite hold here too.
    assert item.x_vector_only_mode is False
    assert item.icl_mode is True
    assert item.ref_code is not None
    assert item.ref_spk_embedding is not None

    # Design provenance travels with the result.
    assert res.description == DESCRIPTION
    assert res.language == "Auto"
    assert res.ref_text == TEXT
    # Three-phase evidence discipline starts here: design and prompt creation
    # are timed SEPARATELY (never collapsed into one number).
    assert res.timings["design_s"] > 0.0
    assert res.timings["prompt_s"] > 0.0
    print(
        f"[timing] design phase design_s={res.timings['design_s']:.1f}s "
        f"prompt_s={res.timings['prompt_s']:.1f}s"
    )


def test_reuse_across_two_sentences(bc_model, designed):
    """One designed voice regenerates TWO different sentences, both sane."""
    _reseed()
    wav1, sr1, gen_s1 = voice_workflow.reuse_voice(
        bc_model,
        prompt_items=designed.prompt_items,
        text=TEXT,
        language="Auto",
        max_new_tokens=MAX_NEW_TOKENS,
    )
    testing.assert_wav_sane(wav1, sr_expected=sr1)

    _reseed()
    wav2, sr2, gen_s2 = voice_workflow.reuse_voice(
        bc_model,
        prompt_items=designed.prompt_items,
        text=ALT_TEXT,
        language="Auto",
        max_new_tokens=MAX_NEW_TOKENS,
    )
    testing.assert_wav_sane(wav2, sr_expected=sr2)

    # Robust-only distinction: identical output for two different sentences
    # would be a probability-zero coincidence.
    assert _distinct(wav1, wav2)
    # Each reuse render is timed separately (never collapsed).
    assert gen_s1 > 0.0 and gen_s2 > 0.0
    print(
        f"[timing] reuse phase gen_s_1={gen_s1:.1f}s gen_s_2={gen_s2:.1f}s "
        f"(durations {len(wav1) / sr1:.2f}s / {len(wav2) / sr2:.2f}s)"
    )


def test_save_load_roundtrip_then_reuse(bc_model, designed, tmp_path):
    """save_voice -> load_voice parity -> regeneration still sane."""
    import torch

    path = tmp_path / "designed_voice.pt"
    voice_workflow.save_voice(designed, path)
    assert path.is_file() and path.stat().st_size > 0

    loaded = voice_workflow.load_voice(path)

    # Meta sidecar survives the official weights_only roundtrip (pinned
    # variant: ONE .pt file; no sibling .json -- see module docstring).
    assert loaded.description == designed.description
    assert loaded.language == designed.language
    assert loaded.ref_text == designed.ref_text
    assert loaded.preview is None  # the preview waveform is not persisted
    assert loaded.timings == {}    # timings are per-run, not voice properties

    # Field-for-field, device-agnostic parity of the official items.
    from qwen_tts import VoiceClonePromptItem

    original, rebuilt = designed.prompt_items[0], loaded.prompt_items[0]
    assert isinstance(rebuilt, VoiceClonePromptItem)
    assert rebuilt.ref_code.dtype == original.ref_code.dtype
    assert torch.equal(rebuilt.ref_code.detach().cpu(),
                       original.ref_code.detach().cpu())
    assert rebuilt.ref_spk_embedding.dtype == original.ref_spk_embedding.dtype
    assert torch.equal(rebuilt.ref_spk_embedding.detach().cpu(),
                       original.ref_spk_embedding.detach().cpu())
    assert (rebuilt.x_vector_only_mode, rebuilt.icl_mode, rebuilt.ref_text) == (
        original.x_vector_only_mode, original.icl_mode, original.ref_text
    )
    # Transcript rule survives the roundtrip too.
    assert rebuilt.ref_text == TEXT

    # A saved-then-loaded voice still generates sane audio on the Base model.
    _reseed()
    wav, sr, gen_s = voice_workflow.reuse_voice(
        bc_model,
        prompt_items=loaded.prompt_items,
        text=ALT_TEXT,
        language="Auto",
        max_new_tokens=MAX_NEW_TOKENS,
    )
    testing.assert_wav_sane(wav, sr_expected=sr)
    assert gen_s > 0.0


def test_official_schema_preserved(designed, tmp_path):
    """The .pt payload's "items" expose EXACTLY the official dataclass fields."""
    import torch
    from qwen_tts import VoiceClonePromptItem

    path = tmp_path / "schema_voice.pt"
    voice_workflow.save_voice(designed, path)

    # The OFFICIAL load options (what qwen_tts/cli/demo.py runs).
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict) and "items" in payload

    official_fields = {f.name for f in fields(VoiceClonePromptItem)}
    raw_items = payload["items"]
    assert isinstance(raw_items, list) and len(raw_items) == len(designed.prompt_items)
    for d in raw_items:
        assert isinstance(d, dict)
        assert set(d.keys()) == official_fields
        assert d["ref_text"] == TEXT

    # The design-provenance sidecar rides along in the same file.
    assert payload["voice_meta"] == {
        "description": DESCRIPTION,
        "language": "Auto",
        "ref_text": TEXT,
    }
