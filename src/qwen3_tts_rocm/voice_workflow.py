"""First-class Voice Design -> reusable-voice workflow (音色设计到可复用音色).

Composes the three OFFICIAL inference APIs into the capability the official
demo never wired end-to-end: describe a voice in natural language, hear a
preview, then PERSIST that voice as the official reusable clone-prompt payload
and regenerate arbitrary new sentences with it -- no download/re-upload hop::

    from qwen3_tts_rocm import loader, voice_workflow

    vd = loader.load("voice-design")                     # design phase ...
    bc = loader.load("base")                             # ... needs Base for
                                                         # the prompt phase
    res = voice_workflow.design_voice(
        vd, prompt_model=bc,
        text="今天的天气真不错，适合去公园散步。",
        language="Auto", description="年轻女性，声音清亮，语速轻快",
    )                                                    # preview + prompt items
    voice_workflow.save_voice(res, "voices/bright.pt")   # official payload
    loader.unload(vd)                                    # Base alone remains

    res2 = voice_workflow.load_voice("voices/bright.pt")
    wav, sr, gen_s = voice_workflow.reuse_voice(
        bc, prompt_items=res2.prompt_items, text="晚风轻轻吹过湖面。", language="Auto",
    )

Design contract
---------------
* **Official objects only** (binding): every model call is
  ``generate_voice_design`` / ``create_voice_clone_prompt`` /
  ``generate_voice_clone`` on whatever object :func:`qwen3_tts_rocm.loader.load`
  returned -- no wrapper, no proprietary voice representation.  The persisted
  payload's ``"items"`` key is byte-format-identical to the official demo's
  ``{"items": [asdict(item) ...]}`` saved with ``torch.save`` (the official
  loader reads only ``payload["items"]``, so our files load in the stock demo).
* **Transcript rule (by construction)**: the reference audio handed to
  ``create_voice_clone_prompt`` is the very waveform ``generate_voice_design``
  returned, and its ``ref_text`` is the very ``text`` that produced it -- one
  local variable feeds both calls, so the invariant cannot drift.
* **Model split (probe-proven, 2026-09-20)**: the official wrapper HARD-GATES
  each call on ``tts_model_type`` -- ``generate_voice_design`` requires the
  VoiceDesign checkpoint (a Base object raises the official ``ValueError``
  "does not support generate_voice_design") and ``create_voice_clone_prompt``
  requires the Base checkpoint (a VoiceDesign object raises "... does not
  support create_voice_clone_prompt").  No single official model serves both
  phases, so :func:`design_voice` takes the VoiceDesign object as ``model``
  and the Base object as ``prompt_model`` (default: ``model`` itself, which
  propagates the official error honestly when misused).  :func:`reuse_voice`
  runs on the Base checkpoint.  On unified-memory APUs, unload the VoiceDesign
  model once the DesignResult exists and keep only Base resident.
* **Latency guardrail**: both entry points default ``max_new_tokens=512``
  (the official 2048 default once cost ~23 min on a degenerate loop here).

Persistence variant (pinned by ``tests/test_voice_workflow.py``)
---------------------------------------------------------------
``save_voice`` writes ONE ``.pt`` file::

    {"items": [asdict(item) ...],        # official schema, untouched
     "voice_meta": {"description": ..., "language": ..., "ref_text": ...}}

The installed torch 2.12 ``weights_only=True`` unpickler accepts the
plain-string ``voice_meta`` sidecar (dict/str/bool are allowlisted globals),
so no sibling ``.json`` is needed; the roundtrip test would have caught the
rejection and forced the sidecar-out variant.  Official prompt items carry
torch tensors, which pass through verbatim (byte-identical to the official
demo's save).  Array-like values from test doubles are converted to builtin
lists first -- the same rule the demo UI's ``_savable_payload`` applies,
because ``weights_only=True`` refuses numpy reconstruct globals while the
official loader re-wraps plain lists via ``torch.tensor`` just as happily.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DesignResult",
    "design_voice",
    "load_voice",
    "reuse_voice",
    "save_voice",
]

#: Official sampling-kwarg latency cap applied to every generation here (the
#: demo-backend guardrail value; caller overrides still win via the kwarg).
MAX_NEW_TOKENS = 512

_ITEMS_EMPTY = "Empty voice items (音色为空)."
_VOICE_FILE_INVALID = "Invalid file format (文件格式不正确)."
_ITEM_FORMAT_INVALID = "Invalid item format in file (文件内部格式错误)."
_SPK_MISSING = "Missing ref_spk_embedding (缺少说话人向量)."


@dataclass
class DesignResult:
    """One designed voice: preview audio + official reusable prompt items.

    Attributes:
        preview: ``(wav, sr)`` from ``generate_voice_design`` (``None`` after a
            :func:`load_voice` -- the preview waveform is not persisted; the
            .pt payload stays byte-compatible with the official demo's).
        prompt_items: official :class:`qwen_tts.VoiceClonePromptItem` list --
            THE reusable voice, consumable by ``generate_voice_clone`` directly.
        description: the natural-language timbre description that designed it.
        language: language string the design phase used.
        ref_text: transcript rule witness -- exactly the design ``text``.
        timings: ``{"design_s": ..., "prompt_s": ...}`` wall seconds for the
            two design-phase official calls (reuse timings come back from
            :func:`reuse_voice` instead).
    """

    preview: tuple[Any, int] | None = None
    prompt_items: list[Any] = field(default_factory=list)
    description: str | None = None
    language: str | None = None
    ref_text: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


def design_voice(model, *, text: str, language: str, description: str,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 prompt_model=None, **gen_kwargs) -> DesignResult:
    """Describe a voice, preview it, and mint its reusable prompt items.

    Two OFFICIAL calls on official model objects (see the module's
    probe-proven model split):

    1. on *model* (the VoiceDesign checkpoint):
       ``generate_voice_design(text=..., instruct=description, language=...,
       max_new_tokens=..., **gen_kwargs)`` -> ``(wavs, sr)``;
    2. on *prompt_model* (the Base checkpoint) :
       ``create_voice_clone_prompt(ref_audio=(wavs[0], sr), ref_text=text)``.

    *prompt_model* may be the Base model object itself OR a zero-arg callable
    returning one (invoked only between the two phases -- the demo backend
    passes a lambda over its size-1 model LRU, so the VoiceDesign weights are
    evicted exactly when the prompt phase begins).  ``None`` defaults to
    *model*, which propagates the official ValueError honestly when *model*
    is a VoiceDesign object.

    *gen_kwargs* (e.g. ``temperature``) are forwarded verbatim into the
    official ``generate_voice_design`` call alongside the 512 default.

    Transcript rule by construction: the single stripped *text* local feeds
    BOTH calls, so ``item.ref_text`` is always exactly the sentence the
    reference waveform speaks.  Returns the preview pair, the prompt items and
    per-phase wall timings.
    """
    stripped = text.strip() if isinstance(text, str) else text
    forward = dict(gen_kwargs)
    forward.setdefault("max_new_tokens", max_new_tokens)
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_design(
        text=stripped,
        instruct=description,
        language=language,
        **forward,
    )
    design_s = time.perf_counter() - t0
    if len(wavs) == 0:  # pragma: no cover - official API always returns >=1
        raise ValueError("generate_voice_design returned no waveform (设计阶段无输出).")

    prompt_model = model if prompt_model is None else prompt_model
    if not hasattr(prompt_model, "create_voice_clone_prompt") and callable(prompt_model):
        prompt_model = prompt_model()  # lazy factory (demo-backend LRU seam)
    t1 = time.perf_counter()
    items = prompt_model.create_voice_clone_prompt(
        ref_audio=(wavs[0], sr),
        ref_text=stripped,
    )
    prompt_s = time.perf_counter() - t1
    print(
        f"[timing] design_voice design_s={design_s:.1f}s prompt_s={prompt_s:.1f}s "
        f"(preview {len(wavs[0]) / sr:.2f}s @ {sr} Hz, n_items={len(items)})"
    )
    return DesignResult(
        preview=(wavs[0], int(sr)),
        prompt_items=list(items),
        description=description,
        language=language,
        ref_text=stripped,
        timings={"design_s": design_s, "prompt_s": prompt_s},
    )


def reuse_voice(model, *, prompt_items, text: str, language: str,
                max_new_tokens: int = MAX_NEW_TOKENS,
                **gen_kwargs) -> tuple[np.ndarray, int, float]:
    """Regenerate *text* with previously-built prompt items on the Base model.

    One OFFICIAL call: ``generate_voice_clone(text=..., language=...,
    voice_clone_prompt=prompt_items, max_new_tokens=..., **gen_kwargs)``.
    *gen_kwargs* (e.g. ``temperature``) are forwarded verbatim alongside the
    512 default.  Returns ``(wav, sr, generate_s)`` with the generation wall
    seconds recorded separately, so evidence keeps design / prompt / reuse
    timings apart.
    """
    stripped = text.strip() if isinstance(text, str) else text
    forward = dict(gen_kwargs)
    forward.setdefault("max_new_tokens", max_new_tokens)
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=stripped,
        language=language,
        voice_clone_prompt=prompt_items,
        **forward,
    )
    generate_s = time.perf_counter() - t0
    print(
        f"[timing] reuse_voice generate_s={generate_s:.1f}s "
        f"(wav {len(wavs[0]) / sr:.2f}s @ {sr} Hz)"
    )
    return wavs[0], int(sr), generate_s


def save_voice(result: DesignResult, path) -> None:
    """Persist *result* as the official demo payload plus a string sidecar.

    The ``"items"`` key is written EXACTLY like the official
    ``qwen_tts/cli/demo.py::save_prompt`` (``[asdict(item) ...]`` under
    ``torch.save``), so stock official demos load our files; official prompt
    items carry torch tensors and pass through untouched (byte-identical to
    the official save).  Array-like values from test doubles become builtin
    lists first -- ``weights_only=True`` refuses numpy globals while the
    official reconstruction consumes plain lists equally well.  The extra
    ``"voice_meta"`` dict (description / language / ref_text) carries the
    design provenance through plain allowlisted types.
    """
    import torch  # lazy: keep module import free of heavy deps

    rows: list[dict[str, Any]] = []
    for it in result.prompt_items:
        row = asdict(it)
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
        rows.append(row)
    payload = {
        "items": rows,
        "voice_meta": {
            "description": result.description,
            "language": result.language,
            "ref_text": result.ref_text,
        },
    }
    target = Path(path)
    if str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(target))


def load_voice(path) -> DesignResult:
    """Rebuild a :class:`DesignResult` saved by :func:`save_voice`.

    The ``"items"`` reconstruction replays the OFFICIAL
    ``qwen_tts/cli/demo.py::load_prompt_and_gen`` logic verbatim (``torch.load
    ... weights_only=True`` + per-item tensor re-wrap + the upstream
    ``x_vector_only_mode``/``icl_mode`` defaults); ``"voice_meta"`` restores
    description / language / ref_text.  The result's ``preview`` is ``None``
    (the preview waveform is intentionally not persisted) and ``timings`` is
    empty (timings are per-run measurements, not voice properties).
    """
    import torch  # lazy: ambient wheel from earlier tasks

    from .loader import _suppress_fd_stdout_stderr

    with _suppress_fd_stdout_stderr():
        from qwen_tts import VoiceClonePromptItem

    # str/Path are used as-is; Gradio-style file objects carry their full
    # path in .name/.path (their .name is NOT a pathlib basename).
    if isinstance(path, (str, Path)):
        resolved = Path(path)
    else:
        resolved = Path(getattr(path, "name", None)
                        or getattr(path, "path", None)
                        or str(path))
    payload = torch.load(str(resolved), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(_VOICE_FILE_INVALID)
    raw_items = payload["items"]
    if not isinstance(raw_items, list) or len(raw_items) == 0:
        raise ValueError(_ITEMS_EMPTY)

    items: list = []
    for d in raw_items:
        if not isinstance(d, dict):
            # ValueError by OFFICIAL PARITY: qwen_tts/cli/demo.py's loader
            # raises ValueError on malformed items.
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
                icl_mode=bool(d.get(
                    "icl_mode", not bool(d.get("x_vector_only_mode", False))
                )),
                ref_text=d.get("ref_text", None),
            )
        )

    meta = payload.get("voice_meta") or {}
    return DesignResult(
        preview=None,
        prompt_items=items,
        description=meta.get("description"),
        language=meta.get("language"),
        ref_text=meta.get("ref_text"),
        timings={},
    )
