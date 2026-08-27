"""Central smart-default loader: the front door of qwen3-tts-rocm.

``load()`` resolves an alias (or path/ref) through :mod:`qwen3_tts_rocm.models`,
applies ROCm smart defaults, and returns **exactly what the official
``qwen_tts.Qwen3TTSModel.from_pretrained`` returns** -- the native object,
never a wrapper.  Everything else you call afterwards is pure official API,
which is the core promise of this project (零修改使用官方模型对象).

Usage (mirrors the README quickstart)::

    from qwen3_tts_rocm import loader

    model = loader.load("custom-voice")            # HIP GPU defaults: bf16 + sdpa
    wavs, sr = model.generate_custom_voice(
        text="你好，欢迎来到杭州。", language="Chinese", speaker="Cherry")
    # on CPU:
    model = loader.load("base", device="cpu", dtype="float32")
    loader.unload(model)                            # best-effort teardown, never raises

Smart defaults (spec §5.3)
--------------------------
* ``dtype="bfloat16"`` (:data:`DEFAULT_DTYPE`) unless overridden; accepted
  verbatim as a string *or* ``torch.dtype``, forwarded unchanged.
* attention implementation resolution order: explicit ``attn_implementation``
  argument wins; if it is ``"flash_attention_2"`` and flash-attn is not
  importable (AMD ships no official flash-attn ROCm wheel) a :class:`RuntimeError`
  explains this and suggests omitting it or using ``"sdpa"``; otherwise a HIP
  GPU device defaults to ``"sdpa"``, while CPU leaves attention to the official
  default entirely.
* ``device=None`` -> :func:`qwen3_tts_rocm.env.pick_device`.

No implicit downloads
---------------------
Loading never fetches multi-GB weights behind your back: when the target is
not downloaded yet, ``load()`` raises a :class:`RuntimeError` pointing at
``bash scripts/download_models.sh`` /
``python -c "from qwen3_tts_rocm import models; models.download([...])"``.
Only an explicit download invocation ever touches the network.

Note that neither this module nor anything it imports may import ``qwen_tts``
eagerly -- the official package is imported lazily inside :func:`load` so test
doubles can take its place in ``sys.modules`` first.
"""

from __future__ import annotations

import gc
from importlib.util import find_spec
from typing import TYPE_CHECKING

from . import env, models, patch

if TYPE_CHECKING:  # editors only; never executed (lazy-import discipline)
    from qwen_tts import Qwen3TTSModel

__all__ = ["DEFAULT_DTYPE", "load", "resolve_attn", "unload"]

#: Default precision handed to ``from_pretrained`` (bf16 is the sweet spot for
#: gfx1151 unified-memory APUs; switch to "float32"/torch.float32 for CPU).
DEFAULT_DTYPE = "bfloat16"

_FLASH_KEY = "flash_attention_2"

_HIP_DEFAULT_ATTN = "sdpa"

_DOWNLOAD_HOWTO = (
    "Download it explicitly first (请先显式下载权重):\n"
    "  bash scripts/download_models.sh {alias}\n"
    '  python -c "from qwen3_tts_rocm import models; models.download(\'{alias}\')"'
)


def _flash_available() -> bool:
    """True iff flash-attn is installable/importable -- probed via ``find_spec``
    so we never pay the actual import. Guarded: broken namespace packages
    must not break loading."""
    try:
        return find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


def _is_hip_device(device: object) -> bool:
    """True when *device* targets a GPU under an AMD ROCm/HIP torch build."""
    if not str(device).startswith("cuda"):
        return False
    try:
        import torch  # lazy; ambient ROCm wheel from earlier tasks' installs

        return bool(getattr(getattr(torch, "version", None), "hip", None))
    except Exception:  # noqa: BLE001 - probing never blocks a load()
        return False


def resolve_attn(attn_implementation: str | None = None,
                 device: str | None = None) -> str | None:
    """Resolve the attention implementation per spec §5.3.

    Explicit argument wins (with the flash-attn guard for
    ``"flash_attention_2"``); a resolved HIP GPU defaults to ``"sdpa"``;
    everything else returns ``None`` meaning "leave it to the official
    default" (the key is simply omitted)."""
    if attn_implementation is not None:
        if attn_implementation == _FLASH_KEY and not _flash_available():
            raise RuntimeError(
                f"attn_implementation={attn_implementation!r} requested but flash "
                "attention (flash-attn) is not importable in this environment: AMD "
                "ROCm has no official flash-attn wheel. Suggestion: omit the "
                f"argument (HIP GPUs get {_HIP_DEFAULT_ATTN!r} by default) or pass "
                f"attn_implementation='{_HIP_DEFAULT_ATTN}' explicitly."
                " | AMD 官方未提供 ROCm 版 flash-attn 轮子，建议省略该参数"
                f"（HIP GPU 默认即 '{_HIP_DEFAULT_ATTN}'）。"
            )
        return attn_implementation
    if device is not None and _is_hip_device(device):
        return _HIP_DEFAULT_ATTN
    return None


def load(
    model_ref: str,
    device: str | None = None,
    dtype: object = DEFAULT_DTYPE,
    attn_implementation: str | None = None,
    **kwargs,
) -> Qwen3TTSModel:
    """Load a registered Qwen3-TTS model with ROCm smart defaults.

    Args:
        model_ref: registry alias (``"custom-voice"``, ``"base-0.6b"``, ...),
            flattened repo name, full ``"Qwen/..."`` id, or an existing local
            directory (see :func:`qwen3_tts_rocm.models.resolve_path`).
        device: explicit torch device string like ``"cuda:0"`` / ``"cpu"``;
            ``None`` auto-picks via :func:`env.pick_device`.
        dtype: forwarded verbatim into ``from_pretrained`` as ``dtype=``
            (string or ``torch.dtype``); ``None`` defers to the official default.
        attn_implementation: see :func:`resolve_attn`; ``None`` = smart default.
        **kwargs: any further keyword arguments are passed to the official
            ``Qwen3TTSModel.from_pretrained`` untouched (explicit ``device_map``
            / ``dtype`` keys win over the derived ones).

    Returns:
        Whatever ``qwen_tts.Qwen3TTSModel.from_pretrained`` returns -- the
        native official model object, unwrapped.

    Raises:
        RuntimeError: weights not downloaded yet (with the exact download
            commands), or flash-attn requested without flash-attn installed.
        KeyError: unknown alias/reference.

    Example::

        model = loader.load("custom-voice")                 # bf16 + sdpa on GPU
        wavs, sr = model.generate_custom_voice(text="Hi", language="English",
                                               speaker="Cherry")
    """
    patch.apply_compat_patches()

    resolved = models.resolve_path(model_ref)  # KeyError lists accepted forms
    if not models.is_downloaded(model_ref):
        raise RuntimeError(
            f"model {str(model_ref)!r} is not downloaded yet "
            f"(expected at {resolved}); refusing to fetch multi-GB weights "
            "implicitly at load time (加载时不会自动下载数 GB 权重).\n"
            + _DOWNLOAD_HOWTO.format(alias=model_ref)
        )

    if device is None:
        device = env.pick_device("auto")

    forward = dict(kwargs)
    forward.setdefault("device_map", device)
    if dtype is not None:
        forward.setdefault("dtype", dtype)
    resolved_attn = resolve_attn(attn_implementation, device)
    if resolved_attn is not None:
        forward.setdefault("attn_implementation", resolved_attn)

    import qwen_tts  # lazy by design; may be faked in sys.modules by tests

    return qwen_tts.Qwen3TTSModel.from_pretrained(str(resolved), **forward)


def unload(model: object) -> None:
    """Best-effort teardown of a loaded model (尽力释放，绝不抛错).

    Deletes the official wrapper's ``model``/``processor`` attributes
    (whichever exist), runs :func:`gc.collect`, and empties the caching
    allocator -- but only when a CUDA-visible device exists under a HIP
    torch build.  Never raises, even for objects missing both attributes."""
    for attr in ("model", "processor"):
        try:
            delattr(model, attr)
        except Exception:  # noqa: BLE001,S110 - best effort means exactly that
            pass
    gc.collect()
    try:
        import torch

        hip = getattr(getattr(torch, "version", None), "hip", None)
        if hip and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001,S110 - teardown must never raise
        pass
