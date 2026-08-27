"""Compatibility-patch layer between this adapter and the official ``qwen_tts``
package (官方包兼容层).

Design contract (zero-modification promise): we never vendor nor subclass the
official implementation -- any behaviour adjustment that turns out to be
necessary gets registered here as one callable in :data:`PATCHES` and applied
by :func:`apply_compat_patches`, which ``loader.load()`` runs *before*
touching the official loader boundary.  Today the list is deliberately empty:
official ``qwen-tts`` 0.1.x works unmodified on ROCm/HIP.

Version advisory
----------------
``apply_compat_patches`` probes the installed (or faked/monkeypatched)
``qwen_tts`` module's ``__version__`` and emits a :class:`UserWarning` unless
it starts with ``"0.1."`` (loose series match, not an equality assert --
warning, never raising).  Two ground truths shaped the probe being fully
guarded and lazy:

* the real distribution ``qwen-tts==0.1.1`` does **not** expose a
  ``__version__`` attribute at all (probe returns ``None`` -> advisory);
* neither this module nor its callers may import ``qwen_tts`` eagerly (the
  loader's tests substitute ``sys.modules["qwen_tts"]``), so the probe looks
  at ``sys.modules`` first and only falls back to
  :func:`importlib.import_module` inside the function.
"""

from __future__ import annotations

import importlib
import sys
import warnings

__all__ = [
    "PATCHES",
    "SUPPORTED_QWEN_TTS_SERIES",
    "apply_compat_patches",
    "official_version",
]

#: Version series the compat layer was validated against (loose startswith).
SUPPORTED_QWEN_TTS_SERIES = "0.1."

#: Registered patch callables, applied in list order by
#: :func:`apply_compat_patches`.  Empty today -- ROCm-compatible as shipped.
PATCHES: list = []


def _official_module():
    """Lazily locate the (possibly faked) ``qwen_tts`` module; ``None`` if absent."""
    cached = sys.modules.get("qwen_tts")
    if cached is not None:
        return cached
    try:
        return importlib.import_module("qwen_tts")
    except Exception:  # noqa: BLE001 - advisory layer must never raise
        return None


def official_version() -> str | None:
    """Best-effort ``qwen_tts.__version__``; ``None`` when missing/unreadable."""
    module = _official_module()
    if module is None:
        return None
    try:
        raw = getattr(module, "__version__", None)
    except Exception:  # noqa: BLE001 - hostile shims must not break load()
        return None
    return str(raw) if raw else None


def apply_compat_patches() -> None:
    """Apply every callable in :data:`PATCHES` (none today), then emit a
    :class:`UserWarning` when the running ``qwen_tts`` falls outside the
    validated ``"0.1."`` series or hides its version.  Never raises."""
    for apply_one in tuple(PATCHES):
        apply_one()

    version = official_version()
    if version is None:
        warnings.warn(
            "could not read qwen_tts.__version__ (the installed qwen-tts "
            "distribution may not expose it); the compatibility layer of "
            "qwen3-tts-rocm is validated against qwen-tts "
            f"{SUPPORTED_QWEN_TTS_SERIES}x",
            stacklevel=2,
        )
    elif not version.startswith(SUPPORTED_QWEN_TTS_SERIES):
        warnings.warn(
            f"qwen-tts {version} is outside the validated "
            f"{SUPPORTED_QWEN_TTS_SERIES}x series; qwen3-tts-rocm compat "
            "patches may not cover behavioural changes in it",
            stacklevel=2,
        )
