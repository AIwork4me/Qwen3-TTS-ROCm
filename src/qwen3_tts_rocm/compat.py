"""Compatibility and version-advisory layer between this adapter and the
official ``qwen_tts`` package (兼容与版本提示层; renamed from ``patch.py`` --
the old name overstated what this module does).

What this module actually is:

* **A version advisory.** :func:`apply_compat_patches` probes the effective
  ``qwen_tts`` version and warns (never raises) when it falls outside the
  validated ``"0.1."`` series.  That is the whole active behaviour today.
* **A reserved -- and EMPTY -- patch point.** :data:`PATCHES` exists so that a
  future behavioural adjustment, should one ever become necessary, has one
  obvious registered home instead of being scattered through call sites.
  ``loader.load()`` runs :func:`apply_compat_patches` before touching the
  official loader boundary.  The list is deliberately empty and stays empty
  as long as the zero-modification promise holds: official ``qwen-tts``
  0.1.x works unmodified on ROCm/HIP, and no patch has ever been needed.
  Nothing in this module vendors, subclasses or monkey-patches upstream code.

Version advisory
----------------
``apply_compat_patches`` probes the effective ``qwen_tts`` version and emits a
:class:`UserWarning` unless it starts with ``"0.1."`` (loose series match, not
an equality assert -- warning, never raising).  Two ground truths shaped the
probe:

* the real distribution ``qwen-tts==0.1.1`` exposes **no** ``__version__``
  attribute on the module itself; the version lives in its package metadata;
* neither this module nor its callers may import ``qwen_tts`` eagerly --
  importing it prints the official package's heavy startup banners and may be
  pure waste when ``loader.load()`` refuses afterwards anyway.  The probe
  therefore never calls :func:`importlib.import_module`.
"""

from __future__ import annotations

import importlib.metadata
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


def official_version() -> str | None:
    """Best-effort effective ``qwen_tts`` version; ``None`` when unidentifiable.

    Probe order (deliberately import-free -- a real ``qwen_tts`` import prints
    heavy banners and ``load()`` may refuse before ever needing the package):

    1. ``__version__`` of the module object *already* present in
       ``sys.modules["qwen_tts"]`` (warm interpreter or test double);
    2. :func:`importlib.metadata.version` for the ``qwen-tts``
       distribution -- reliable even though the real 0.1.1 module exposes no
       ``__version__`` attribute;
    3. ``None`` only when genuinely unidentifiable.
    """
    module = sys.modules.get("qwen_tts")
    if module is not None:
        try:
            raw = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001 - hostile shims must not break load()
            raw = None
        if raw:
            return str(raw)
    # Not loaded (or versionless): consult installed-dist metadata instead of
    # force-importing the heavy package.
    try:
        return str(importlib.metadata.version("qwen-tts"))
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - advisory layer must never raise
        return None


def apply_compat_patches() -> None:
    """Apply every callable in :data:`PATCHES` (none today), then emit a
    :class:`UserWarning` when the running ``qwen_tts`` falls outside the
    validated ``"0.1."`` series, or when its version is unidentifiable both
    from the loaded module and the installed distribution.  Never raises."""
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
