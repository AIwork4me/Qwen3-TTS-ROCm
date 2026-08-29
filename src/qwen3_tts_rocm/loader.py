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
  importable (the validated AMD ROCm wheel stack ships no flash-attn build) a
  :class:`RuntimeError` explains this and suggests omitting it or using
  ``"sdpa"``; otherwise a HIP GPU device defaults to ``"sdpa"``, while CPU
  leaves attention to the official default entirely.
* ``device=None`` -> :func:`qwen3_tts_rocm.env.pick_device`.

No implicit downloads
---------------------
Loading never fetches multi-GB weights behind your back: when the target is
not downloaded yet, ``load()`` raises a :class:`RuntimeError` pointing at
commands you run yourself afterwards -- ``bash scripts/download_models.sh
<alias>`` (a helper shipped with this repository, after installing the
package; a single repo is only ~0.7-4.4GB) or the one-liner
``python -c "from qwen3_tts_rocm.models import download; download('all')"``.
Only an explicit download invocation ever touches the network.

Noise governance (UX-fix U3a)
-----------------------------
* ``QWEN3_TTS_ROCM_QUIET=1`` silences the one-line expectation announcement
  (加载提示) printed to stderr at the start of every ``load()``.
* The official package prints a SoX "not found" ad and a flash-attn banner
  straight to the file descriptors at import time.  The lazy
  ``import qwen_tts`` is wrapped in an fd-level capture (both fds →
  /dev/null, always restored); set ``QWEN3_TTS_ROCM_VERBOSE_IMPORT=1`` to
  keep that import-time output visible when debugging.  The capture covers
  the import ONLY -- model loading and generation output/errors always show.

Note that neither this module nor anything it imports may import ``qwen_tts``
eagerly -- the official package is imported lazily inside :func:`load` so test
doubles can take its place in ``sys.modules`` first.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import warnings
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

_DOCS_POINTER = " (详见 docs/troubleshooting.md / see docs/troubleshooting.md)"

_QUIET_ENV = "QWEN3_TTS_ROCM_QUIET"

_VERBOSE_IMPORT_ENV = "QWEN3_TTS_ROCM_VERBOSE_IMPORT"


def _alias_for_hint(model_ref: str) -> str:
    """Best-effort registry alias behind the refused *model_ref*.

    ``scripts/download_models.sh`` accepts aliases only, so repo ids and
    flattened names are mapped back to their alias; anything else (an
    existing-but-incomplete local directory) degrades to a placeholder the
    user replaces with one of the known aliases.
    """
    text = str(model_ref)
    if text in models.REPOS:
        return text
    try:
        return models.alias_of_repo(text)
    except KeyError:
        return "<alias>"


def _download_howto(model_ref: str) -> str:
    """Self-serve download instructions for a not-downloaded *model_ref*.

    Leads with the per-alias command -- one repo is a fraction of the full
    download (tokenizer≈0.7GB / 1.7B≈4.4GB / 0.6B≈2.5GB) -- then mentions the
    all-six fetch second (UX-fix U3a / A-2).
    """
    return (
        "Run one of these commands yourself first (请自行运行以下任一命令完成下载):\n"
        f"  bash scripts/download_models.sh {_alias_for_hint(model_ref)}"
        "   # just this one repo (只下载这一个: "
        "tokenizer≈0.7GB / 1.7B≈4.4GB / 0.6B≈2.5GB)\n"
        '  python -c "from qwen3_tts_rocm.models import download; download(\'all\')"'
        "   # all six repos (~18GB, 全部六个)"
        + _DOCS_POINTER
    )


@contextlib.contextmanager
def _suppress_fd_stdout_stderr():
    """Redirect OS file descriptors 1 and 2 to /dev/null for the block.

    Used ONLY around the lazy ``import qwen_tts`` (first load per process):
    the official package prints its SoX "not found" ad and flash-attn banner
    directly to the file descriptors at import time -- before any
    Python-level capture could help.  Both saved fds are restored in the
    ``finally`` clause, so the restore also happens when the body raises.
    Escape hatch: ``QWEN3_TTS_ROCM_VERBOSE_IMPORT=1`` yields without touching
    the fds so the upstream import-time output stays visible.
    """
    if os.environ.get(_VERBOSE_IMPORT_ENV) == "1":
        yield  # debugging: keep the upstream import-time output
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Drain Python userspace buffers while fd 1/2 still point at /dev/null:
        # a module-level print during the captured import sits buffered on a
        # non-tty stdout and would otherwise flush into the freshly RESTORED
        # fds right after the dup2s below, leaking the banner anyway.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001,S110 - suppression must never raise
                pass
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved:
            os.close(fd)
        os.close(devnull)


def _announce_first_load(resolved: object) -> None:
    """One bilingual expectation line before the (slow, chatty) first load.

    Skipped entirely when ``QWEN3_TTS_ROCM_QUIET=1``.
    """
    if os.environ.get(_QUIET_ENV) == "1":
        return
    print(
        f"[qwen3-tts-rocm] 加载 {resolved} … 首次加载可能较慢（取决于文件系统"
        "缓存与首次内核初始化），出现大量内核日志属正常 / loading may be slower "
        "on the first run (filesystem cache + one-time kernel initialization); "
        "verbose kernel logs are expected on first run",
        file=sys.stderr,
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
                "attention (flash-attn) is not importable in this environment: the "
                "validated AMD ROCm wheel stack ships no flash-attn build, and "
                "FlashAttention is not enabled by this project. Suggestion: omit the "
                f"argument (HIP GPUs get {_HIP_DEFAULT_ATTN!r} by default) or pass "
                f"attn_implementation='{_HIP_DEFAULT_ATTN}' explicitly."
                " | 当前已验证的 ROCm wheel 栈未包含 flash-attn（本项目未启用），"
                f"建议省略该参数（HIP GPU 默认即 '{_HIP_DEFAULT_ATTN}'）。"
                + _DOCS_POINTER
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
    _announce_first_load(resolved)
    # require_weights=True: a config-only partial repo (the exact state an
    # interrupted download leaves behind) must fall through to the actionable
    # RuntimeError below, never to transformers' raw missing-weights OSError.
    if not models.is_downloaded(model_ref, require_weights=True):
        raise RuntimeError(
            f"model {str(model_ref)!r} is not downloaded yet "
            f"(expected at {resolved}); refusing to fetch multi-GB weights "
            "implicitly at load time (加载时不会自动下载数 GB 权重).\n"
            + _download_howto(model_ref)
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

    with _suppress_fd_stdout_stderr():
        import qwen_tts  # lazy by design; may be faked in sys.modules by tests

    with warnings.catch_warnings():
        # transformers emits two "…Efficient attention…" UserWarnings per load
        # when sdpa is active; on ROCm they are pure noise (UX-fix U3a / B-1).
        # Scoped to this one call: unrelated warnings keep warning normally.
        warnings.filterwarnings("ignore", message=".*Efficient attention.*")
        return qwen_tts.Qwen3TTSModel.from_pretrained(str(resolved), **forward)


def unload(model: object) -> None:
    """Best-effort teardown of a loaded model (尽力释放，绝不抛错).

    Deletes the official wrapper's ``model``/``processor`` attributes
    (whichever exist), runs :func:`gc.collect`, and empties the caching
    allocator -- but only when a CUDA-visible device exists under a HIP
    torch build.  Never raises, even for objects missing both attributes or
    on exotic gc failures."""
    for attr in ("model", "processor"):
        try:
            delattr(model, attr)
        except Exception:  # noqa: BLE001,S110 - best effort means exactly that
            pass
    try:
        gc.collect()
    except Exception:  # noqa: BLE001,S110 - absolute never-raise guarantee
        pass
    try:
        import torch

        hip = getattr(getattr(torch, "version", None), "hip", None)
        if hip and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001,S110 - teardown must never raise
        pass
