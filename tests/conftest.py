# tests/conftest.py
"""Shared fixtures for the qwen3-tts-rocm test suite (Tasks 9-17).

* ``gpu``         -- session-scoped gate: skip every gpu-marked test unless a
                     working AMD ROCm torch stack is importable and visible.
* ``get_model``   -- factory fixture returning the official model for an alias,
                     backed by a per-session cache dict {alias: model} shared
                     across test modules so multi-GB weights load only once;
                     finalizer unloads everything at session end.
* ``model_alias`` -- parametrizable alias resolver (use with ``indirect=True``).
* ``models_ready(*aliases)`` -- marker helper: ``@models_ready("base")`` marks
                     the test skipped while required weights are not downloaded.

Plain GPU-suite helpers (Task 13 promotion; previously duplicated verbatim in
both ``tests/test_generate_custom_voice.py`` and ``tests/test_voice_design.py``
-- the third consumer in this suite made them shared):

* ``_reseed(seed)``             -- torch RNG reset before stochastic pairs.
* ``_timed_generate(model, method, **kwargs)`` -- one keyword-first call to
  ``getattr(model, method)``, printing an evidence ``[timing]`` line.
* ``_distinct(a, b)``           -- robust "two generations differ" predicate.

Import them in test modules via ``from conftest import ...`` (pytest puts this
directory on sys.path for non-package suites).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from qwen3_tts_rocm import env, loader, models

#: Module-global cache so loading cost is paid once per session, across test
#: modules. A session fixture owns its teardown; the dict itself is shared.
_MODEL_CACHE: dict[str, object] = {}


def _gpu_available_or_skip() -> None:
    """Raise pytest.skip unless require_rocm_torch() passes (separable for tests)."""
    try:
        env.require_rocm_torch()
    except Exception as exc:  # noqa: BLE001 - any diagnostic failure => skip
        pytest.skip(
            f"working AMD ROCm GPU stack unavailable, skipping GPU test "
            f"(ROCm 环境不可用，跳过 GPU 测试): {exc}"
        )


@pytest.fixture(scope="session")
def gpu():
    """Session gate for GPU-dependent tests: pass-through here (HIP present)."""
    _gpu_available_or_skip()


@pytest.fixture(scope="session")
def _model_cache():
    """The {alias: official-model} dict + unload-everything finalizer."""
    yield _MODEL_CACHE
    for alias in list(_MODEL_CACHE):
        loader.unload(_MODEL_CACHE.pop(alias))


def _build_get_model(cache: dict[str, object] | None = None):
    """Factory closure; exposed for hermetic unit-testing of the real logic."""
    if cache is None:
        cache = _MODEL_CACHE

    def get_model(alias: str):
        if alias in cache:
            return cache[alias]
        if not models.is_downloaded(alias):
            pytest.skip(
                f"model weights for {alias!r} are not downloaded "
                f"({models.local_dir()}); run `bash scripts/download_models.sh` "
                f"first instead of surprising CI with a multi-GB fetch "
                f"(模型未下载，请先运行下载脚本)"
            )
        cache[alias] = loader.load(alias)
        return cache[alias]

    return get_model


@pytest.fixture
def get_model(_model_cache):
    """Function factory per binding contract; the CACHE is session-scoped.

    ``def test_x(get_model): model = get_model("custom-voice")``
    """
    return _build_get_model(_model_cache)


@pytest.fixture
def model_alias(request):
    """Pass through ``request.param``; use with @pytest.mark.parametrize(...,
    indirect=True) to drive parametrized model suites over registry aliases."""
    return request.param


def models_ready(*aliases: str):
    """Marker helper: skip the decorated test until all *aliases* are downloaded.

    Condition is evaluated once at collection time (weights never appear
    mid-session without an explicit download step).
    """
    missing = [a for a in aliases if not models.is_downloaded(a)]
    return pytest.mark.skipif(
        bool(missing),
        reason=(
            "required model weight(s) not downloaded (模型未下载): "
            + ", ".join(missing)
            + "; run `bash scripts/download_models.sh " + " ".join(missing) + "`"
        ),
    )


# ---------------------------------------------------------------------------
# Plain GPU-suite helpers (Task 13 promotion of the T11/T12 duplicates)
# ---------------------------------------------------------------------------


def _reseed(seed: int = 1234) -> None:
    """Reseed torch RNG before stochastic pairs (not an official kwarg)."""
    import torch

    torch.manual_seed(seed)


def _timed_generate(model, method: str, **kwargs):
    """One keyword-first ``getattr(model, method)(**kwargs)`` call.

    Prints an evidence timing line so the teed pytest output doubles as the
    per-call latency record; returns the official ``(wavs, sr)`` shape.
    """
    t0 = time.perf_counter()
    wavs, sr = getattr(model, method)(**kwargs)
    print(
        f"[timing] {method} n_text={len(wavs)} "
        f"took={time.perf_counter() - t0:.1f}s"
    )
    return wavs, sr


def _distinct(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff two waveforms are demonstrably different generations.

    Different shapes already prove divergent token streams; when shapes match,
    compare element-wise over the min-length trimmed arrays (identical
    generations would need a probability-zero coincidence to pass).
    """
    if a.shape != b.shape:
        return True
    n = min(a.shape[-1], b.shape[-1])
    return float(np.abs(a[..., :n] - b[..., :n]).max()) > 0.0
