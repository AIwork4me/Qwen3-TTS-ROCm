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
"""

from __future__ import annotations

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
