"""Task 6: smart-default loader + compat-patch layer (TDD step 1).

Hermetic by design: ``sys.modules["qwen_tts"]`` is replaced with a fake
official module before every ``load()`` call (so the lazy import inside
``load()`` picks up the double, never the real package), and the models-root
layer (``loader.models.resolve_path`` / ``is_downloaded``) is stubbed -- no
test touches the network or multi-GB weights.  Loading a *real* model is
covered later behind the ``requires_download`` marker (Task 10 smoke).

Deliberate adjustments to the task-brief fixture (documented in
``task-6-report.md``):
* ``is_downloaded`` is stubbed to ``True``: the loader MUST refuse to load
  not-yet-downloaded refs (it may never silently fetch multi-GB weights at
  inference time), and the brief's stub path ``/models/X`` does not exist,
  so the refusal guard would fire without this stub.
* the fake ``from_pretrained`` returns ``cls()`` (an actual fake-official
  instance) instead of ``object()`` so the core API promise -- "load()
  returns the NATIVE official object, never a wrapper" -- is provable via
  ``type(ret) is`` identity rather than being vacuous.
"""

import importlib.metadata
import sys
import types

import pytest

from qwen3_tts_rocm import loader

torch_lite = pytest.importorskip("torch")

_NEEDS_HIP = pytest.mark.skipif(
    not getattr(getattr(torch_lite, "version", None), "hip", None),
    reason="sdpa-by-default resolution only applies to AMD HIP torch builds",
)


@pytest.fixture
def fake_official(monkeypatch):
    """Fake ``qwen_tts`` module + stubbed registry boundary; records kwargs."""
    seen = {}
    mod_qt = types.ModuleType("qwen_tts")

    class FakeQwen3TTSModel:
        @classmethod
        def from_pretrained(cls, ref, **kw):
            seen.update(ref=ref, **kw)
            return cls()

    mod_qt.Qwen3TTSModel = FakeQwen3TTSModel
    mod_qt.__version__ = "0.1.1"
    monkeypatch.setitem(sys.modules, "qwen_tts", mod_qt)
    monkeypatch.setattr(loader.models, "resolve_path", lambda r: "/models/X")
    monkeypatch.setattr(loader.models, "is_downloaded", lambda r: True)
    return seen


# ---------------------------------------------------------------------------
# Smart-default behaviour on a HIP GPU (brief step-1 tests, kept verbatim)
# ---------------------------------------------------------------------------


@_NEEDS_HIP
def test_defaults_sdpa_on_hip(fake_official, monkeypatch):
    loader.load("base", device="cuda:0")
    assert fake_official["attn_implementation"] == "sdpa"
    assert fake_official["dtype"] == "bfloat16"


def test_flash_attn_guard(fake_official, monkeypatch):
    monkeypatch.setattr(loader, "_flash_available", lambda: False)
    with pytest.raises(RuntimeError, match="flash.?atten"):
        loader.load("base", attn_implementation="flash_attention_2")


def test_flash_guard_message_names_rocm_wheel_gap_and_alternatives(
    fake_official, monkeypatch
):
    """The error must explain WHY (no official ROCm flash-attn wheel) and what to do."""
    monkeypatch.setattr(loader, "_flash_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        loader.load("base", attn_implementation="flash_attention_2")
    msg = str(excinfo.value)
    assert "flash attention (flash-attn)" in msg
    assert "ROCm" in msg  # names the missing-wheel situation
    assert "sdpa" in msg  # suggests the working alternative
    assert "omit" in msg.lower()


def test_flash_attention_accepted_when_importable(fake_official, monkeypatch):
    monkeypatch.setattr(loader, "_flash_available", lambda: True)
    loader.load("base", device="cuda:0", attn_implementation="flash_attention_2")
    assert fake_official["attn_implementation"] == "flash_attention_2"


def test_cpu_leaves_attn_to_official_default(fake_official):
    """On cpu the loader adds NO attn key -- the official default stands."""
    loader.load("base", device="cpu")
    assert "attn_implementation" not in fake_official


def test_explicit_non_flash_attn_passthrough(fake_official):
    loader.load("base", device="cuda:0", attn_implementation="eager")
    assert fake_official["attn_implementation"] == "eager"


# ---------------------------------------------------------------------------
# Verbatim passthrough contract (the zero-modification promise)
# ---------------------------------------------------------------------------


def test_kwargs_passthrough(fake_official):
    loader.load("base", device="cpu", dtype="float32", custom_flag=42)
    assert fake_official["custom_flag"] == 42 and fake_official["dtype"] == "float32"


def test_dtype_torch_dtype_object_passthrough_verbatim(fake_official):
    loader.load("base", device="cpu", dtype=torch_lite.bfloat16)
    assert fake_official["dtype"] is torch_lite.bfloat16


def test_device_forwarded_as_device_map(fake_official):
    loader.load("base", device="cuda:0")
    assert fake_official["device_map"] == "cuda:0"


def test_load_returns_native_official_model_never_a_wrapper(fake_official):
    """Core API promise: whatever official from_pretrained returns comes back untouched."""
    ret = loader.load("base", device="cpu")
    official_cls = sys.modules["qwen_tts"].Qwen3TTSModel
    assert type(ret) is official_cls


def test_ref_resolved_to_local_path_before_from_pretrained(fake_official):
    """Never hand the raw repo id across: from_pretrained sees the local dir."""
    loader.load("base", device="cpu")
    assert fake_official["ref"] == "/models/X"  # recorded by the fake
    # The fake is a classmethod; calling through it proves we exercised it.
    assert sys.modules["qwen_tts"].Qwen3TTSModel.from_pretrained.__self__


def test_explicit_kwargs_win_over_smart_defaults(fake_official, tmp_path):
    loader.load(
        "base",
        device="cuda:0",
        device_map={"": 0},
        dtype=torch_lite.float16,
        attn_implementation=None,
    )
    assert fake_official["device_map"] == {"": 0}
    assert fake_official["dtype"] is torch_lite.float16


# ---------------------------------------------------------------------------
# No implicit downloads: missing weights -> actionable RuntimeError
# ---------------------------------------------------------------------------


def test_not_downloaded_raises_download_hint_without_touching_official(
    fake_official, monkeypatch
):
    monkeypatch.setattr(loader.models, "is_downloaded", lambda r: False)
    with pytest.raises(RuntimeError) as excinfo:
        loader.load("base")
    msg = str(excinfo.value)
    assert "download" in msg.lower()
    assert "bash scripts/download_models.sh" in msg  # repo helper script
    assert "from qwen3_tts_rocm.models import download" in msg  # python one-liner
    assert "download('all')" in msg  # ...with its concrete invocation
    assert not fake_official  # from_pretrained never invoked


def test_unknown_alias_surfaces_registry_keyerror(fake_official, monkeypatch):
    def boom(r):
        raise KeyError("unknown model reference 'nope'")

    monkeypatch.setattr(loader.models, "resolve_path", boom)
    with pytest.raises(KeyError, match="unknown"):
        loader.load("nope")


# ---------------------------------------------------------------------------
# unload(): best-effort teardown, never raises
# ---------------------------------------------------------------------------


def test_unload_safe(monkeypatch):
    """Brief test: odd inputs (attrs absent entirely) must not raise."""
    class Dummy: pass
    called = {}
    monkeypatch.setattr(
        torch_lite.cuda, "empty_cache", lambda: called.setdefault("x", True),
        raising=False,
    )
    loader.unload(Dummy())     # must not raise even for odd inputs


def test_unload_deletes_attrs_and_clears_cache(monkeypatch):
    """Instance attributes like on the real Qwen3TTSModel (model/processor are
    assigned in its __init__) get deleted; teardown stays silent otherwise."""
    called = []
    import types as _types

    dummy = _types.SimpleNamespace(model="weights", processor="proc")
    monkeypatch.setattr(torch_lite.cuda, "empty_cache", lambda: called.append(1))
    monkeypatch.setattr(torch_lite.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch_lite.version, "hip", "7.14.60850", raising=False)
    loader.unload(dummy)
    assert not hasattr(dummy, "model") and not hasattr(dummy, "processor")
    assert called == [1]


def test_unload_skips_empty_cache_on_cpu_build(monkeypatch):
    called = []
    monkeypatch.setattr(torch_lite.cuda, "empty_cache", lambda: called.append(1))
    monkeypatch.setattr(torch_lite.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch_lite.version, "hip", None, raising=False)
    loader.unload(object())
    assert called == []


# ---------------------------------------------------------------------------
# patch.py: apply_compat_patches + loose guarded version advisory
# ---------------------------------------------------------------------------


def test_patch_list_empty_today():
    from qwen3_tts_rocm import patch

    assert patch.PATCHES == []


def test_patch_matching_version_is_silent(fake_official, recwarn):
    from qwen3_tts_rocm import patch

    patch.apply_compat_patches()
    version_warnings = [w for w in recwarn.list if "0.1" in str(w.message)]
    assert version_warnings == []


def test_patch_unknown_version_warns_but_does_not_raise(fake_official, monkeypatch):
    sys.modules["qwen_tts"].__version__ = "0.2.0"
    from qwen3_tts_rocm import patch

    with pytest.warns(UserWarning, match="0\\.1"):
        patch.apply_compat_patches()


def test_patch_case_a_no_module_no_metadata_warns_version_none(monkeypatch):
    """Case A: fresh interpreter with qwen_tts absent AND the distribution
    unresolvable -> advisory fires and official_version() is None.  The probe
    must NOT import the (heavy, banner-printing) real package."""
    from qwen3_tts_rocm import patch

    monkeypatch.delitem(sys.modules, "qwen_tts", raising=False)

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert patch.official_version() is None
    with pytest.warns(UserWarning, match="__version__"):
        patch.apply_compat_patches()


def test_patch_case_b_versionless_module_falls_back_to_dist_metadata(
    fake_official, monkeypatch, recwarn
):
    """Case B: module in sys.modules without __version__ but the installed
    distribution metadata knows it -> metadata value wins silently.  Mirrors
    this host's ground truth (real qwen-tts==0.1.1 exposes no __version__)."""
    del sys.modules["qwen_tts"].__version__
    from qwen3_tts_rocm import patch

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.1")
    assert patch.official_version() == "0.1.1"
    patch.apply_compat_patches()
    assert [w for w in recwarn.list if "qwen" in str(w.message).lower()] == []


def test_patch_case_c_loaded_module_attr_wins_without_touching_metadata(
    fake_official, monkeypatch
):
    """Case C: qwen_tts already loaded WITH __version__ -> that attr is used
    directly; the metadata layer is never consulted at all."""
    from qwen3_tts_rocm import patch

    sys.modules["qwen_tts"].__version__ = "0.1.5"

    def explode(name):
        raise AssertionError("importlib.metadata.version must not be called")

    monkeypatch.setattr(importlib.metadata, "version", explode)
    assert patch.official_version() == "0.1.5"
    patch.apply_compat_patches()  # stays inside the validated series: silent


def test_load_runs_compat_patches_first(fake_official, monkeypatch):
    calls = []
    monkeypatch.setattr(loader.patch, "apply_compat_patches", lambda: calls.append(1))
    loader.load("base", device="cpu")
    assert calls == [1]
