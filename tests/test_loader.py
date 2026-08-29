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
import os
import sys
import types
import warnings

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
    """The error must explain WHY (scoped to the validated stack, not a claim
    about ROCm support in general) and what to do instead."""
    monkeypatch.setattr(loader, "_flash_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        loader.load("base", attn_implementation="flash_attention_2")
    msg = str(excinfo.value)
    assert "flash attention (flash-attn)" in msg
    assert "ROCm" in msg  # names the stack situation
    assert "sdpa" in msg  # suggests the working alternative
    assert "omit" in msg.lower()
    assert "docs/troubleshooting.md" in msg  # UX-fix U2: docs pointer


def test_flash_guard_message_makes_no_generalized_claims(
    fake_official, monkeypatch
):
    """Both language segments stay scoped to the validated stack: no verdicts
    about AMD/ROCm support at large."""
    monkeypatch.setattr(loader, "_flash_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        loader.load("base", attn_implementation="flash_attention_2")
    msg = str(excinfo.value)
    assert "validated" in msg  # scoped to the validated stack
    for banned in ("no official", "官方未提供", "AMD does not support",
                   "no flash-attn on ROCm"):
        assert banned not in msg, banned


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
    assert "docs/troubleshooting.md" in msg  # UX-fix U2: docs pointer
    assert not fake_official  # from_pretrained never invoked


def test_download_howto_leads_with_per_alias_command(fake_official, monkeypatch):
    """UX-fix U3a / A-2 (code half): the refusal must lead with the
    per-alias download (a single repo, e.g. 0.7-4.4GB) and only mention the
    all-six fetch second."""
    monkeypatch.setattr(loader.models, "is_downloaded", lambda r: False)
    with pytest.raises(RuntimeError) as excinfo:
        loader.load("base")
    msg = str(excinfo.value)
    per_alias = msg.find("bash scripts/download_models.sh base")
    all_six = msg.find("download('all')")
    assert 0 <= per_alias < all_six  # per-alias command comes first
    assert "from qwen3_tts_rocm.models import download" in msg  # all-six line kept


def test_download_howto_maps_repo_ids_to_their_alias(fake_official, monkeypatch):
    """download_models.sh accepts aliases only: a refused flattened name or
    full repo id still yields a runnable per-alias command."""
    monkeypatch.setattr(loader.models, "is_downloaded", lambda r: False)
    for ref in ("Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        with pytest.raises(RuntimeError) as excinfo:
            loader.load(ref)
        assert "bash scripts/download_models.sh base" in str(excinfo.value)


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


# ---------------------------------------------------------------------------
# UX-fix U3a / B-1: loader noise governance
# ---------------------------------------------------------------------------


def test_suppress_fd_swallows_fd_level_writes_and_restores(capfd):
    """True fd-level capture (os.dup/dup2): writes landing on fd 1 inside the
    block never reach the terminal, and BOTH fds are usable again after."""
    with loader._suppress_fd_stdout_stderr():
        os.write(1, b"hidden-stdout")
        os.write(2, b"hidden-stderr")
    captured = capfd.readouterr()  # capfd decodes; fd-level writes still unseen
    assert "hidden-stdout" not in captured.out
    assert "hidden-stderr" not in captured.err

    os.write(1, b"visible-stdout")  # restored for real writes afterwards
    os.write(2, b"visible-stderr")
    after = capfd.readouterr()
    assert "visible-stdout" in after.out and "visible-stderr" in after.err


def test_suppress_fd_restores_fds_when_body_raises(capfd):
    """The restore MUST happen on the exception path too."""
    with pytest.raises(RuntimeError, match="boom"), loader._suppress_fd_stdout_stderr():
        os.write(1, b"hidden-before-boom")
        raise RuntimeError("boom")
    assert "hidden-before-boom" not in capfd.readouterr().out
    os.write(1, b"restored-after-exception")
    assert "restored-after-exception" in capfd.readouterr().out


def test_suppress_fd_swallows_buffered_python_prints():
    """Regression, proven in a subprocess (pytest's own capture would mask fd
    buffering): with stdout block-buffered (non-tty), a module-level print
    from the captured import sits in sys.stdout's userspace buffer and used to
    flush into the RESTORED fd 1 right after the block -- leaking the upstream
    banner anyway.  The restore must drain the buffers into /dev/null first."""
    import subprocess

    code = (
        "import sys\n"
        "from qwen3_tts_rocm import loader\n"
        "with loader._suppress_fd_stdout_stderr():\n"
        "    print('import-time-banner')\n"
        "sys.stdout.flush()\n"
        "print('after-block-visible')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "import-time-banner" not in proc.stdout  # buffered banner swallowed
    assert "after-block-visible" in proc.stdout  # normal output unaffected


def test_verbose_import_env_keeps_fd_output_visible(monkeypatch, capfd):
    """Escape hatch: QWEN3_TTS_ROCM_VERBOSE_IMPORT=1 skips the capture so the
    upstream import-time output can be debugged."""
    monkeypatch.setenv("QWEN3_TTS_ROCM_VERBOSE_IMPORT", "1")
    with loader._suppress_fd_stdout_stderr():
        os.write(1, b"loud-import")
        os.write(2, b"loud-import-err")
    after = capfd.readouterr()
    assert "loud-import" in after.out and "loud-import-err" in after.err


def test_load_announces_first_run_expectation_on_stderr(fake_official, capsys):
    """One bilingual line at the start of load(): the first load may be slower
    (filesystem cache + one-time kernel init) and floods the terminal with
    kernel logs -- that is normal. No fixed duration is promised."""
    loader.load("base", device="cpu")
    err = capsys.readouterr().err
    assert "首次加载" in err
    assert "verbose kernel logs are expected on first run" in err
    assert "/models/X" in err  # the resolved target, short form


def test_first_load_announcement_promises_no_fixed_duration(fake_official, capsys):
    """The announcement must not predict a wall-clock duration it cannot back."""
    loader.load("base", device="cpu")
    err = capsys.readouterr().err
    assert "数十秒" not in err
    assert "tens of seconds" not in err
    assert "缓存" in err and "initialization" in err  # names the real causes


def test_quiet_env_suppresses_expectation_line(fake_official, capsys, monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_ROCM_QUIET", "1")
    loader.load("base", device="cpu")
    assert "首次加载" not in capsys.readouterr().err


def test_fd_capture_wraps_import_only_not_from_pretrained(fake_official, capfd, monkeypatch):
    """Real load errors must stay visible: only the lazy ``import qwen_tts``
    is captured; anything from_pretrained writes reaches the terminal."""
    mod_qt = types.ModuleType("qwen_tts")

    class LoudFakeModel:
        @classmethod
        def from_pretrained(cls, ref, **kw):
            os.write(2, b"visible-during-from-pretrained")
            return cls()

    mod_qt.Qwen3TTSModel = LoudFakeModel
    monkeypatch.setitem(sys.modules, "qwen_tts", mod_qt)

    loader.load("base", device="cpu")
    assert "visible-during-from-pretrained" in capfd.readouterr().err


def test_load_filters_sdpa_efficient_attention_warnings_not_others(
    fake_official, monkeypatch
):
    """The two sdpa fallback UserWarnings ("Flash Efficient attention" /
    "Mem Efficient attention") are filtered for the duration of
    from_pretrained only; unrelated warnings still propagate."""
    mod_qt = types.ModuleType("qwen_tts")

    class WarnFakeModel:
        @classmethod
        def from_pretrained(cls, ref, **kw):
            warnings.warn(
                "Flash Efficient attention is not available, falling back.",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn(
                "Mem Efficient attention is not available, falling back.",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn("Some unrelated deprecation notice", UserWarning, stacklevel=2)
            return cls()

    mod_qt.Qwen3TTSModel = WarnFakeModel
    monkeypatch.setitem(sys.modules, "qwen_tts", mod_qt)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader.load("base", device="cpu")
    msgs = [str(w.message) for w in caught if w.category is UserWarning]
    assert not any("Efficient attention" in m for m in msgs)  # both sdpa ads gone
    assert any("unrelated deprecation" in m for m in msgs)  # everything else kept
