# tests/test_check_upstream_drift.py
"""Task 15 (brief 16): CPU unit tests for scripts/check_upstream_drift.py.

NO network is ever touched here:

* ``diff_state`` is pure — fed synthetic dicts only (brief Step 15.3).
* The CLI exit-code-convention tests replace ``collect()`` wholesale with a
  synthetic state (or a raised :class:`NetworkError`), so no socket is opened,
  nothing is monkeypatched at the urllib/socket layer.

The script is loaded via importlib from its path because ``scripts/`` is not
an installed package (same pattern as tests/test_benchmark.py); the module is
stdlib-only at import time, so this costs nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_upstream_drift.py"
_spec = importlib.util.spec_from_file_location("check_upstream_drift_script", _SCRIPT)
drift = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("check_upstream_drift_script", drift)
_spec.loader.exec_module(drift)


# ---------------------------------------------------------------------------
# Fixtures: synthetic baseline / current states (shapes match collect())
# ---------------------------------------------------------------------------

def _state(sha: str = "a" * 40, version: str = "0.1.1",
           files: dict | None = None, api: dict | None = None) -> dict:
    return {
        "upstream_sha": sha,
        "pypi_version": version,
        "file_hashes": files if files is not None else {
            "README.md": "blob-1",
            "finetuning/sft_12hz.py": "blob-2",
        },
        "api_surface": api if api is not None else {
            "generate_custom_voice": "(self, text: str, speaker: str, **kwargs) -> None",
            "create_voice_clone_prompt": "(self, reference_audio, text: str) -> None",
        },
    }


# ---------------------------------------------------------------------------
# diff_state on synthetic dicts (brief Step 15.3)
# ---------------------------------------------------------------------------

def test_diff_state_identical_is_empty():
    """Identical states -> {} (no drift, including empty dict sections)."""
    assert drift.diff_state(_state(), _state()) == {}
    empty = {"upstream_sha": "s", "pypi_version": "v",
             "file_hashes": {}, "api_surface": {}}
    assert drift.diff_state(empty, dict(empty)) == {}


def test_diff_state_upstream_sha_change_reported():
    """Changed SHA -> {upstream_sha: {baseline, current}} with both values."""
    base, cur = _state(sha="a" * 40), _state(sha="b" * 40)
    d = drift.diff_state(base, cur)
    assert d == {"upstream_sha": {"baseline": "a" * 40, "current": "b" * 40}}


def test_diff_state_pypi_version_change_reported():
    """Changed PyPI version -> both values reported."""
    base, cur = _state(version="0.1.1"), _state(version="0.1.2")
    d = drift.diff_state(base, cur)
    assert d == {"pypi_version": {"baseline": "0.1.1", "current": "0.1.2"}}


def test_diff_state_sha_and_version_both_reported_together():
    """Both scalar keys drift -> both entries present, nothing else."""
    d = drift.diff_state(_state(sha="a" * 40, version="0.1.1"),
                         _state(sha="b" * 40, version="0.1.2"))
    assert set(d) == {"upstream_sha", "pypi_version"}


def test_diff_state_file_hashes_added():
    cur = _state(files={"README.md": "blob-1", "finetuning/sft_12hz.py": "blob-2",
                        "checkpoints/new.safetensors": "blob-3"})
    d = drift.diff_state(_state(), cur)
    assert d["file_hashes"] == {"added": ["checkpoints/new.safetensors"],
                                "removed": [], "changed": []}


def test_diff_state_file_hashes_removed():
    base = _state(files={"README.md": "blob-1", "finetuning/sft_12hz.py": "blob-2",
                         "old/gone.md": "blob-0"})
    d = drift.diff_state(base, _state())
    assert d["file_hashes"] == {"added": [], "removed": ["old/gone.md"], "changed": []}


def test_diff_state_file_hashes_changed_exact_set():
    cur = _state(files={"README.md": "blob-1-changed",
                        "finetuning/sft_12hz.py": "blob-2"})
    d = drift.diff_state(_state(), cur)
    assert d["file_hashes"] == {"added": [], "removed": [], "changed": ["README.md"]}


def test_diff_state_file_hashes_add_remove_change_simultaneous():
    """Exact sets when add + remove + change land in one diff."""
    base = _state(files={"keep.md": "k", "gone.md": "g", "shift.md": "old"})
    cur = _state(files={"keep.md": "k", "shift.md": "new", "fresh.md": "f"})
    d = drift.diff_state(base, cur)
    assert d["file_hashes"] == {"added": ["fresh.md"], "removed": ["gone.md"],
                                "changed": ["shift.md"]}


def test_diff_state_api_surface_signature_change():
    """A generate-API signature change -> method listed under 'changed'."""
    base = _state(api={"generate_custom_voice": "(self, text: str, speaker: str) -> None",
                       "generate_voice_clone": "(self, reference_audio, text: str) -> None"})
    cur = _state(api={"generate_custom_voice": "(self, text: str, speaker: str, language: str) -> None",
                      "generate_voice_clone": "(self, reference_audio, text: str) -> None"})
    d = drift.diff_state(base, cur)
    assert d["api_surface"] == {"added": [], "removed": [],
                                "changed": ["generate_custom_voice"]}


def test_diff_state_api_surface_new_method():
    """A NEW upstream generate method appears -> 'added' (never auto-green)."""
    base = _state(api={"generate_custom_voice": "(self, text) -> None"})
    cur = _state(api={"generate_custom_voice": "(self, text) -> None",
                      "generate_streaming": "(self, text) -> None"})
    d = drift.diff_state(base, cur)
    assert d["api_surface"]["added"] == ["generate_streaming"]
    assert d["api_surface"]["changed"] == []
    assert d["api_surface"]["removed"] == []


def test_diff_state_ignores_extra_baseline_keys():
    """Unknown baseline keys must not crash the comparison (forward compat)."""
    base = {**_state(), "future_key": {"x": 1}}
    assert drift.diff_state(base, _state()) == {}


# ---------------------------------------------------------------------------
# CLI exit-code convention (collect() replaced — no network, no urlopen mock)
# ---------------------------------------------------------------------------

def _write_baseline(tmp_path: Path, state: dict) -> Path:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(state))
    return p


def test_cli_unchanged_exit_0(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(drift, "collect", lambda: _state())
    p = _write_baseline(tmp_path, _state())
    rc = drift.main(["--baseline", str(p)])
    out = capsys.readouterr().out
    assert rc == 0 and out.strip() == "UNCHANGED"


def test_cli_drift_exit_1_and_never_auto_green(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(drift, "collect", lambda: _state(sha="b" * 40))
    p = _write_baseline(tmp_path, _state(sha="a" * 40))
    rc = drift.main(["--baseline", str(p)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "UPSTREAM DRIFT DETECTED" in out
    # The wording contract: drift demands revalidation, never auto-greens.
    assert "revalidation required" in out
    assert "NOT auto-marked Radeon-compatible" in out
    assert '"upstream_sha"' in out  # the diff body is printed


def test_cli_network_error_exit_2(tmp_path, capsys, monkeypatch):
    def boom():
        raise drift.NetworkError("GET https://pypi.org/pypi/qwen-tts/json failed after retry")

    monkeypatch.setattr(drift, "collect", boom)
    p = _write_baseline(tmp_path, _state())
    rc = drift.main(["--baseline", str(p)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "NETWORK ERROR" in err and "drift NOT determined" in err


def test_cli_collection_error_exit_2(tmp_path, capsys, monkeypatch):
    """A non-network collection failure (e.g. qwen_tts missing) is also exit 2."""
    def boom():
        raise ImportError("No module named 'qwen_tts'")

    monkeypatch.setattr(drift, "collect", boom)
    p = _write_baseline(tmp_path, _state())
    rc = drift.main(["--baseline", str(p)])
    err = capsys.readouterr().err
    assert rc == 2 and "COLLECTION ERROR" in err and "drift NOT determined" in err


def test_cli_missing_baseline_exit_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(drift, "collect", lambda: _state())
    rc = drift.main(["--baseline", str(tmp_path / "nope.json")])
    err = capsys.readouterr().err
    assert rc == 2 and "BASELINE ERROR" in err


def test_cli_update_baseline_writes_current_state(tmp_path, capsys, monkeypatch):
    cur = _state(sha="c" * 40, version="0.2.0")
    monkeypatch.setattr(drift, "collect", lambda: cur)
    p = tmp_path / "baseline.json"
    rc = drift.main(["--baseline", str(p), "--update-baseline"])
    out = capsys.readouterr().out
    assert rc == 0 and out.strip() == "BASELINE UPDATED"
    assert json.loads(p.read_text()) == cur
    # And the freshly written baseline now compares UNCHANGED, exit 0.
    assert drift.main(["--baseline", str(p)]) == 0


def test_help_documents_exit_code_convention(capsys):
    """--help must state the 0/1/2 convention (network != drift != unchanged)."""
    with pytest.raises(SystemExit) as ei:
        drift.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "0 = UNCHANGED" in out
    assert "1 = UPSTREAM DRIFT DETECTED" in out
    assert "2 = network/collection error" in out
