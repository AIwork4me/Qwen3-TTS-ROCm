"""Tests for the python-version guard in ``scripts/install.sh``.

The guard block (between the ``>>>`` / ``<<<`` markers) must fail fast on an
unsupported interpreter BEFORE the multi-GB ROCm wheel downloads start. Both
tests execute the exact block in a subprocess — no venv, no downloads.
"""

import pathlib
import re
import subprocess
import sys

INSTALL_SH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "install.sh"


def _guard_source() -> str:
    """Extract the python body of the guard block from install.sh."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(
        r">>> python-version-guard.*?<<'PY'\n(.*?)\nPY\n# <<< python-version-guard",
        text,
        re.DOTALL,
    )
    assert match, "python-version-guard block not found (markers moved?)"
    return match.group(1)


def test_install_sh_syntax():
    subprocess.run(["bash", "-n", str(INSTALL_SH)], check=True)


def test_guard_passes_on_supported_interpreter():
    result = subprocess.run(
        [sys.executable, "-c", _guard_source()],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"Python {sys.version_info.major}.{sys.version_info.minor}" in result.stdout


def test_guard_fails_fast_on_unsupported_interpreter():
    # Simulate an old interpreter: a shim runs before the guard's own import
    # and replaces sys.version_info with a 3.9 version_info-like namedtuple
    # (plain tuple would break the guard's `v.major` access). No real 3.9 needed.
    shim = (
        "import sys as _sys, collections as _c; "
        "_VI = _c.namedtuple('version_info', 'major minor micro releaselevel serial'); "
        "_sys.version_info = _VI(3, 9, 0, 'final', 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", shim + _guard_source()],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "3.10+" in result.stderr  # states the requirement
    assert "3.9" in result.stderr  # reports the detected version
