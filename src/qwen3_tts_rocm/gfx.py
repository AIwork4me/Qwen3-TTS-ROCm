"""Architecture support contract for Qwen3-TTS-ROCm (v0.3).

Single Python source of truth for the gfx-target mapping designed in
``docs/development/multiarch-design.md``:

- ``gfx1100`` -> wheel extras ``device-gfx1100`` (validation state:
  **current** — freshly executable/validated in v0.3 on the Radeon Pro
  W7900D host available to this program);
- ``gfx1151`` -> wheel extras ``device-gfx1151`` (validation state:
  **historical** — validated on the Radeon 8060S host before v0.3; the
  hardware was NOT available during v0.3 and was not rerun).

``scripts/install.sh`` carries a deliberately literal bash twin of this
table (it must resolve the target before any venv exists); a CPU test
asserts the two tables cannot drift.  Matching is exact-string only: no
family patterns (``gfx11*`` is never accepted), no HSA spoofing, and any
architecture outside this table is "not validated" — never "unsupported"
(the project makes no claim about hardware it has not validated).

This module is stdlib-only and safe to import from ``env.py``/CLI paths
on machines without torch.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AMD_ROCM_INDEX",
    "KNOWN_TARGETS",
    "TARGETS",
    "TORCHAUDIO_PIN",
    "TORCHVISION_PIN",
    "TORCH_PIN",
    "VALIDATION_CURRENT",
    "VALIDATION_HISTORICAL",
    "GfxTarget",
    "classification_line",
    "classify_arch",
    "is_apu_arch",
    "pip_install_argv",
]

# --- pinned wheel stack (shared with scripts/install.sh; do not drift) -------
AMD_ROCM_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
TORCH_PIN = "2.12.0+rocm7.14.0"
TORCHVISION_PIN = "0.27.0+rocm7.14.0"
TORCHAUDIO_PIN = "2.11.0+rocm7.14.0"  # ships no device extra on this index

VALIDATION_CURRENT = "current"
VALIDATION_HISTORICAL = "historical"


@dataclass(frozen=True)
class GfxTarget:
    """One supported architecture: wheel extras + evidence class."""

    name: str
    extras: str
    validation_state: str
    hardware: str  # the evidence host class (descriptive, not a constraint)


TARGETS: dict[str, GfxTarget] = {
    "gfx1100": GfxTarget(
        name="gfx1100",
        extras="device-gfx1100",
        validation_state=VALIDATION_CURRENT,
        hardware="Radeon Pro W7900D (48 GB discrete)",
    ),
    "gfx1151": GfxTarget(
        name="gfx1151",
        extras="device-gfx1151",
        validation_state=VALIDATION_HISTORICAL,
        hardware="Radeon 8060S (Ryzen AI Max+ PRO 395 iGPU)",
    ),
}

KNOWN_TARGETS = frozenset(TARGETS)

# Unified-memory APU/iGPU class — drives the memory advisory in env.py.
# gfx1100-class parts are discrete GPUs with dedicated VRAM and are NOT in
# this set (the W7900D reports the marketing name "AMD Radeon Graphics",
# which must not be confused for an APU when its arch is known).
_APU_ARCHS = frozenset({"gfx1150", "gfx1151"})


def classify_arch(arch: str | None) -> GfxTarget | None:
    """Return the target record for ``arch`` or ``None`` when not validated."""
    if not arch:
        return None
    return TARGETS.get(str(arch).strip().lower())


def classification_line(arch: str | None) -> str:
    """Human-readable validation-state line for a visible architecture.

    Three distinct, truthful scopes:

    - validated (current-class target);
    - historically validated (gfx1151: evidence predates v0.3 and was NOT
      rerun on gfx1151 hardware during v0.3);
    - ROCm-visible but not validated (everything else).
    """
    target = classify_arch(arch)
    if target is None:
        return f"ROCm-visible but not validated architecture: {arch}"
    if target.validation_state == VALIDATION_CURRENT:
        return (
            f"validated architecture: {target.name} "
            f"(freshly validated in v0.3 on {target.hardware})"
        )
    return (
        f"historically validated architecture: {target.name} "
        f"(validated on {target.hardware} before v0.3; not rerun on "
        f"{target.name} hardware during v0.3)"
    )


def is_apu_arch(arch: str | None) -> bool:
    """True for unified-memory APU/iGPU architectures (gfx115x class)."""
    return bool(arch) and str(arch).strip().lower() in _APU_ARCHS


def pip_install_argv(target: str) -> list[str]:
    """The pinned pip argv ``scripts/install.sh`` builds for ``target``.

    Mirrors the validated commands byte-for-byte (modulo shell quoting):
    gfx1151 reproduces the v0.2.1 command; gfx1100 reproduces the command
    validated in issue #1.  Raises ``KeyError`` for unknown targets — the
    caller (and the bash twin) must reject unknowns first, so this helper
    can never silently produce a wrong-arch install.
    """
    spec = TARGETS[target]  # KeyError == programming error upstream
    return [
        ".venv/bin/python", "-m", "pip", "install",
        "--index-url", AMD_ROCM_INDEX,
        f"torch[{spec.extras}]=={TORCH_PIN}",
        f"torchvision[{spec.extras}]=={TORCHVISION_PIN}",
        f"torchaudio=={TORCHAUDIO_PIN}",
        "--no-input",
    ]
