"""GPU/ROCm environment diagnostics (环境诊断层).

Pure inspection helpers shared by the loader, CLI ``qwen3-tts-rocm-check``
and the troubleshooting docs.  Nothing here raises under normal use:
every torch attribute access is guarded so ``collect()`` NEVER raises,
even on machines without torch installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    "GTT_HINT",
    "EnvReport",
    "GpuInfo",
    "collect",
    "pick_device",
    "require_rocm_torch",
    "rocm_check",
]

# AMD ROCm wheel index used by every install/fix hint below.
_AMD_ROCM_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"

GTT_HINT = ("If generation hits out-of-memory on unified-memory APUs, consider lowering "
            "max_new_tokens or switching to a 0.6B model.")

_ARCHS_KNOWN_APU = {"gfx1150", "gfx1151"}  # Strix family APUs / iGPUs


@dataclass
class GpuInfo:
    """One ROCm-visible GPU as reported by torch.cuda (每块 GPU 的信息)。"""

    name: str
    arch: str | None = None        # e.g. "gfx1151" from gcnArchName
    cu_count: int | None = None    # multi_processor_count (CU 数)


@dataclass
class EnvReport:
    """Aggregated environment snapshot (环境自检结果汇总)。"""

    hip_available: bool = False          # torch built for ROCm/HIP and usable
    rocm_version: str | None = None      # torch.version.hip, e.g. "7.14.0a1"
    gpus: list[GpuInfo] = field(default_factory=list)
    # INFO-prefixed advisories are carried in this list by design
    # (printed verbatim with their own prefix, unlike errors which always get ERROR:).
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _gpu_probe(torch) -> tuple[list[GpuInfo], list[str]]:
    """Enumerate visible HIP devices; guarded, never raises (枚举设备，绝不抛错)。

    torch exposes ``torch.cuda.get_device_count`` on most builds but some ROCm
    wheels only provide ``torch.cuda.device_count``, so try both names.
    """
    gpus, notes = [], []
    count_fn = (getattr(torch.cuda, "get_device_count", None)
                or getattr(torch.cuda, "device_count", None))
    if count_fn is None:
        return gpus, ["torch.cuda exposes no device-count API"]
    count = int(count_fn())
    for i in range(count):
        try:
            props = torch.cuda.get_device_properties(i)
            name = getattr(props, "name", None)
            if not isinstance(name, str) or not name:
                name = torch.cuda.get_device_name(i)
            arch = getattr(props, "gcnArchName", None)
            cu_count = getattr(props, "multi_processor_count", None)
            gpus.append(GpuInfo(name=str(name),
                                arch=str(arch) if arch else None,
                                cu_count=int(cu_count) if cu_count is not None else None))
        except Exception as exc:  # noqa: BLE001 - tolerate partial enumeration
            notes.append(f"GPU #{i} probe failed: {exc}")
    return gpus, notes


def collect() -> EnvReport:
    """Inspect the interpreter's torch/ROCm stack without side effects
    (纯检查：只读环境与 torch 状态，任何情况下都不抛异常)。

    torch is imported lazily *inside* this function so CPU-only callers can
    monkeypatch ``sys.modules["torch"]`` for testing.
    """
    report = EnvReport()

    try:
        import torch  # lazy import so CPU-only callers can monkeypatch sys.modules
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        report.errors.append(
            "PyTorch is not installed or not importable "
            f"(未检测到可用的 PyTorch，无法继续): {exc}. Install the AMD ROCm build: "
            f"pip install torch --index-url {_AMD_ROCM_INDEX}"
        )
        return report

    # torch.version.hip may be absent entirely (CUDA/CPU-only wheels); even merely
    # touching it can raise on exotic torch shims/mocks (non-AttributeError from
    # __getattr__), and str() can raise on hostile objects -> fully guarded,
    # defaults to None (任何访问异常都按"无 HIP"处理，绝不抛出).
    try:
        raw_hip = getattr(getattr(torch, "version", None), "hip", None)
        hip = str(raw_hip) if raw_hip else None
    except Exception:  # noqa: BLE001 - diagnostics never raise
        hip = None
    report.hip_available = hip is not None
    report.rocm_version = hip

    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - diagnostics never raise
        cuda_ok = False

    if hip is None:
        report.errors.append(
            "The installed PyTorch is NOT an AMD ROCm/HIP build (当前 torch 不是 AMD ROCm 构建，"
            "无 HIP 支持). Reinstall with: pip uninstall torch && "
            f"pip install torch --index-url {_AMD_ROCM_INDEX}"
        )
        return report

    if not cuda_ok:
        report.errors.append(
            "AMD ROCm torch found but torch.cuda.is_available() is False "
            "(检测到 AMD HIP 构建但看不到 GPU). Check amdgpu driver load, /dev/kfd "
            "permission (add yourself to the 'render' and 'video' groups) and relogin."
        )

    try:
        report.gpus, notes = _gpu_probe(torch)
        report.warnings.extend(f"WARN: {n}" for n in notes)
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        report.warnings.append(f"WARN: GPU enumeration failed (GPU 枚举失败): {exc}")

    gpus = report.gpus

    # /dev/kfd permission gate — worded differently depending on whether
    # the CUDA/HIP stack itself came up (/dev/kfd 是 ROCm 用户态入口).
    try:
        if os.path.exists("/dev/kfd") and not os.access("/dev/kfd", os.R_OK | os.W_OK):
            if cuda_ok:
                report.warnings.append(
                    "WARN: /dev/kfd exists but is not read/writable by this process; "
                    "model init may fail later (当前用户对 /dev/kfd 无读写权限，建议加入 "
                    "'render' 与 'video' 用户组后重新登录)."
                )
            else:
                report.warnings.append(
                    "WARN: No HIP device visible AND /dev/kfd lacks read/write permission "
                    "for this user — permissions are the most likely root cause "
                    "(看不到 GPU 且 /dev/kfd 权限不足，请检查用户组并重新登录)."
                )
    except OSError as exc:  # pragma: no cover - exotic platforms
        report.warnings.append(f"WARN: could not stat /dev/kfd (无法读取 /dev/kfd): {exc}")

    # HSA_OVERRIDE_GFX_VERSION workarounds break more than they help on gfx1151.
    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if override:
        report.warnings.append(
            f'WARN: HSA_OVERRIDE_GFX_VERSION="{override}" is set, but gfx1151 needs NO '
            "override on ROCm 7.x — remove it to avoid mis-targeted code objects "
            "(gfx1151 在新版 ROCm 上无需设置 HSA_OVERRIDE_GFX_VERSION，建议取消)."
        )

    # CUDA_VISIBLE_DEVICES applies to ROCm/HIP too (对该变量提示其同样生效).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and report.hip_available:
        report.warnings.append(
            f'INFO: CUDA_VISIBLE_DEVICES="{cvd}" is set — on ROCm it applies to the '
            "HIP device order as well (该变量在 ROCm 下同样决定可见 GPU 及顺序)."
        )

    # Unified-memory APU/iGPU: OOM risk factors are structural, so always advise.
    # Requires at least one enumerated GPU (all()/any() over empty lists would
    # otherwise fabricate an APU verdict when cuda reports OK but probes find none).
    try:
        is_apu = bool(gpus) and (
            any(
                (g.arch or "").lower() in _ARCHS_KNOWN_APU or "graphics" in g.name.lower()
                for g in gpus
            ) or all("radeon graphics" in g.name.lower() for g in gpus)
        )
        if is_apu:
            report.warnings.append(
                "INFO: unified-memory APU/iGPU detected — VRAM is shared with system RAM "
                f"during generation. {GTT_HINT} (检测到统一内存的 APU/iGPU，显存与系统内存共享，"
                "生成过长可能耗尽内存)."
            )
    except Exception:  # noqa: BLE001,S110 - advisory only; never block the report
        pass

    # Startup honesty about models (UX-fix U2): report how many registry repos
    # are actually usable on disk instead of failing later at load time
    # (报告已下载模型数，避免等到加载时才发现权重缺失).  Weight-aware check:
    # a config-only partial repo does not count as downloaded.
    try:
        from . import models  # lazy: keeps env.py import-light and cycle-free

        have = sum(
            1
            for alias in models.ALIASES
            if models.is_downloaded(alias, require_weights=True)
        )
        total = len(models.ALIASES)
        extra = (
            ""
            if have == total
            else " Missing ones: `bash scripts/download_models.sh <alias>` "
            "(缺失模型请先运行下载脚本)."
        )
        report.warnings.append(
            f"INFO: models: {have}/{total} downloaded (已下载模型数).{extra}"
        )
    except Exception:  # noqa: BLE001,S110 - diagnostics never raise
        pass

    return report


def rocm_check(verbose: bool = True) -> EnvReport:
    """Run :func:`collect` and optionally pretty-print diagnostics
    (打印双语诊断信息；verbose=False 时完全不输出)。

    Printed lines carry WARN:/ERROR:/INFO: prefixes and a bilingual summary tail.
    """
    report = collect()
    if not verbose:
        return report

    print("[qwen3-tts-rocm] ROCm environment self-check (环境自检)")
    for err in report.errors:
        print(err if err.startswith(("ERROR:", "WARN:", "INFO:")) else f"ERROR: {err}")
    for warn in report.warnings:
        print(warn if warn.startswith(("WARN:", "INFO:", "ERROR:")) else f"WARN: {warn}")
    state = "available (可用)" if report.hip_available else "NOT available (不可用)"
    version = report.rocm_version or "-"
    n_warn = len(report.warnings)
    n_err = len(report.errors)
    print(
        f"Summary: HIP {state}; ROCm version: {version}; GPUs: {len(report.gpus)}"
        f" ({', '.join(g.arch or '?' for g in report.gpus) or 'none'}); "
        f"warnings: {n_warn}; errors: {n_err}"
        f" | 中文摘要：环境自检完成，HIP {state}，版本 {version}，"
        f"可见 GPU {len(report.gpus)} 块，警告 {n_warn} 条，错误 {n_err} 条。"
    )
    # Generic docs pointer, printed ONCE after the summary rather than tacked
    # onto every line (UX-fix U2: 排障文档统一指向，避免刷屏).
    print("→ 更多排障步骤: docs/troubleshooting.md (more: troubleshooting guide)")
    return report


def pick_device(preference: str = "auto") -> str:
    """Resolve a torch device string; explicit values pass through unchanged
    (解析设备字符串；显式指定则原样返回)。"""
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - CPU fallback must never raise
        return "cpu"


def require_rocm_torch() -> None:
    """Raise RuntimeError with a fix hint unless a working AMD ROCm torch
    stack is importable and a HIP device is visible (校验失败时给出修复指引)。"""
    report = collect()
    if not report.errors:
        return
    hint = (
        "A working AMD ROCm PyTorch stack is required. Problems detected (发现的问题): "
        + "; ".join(report.errors)
        + f" | Fix: pip install torch --index-url {_AMD_ROCM_INDEX} "
        "(使用 AMD 官方索引重装 ROCm 版 torch 后重试)"
    )
    raise RuntimeError(hint)
