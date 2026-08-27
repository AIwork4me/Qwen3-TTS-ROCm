"""Real CLI entry point for the enhanced Gradio demo (演示命令行入口).

Replaces the Task-15 stub.  The argument surface mirrors the official
``qwen_tts/cli/demo.py`` so muscle memory transfers verbatim
(positional checkpoint | ``-c/--checkpoint``, ``--device``/``--dtype``/
``--no-flash-attn``, ``--ip/--port/--share/--concurrency/--ssl-*`` and the
optional sampling flags), PLUS the qwen3-tts-rocm conveniences:

* ``--alias``/positional accepts a REGISTRY ALIAS (default ``custom-voice``),
  a flattened/full official repo id or any local directory -- everything goes
  through :func:`qwen3_tts_rocm.models.resolve_path`;
* ``--models-dir`` is the CLI alternative to setting
  ``$QWEN3_TTS_ROCM_MODELS_DIR``;
* loading goes through :class:`~qwen3_tts_rocm.demo.backend.SynthesisService`
  so the UI model switcher keeps exactly ONE model resident (LRU of one);
* ``--concurrency`` defaults to **1** on purpose: single-GPU unified-memory
  queue serializes generation (the upstream default of 16 makes no sense on
  one gfx1151 APU);
* SSR stays disabled (server machines have no node runtime).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

from . import models
from .demo.backend import SynthesisService

__all__ = ["build_parser", "main", "resolve_target"]

#: Aliases offered by the UI switcher radio (usable TTS models only).
USABLE_ALIASES: tuple[str, ...] = tuple(a for a in models.ALIASES if a != "tokenizer")


def _dtype_from_str(s: str):
    """Same accepted spellings as the official demo's dtype parser."""
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch_dtype("bfloat16")
    if s in ("fp16", "float16", "half"):
        return torch_dtype("float16")
    if s in ("fp32", "float32"):
        return torch_dtype("float32")
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def torch_dtype(name: str):
    """Resolve a dtype name WITHOUT importing torch at module scope."""
    import torch  # lazy; heavy ROCm wheel

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_parser() -> argparse.ArgumentParser:
    """CLI surface compatible with the upstream demo plus our extensions."""
    parser = argparse.ArgumentParser(
        prog="qwen3-tts-rocm-demo",
        description=(
            "Launch the enhanced bilingual Gradio demo for Qwen3-TTS on AMD "
            "ROCm (gfx1151).\n\n"
            "Examples:\n"
            "  qwen3-tts-rocm-demo --alias custom-voice --port 8000\n"
            "  qwen3-tts-rocm-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen3-tts-rocm-demo /path/to/local/model --device cpu --dtype float32\n"
            "  qwen3-tts-rocm-demo --alias base-0.6b --concurrency 1\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Positional checkpoint/alias (also supports -c/--checkpoint or --alias)
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help=(f"Model alias, local path or HuggingFace repo id (positional); one of {USABLE_ALIASES}."),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Checkpoint path or official repo id (wins over --alias).",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="Registry alias of the model to load first (default: custom-voice).",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help=(
            "Root directory containing the downloaded model folders "
            "(alternative to $QWEN3_TTS_ROCM_MODELS_DIR)."
        ),
    )

    # Model loading args (mapped onto loader smart defaults)
    parser.add_argument(
        "--device",
        default=None,
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: auto-picked).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Request FlashAttention-2 (default: off -- AMD ships no ROCm "
            "flash-attn wheel; HIP GPUs use sdpa automatically)."
        ),
    )

    # Gradio server args
    parser.add_argument("--ip", default="0.0.0.0", help="Server bind IP (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000).")
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Gradio queue concurrency (default: 1 -- the single-GPU "
            "unified-memory queue serializes generation)."
        ),
    )

    # HTTPS args
    parser.add_argument("--ssl-certfile", default=None, help="SSL certificate file (optional).")
    parser.add_argument("--ssl-keyfile", default=None, help="SSL key file (optional).")
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Verify SSL certificates (default: enabled).",
    )

    # Optional generation-defaults args (pre-fill the Advanced accordion)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Default max new tokens (optional; service guardrail 512).",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument(
        "--subtalker-top-k", type=int, default=None, help="Subtalker top-k (only for tokenizer v2)."
    )
    parser.add_argument(
        "--subtalker-top-p", type=float, default=None, help="Subtalker top-p (only for tokenizer v2)."
    )
    parser.add_argument(
        "--subtalker-temperature",
        type=float,
        default=None,
        help="Subtalker temperature (only for tokenizer v2).",
    )
    return parser


def _registry_alias_for(text: str) -> str | None:
    """Registry alias behind *text* (alias, repo id or flat name); else None."""
    if text in models.REPOS:
        return text
    try:
        return models.alias_of_repo(models.flatten(text))
    except KeyError:
        return None


def resolve_target(args: argparse.Namespace) -> tuple[str, str | None, str]:
    """Interpret the checkpoint/alias arguments.

    Returns ``(model_ref, alias_or_None, mode)`` where *mode* is either

    * ``"registry"`` -- *model_ref* resolves through the registry per-alias;
      the second element names the ALIAS involved: either the explicit alias,
      or one recovered from a known repo id / flattened name (then used as the
      switcher's initial selection);
    * ``"pinned"``   -- an arbitrary local directory/user path: every switcher
      selection loads THIS reference instead (second element is ``None``).
    """
    ckpt = args.checkpoint or args.checkpoint_pos
    alias = getattr(args, "alias", None)

    if ckpt:
        resolved_alias = _registry_alias_for(str(ckpt))
        if resolved_alias is None:
            return str(ckpt), None, "pinned"
        return str(ckpt), resolved_alias, "registry"

    chosen = str(alias or "custom-voice")
    if _registry_alias_for(chosen) is None:
        return chosen, None, "pinned"
    return chosen, chosen, "registry"


def _collect_default_gen_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _build_service(args: argparse.Namespace, model_ref: str, mode: str) -> SynthesisService:
    """SynthesisService whose factory bakes in the loader CLI overrides.

    In ``"pinned"`` mode every switcher selection loads *model_ref* (the
    user's explicit path); in ``"registry"`` mode each alias resolves through
    the registry as usual.
    """
    device = args.device
    dtype = _dtype_from_str(args.dtype)
    attn = "flash_attention_2" if args.flash_attn else None

    def factory(alias: str):
        from . import loader

        ref = str(model_ref) if mode == "pinned" else str(alias)
        return loader.load(ref, device=device, dtype=dtype, attn_implementation=attn)

    return SynthesisService(factory=factory)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, assemble service + UI and serve until interrupted."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.checkpoint or args.checkpoint_pos or args.alias):
        # Enhanced-demo UX: bare launch (notably `docker run ...`) starts serving
        # the default preset-voice model instead of printing help. Pass
        # --help explicitly for usage; the untouched upstream CLI keeps its own
        # help-and-exit behavior (parity-certified separately).
        args.alias = "custom-voice"

    if getattr(args, "models_dir", None):
        os.environ[models.MODELS_DIR_ENV] = str(args.models_dir)

    try:
        model_ref, pinned_alias, mode = resolve_target(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    service = _build_service(args, model_ref, mode)

    from .demo.ui import build_ui, launch_visual_kwargs

    header_info: dict[str, Any] = {
        "alias": pinned_alias or model_ref,
        "checkpoint": str(model_ref),
    }
    app = build_ui(service, header_info)

    launch_kwargs: dict[str, Any] = {
        "server_name": args.ip,
        "server_port": args.port,
        "share": bool(args.share),
        "ssl_verify": bool(args.ssl_verify),
        "ssr_mode": False,  # server hosts ship without node.js
    }
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    print(
        f"[qwen3-tts-rocm] target={model_ref!r} mode={mode} "
        f"alias={header_info['alias']} device={args.device or 'auto'}"
    )
    try:
        app.queue(default_concurrency_limit=max(int(args.concurrency), 1))
        # Gradio 6 carries the official Soft theme / full-width css on launch().
        app.launch(**launch_visual_kwargs(), **launch_kwargs)
    except KeyboardInterrupt:
        pass  # Ctrl-C is a normal shutdown path
    finally:
        service.unload_all()  # release unified memory on exit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
