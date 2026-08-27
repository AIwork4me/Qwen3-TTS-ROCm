"""Model registry: friendly aliases -> official Qwen/Qwen3-TTS repos, plus path resolution.

The registry maps short aliases to the six official Hugging Face-style model ids
(all published under the ``Qwen`` namespace).  Everything downstream -- the
downloader (Task 5) and the loader -- resolves models through this module so
there is a single source of truth for where models live on disk.

Models root
-----------
The directory that contains the downloaded model folders is resolved on every
call (never cached at import time) in this order:

1. ``$QWEN3_TTS_ROCM_MODELS_DIR`` if the environment variable is set;
2. otherwise ``<cwd>/models`` when the current working directory contains a
   ``pyproject.toml`` (i.e. we are inside a checkout of this project);
3. otherwise the per-user cache directory ``~/.cache/qwen3-tts-rocm/models``
   (used for installed deployments outside a project checkout).

A specific model then lives at ``<root>/<flattened-repo-name>``, e.g.
``models/Qwen3-TTS-12Hz-1.7B-Base`` for alias ``base``.

Accepted references for :func:`resolve_path` / :func:`is_downloaded` /
:func:`mark_ok`, in precedence order:

1. an existing directory path (``Path`` or ``str``) is passed through verbatim,
   which allows fully user-managed local copies anywhere on disk;
2. a registry alias such as ``"tokenizer"`` or ``"base-0.6b"``;
3. a flattened repo name such as ``"Qwen3-TTS-12Hz-1.7B-Base"``;
4. a full id such as ``"Qwen/Qwen3-TTS-12Hz-1.7B-Base"``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ALIASES",
    "MODELS_DIR_ENV",
    "REPOS",
    "alias_of_repo",
    "flatten",
    "is_downloaded",
    "local_dir",
    "mark_ok",
    "resolve_path",
]

#: Environment variable overriding the models root directory.
MODELS_DIR_ENV = "QWEN3_TTS_ROCM_MODELS_DIR"

#: Registry of official Qwen3-TTS repositories: alias -> HF-style repo id.
REPOS: dict[str, str] = {
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom-voice-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

#: Friendly aliases, in registry order.
ALIASES: tuple[str, ...] = tuple(REPOS)

def flatten(repo_id: str) -> str:
    """Strip a single leading namespace: ``"Qwen/Foo" -> "Foo"``; bare names pass through."""
    parts = str(repo_id).split("/", 1)
    return parts[1] if len(parts) == 2 else parts[0]


# flattened local folder name -> alias ("Qwen3-TTS-12Hz-1.7B-Base" -> "base")
_FLAT_TO_ALIAS: dict[str, str] = {flatten(repo): alias for alias, repo in REPOS.items()}


def alias_of_repo(repo_id_or_flattened: str) -> str:
    """Reverse lookup: full id (``"Qwen/Foo"``) or flattened name (``"Foo"``) -> alias.

    Raises KeyError naming the known repos when neither matches.
    """
    text = str(repo_id_or_flattened)
    for alias, repo in REPOS.items():
        if repo == text:
            return alias
    flat_alias = _FLAT_TO_ALIAS.get(text)
    if flat_alias is None:
        raise KeyError(
            f"unknown repo id {text!r} (未知的模型仓库); "
            f"known repos: {tuple(REPOS.values())}"
        )
    return flat_alias


def _default_root() -> Path:
    """Fallback models root without the env var: project cwd/<root>/models or user cache."""
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file():
        return cwd / "models"
    return Path.home() / ".cache" / "qwen3-tts-rocm" / "models"


def local_dir(alias: str | None = None) -> Path:
    """Return the models root, or a specific model's directory when *alias* is given.

    With no argument this is just the root resolved from
    ``$QWEN3_TTS_ROCM_MODELS_DIR`` (then ``<project-cwd>/models``, else
    ``~/.cache/qwen3-tts-rocm/models`` -- see module docstring).  With an alias
    it returns ``<root>/<flattened-repo-name>``.  The directory is *not*
    created as a side effect.

    Raises KeyError for unknown aliases.
    """
    env_dir = os.environ.get(MODELS_DIR_ENV)
    root = Path(env_dir).expanduser() if env_dir else _default_root()
    if alias is None:
        return root
    repo_id = REPOS.get(str(alias))
    if repo_id is None:
        raise KeyError(
            f"unknown alias {str(alias)!r} (未知的模型别名); known aliases: {ALIASES}"
        )
    return root / flatten(repo_id)


def resolve_path(ref: str | Path) -> Path:
    """Resolve any accepted reference to a concrete local model directory.

    Precedence: existing directory passthrough, then alias, then flattened repo
    name, then full ``"Qwen/..."`` id.  Raises KeyError listing the accepted
    forms when nothing matches.
    """
    ref_path = Path(ref).expanduser()
    if ref_path.is_dir():
        return ref_path
    text = str(ref)
    if text in REPOS:
        return local_dir(text)
    try:
        return local_dir(alias_of_repo(text))
    except KeyError:
        pass
    raise KeyError(
        f"unknown model reference {text!r} (未知的模型引用); "
        f"accepted forms: an existing directory path, one of the aliases "
        f"{ALIASES}, a flattened repo name such as 'Qwen3-TTS-12Hz-1.7B-Base', "
        f"or a full repo id such as 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'"
    )


def is_downloaded(ref: str | Path) -> bool:
    """True when the resolved target exists and holds ``config.json`` or a ``.ok`` marker.

    Unknown references are simply reported as not downloaded instead of raising.
    """
    try:
        target = resolve_path(ref)
    except KeyError:
        return False
    return target.is_dir() and (
        (target / "config.json").exists() or (target / ".ok").exists()
    )


def mark_ok(ref: str | Path) -> None:
    """Write the ``.ok`` completion marker into the resolved directory.

    Creates the directory tree if needed (the downloader calls this after a
    successful fetch).
    """
    target = resolve_path(ref)
    target.mkdir(parents=True, exist_ok=True)
    (target / ".ok").write_text("", encoding="utf-8")
