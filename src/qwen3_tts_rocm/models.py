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

Downloading
-----------
:func:`download` fetches the official repositories over the network.  Per the
spike findings (``evidence/spike/spike-report.md``), ModelScope is the proven,
working artifact channel on this host, so ``source="modelscope"`` (and the
default ``"auto"``) talks to ModelScope first; ``"auto"`` additionally falls
back to the hf-mirror endpoint (``HF_ENDPOINT=https://hf-mirror.com`` around
the ``huggingface_hub.snapshot_download`` call) when ModelScope fails.  Each
source is retried once before moving on; once every attempt fails a
:class:`RuntimeError` lists both manual download URLs.  Already-downloaded
targets (``config.json`` or ``.ok`` present, see :func:`is_downloaded`) are
skipped, and each successful fetch (or skip) yields its target directory in
the returned list, in alias order.

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
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "ALIASES",
    "MODELS_DIR_ENV",
    "REPOS",
    "SUPPORTED_SOURCES",
    "alias_of_repo",
    "download",
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


# ---------------------------------------------------------------------------
# Downloader (Task 5): official repos via ModelScope-first with hf-mirror
# fallback.  Network transports live behind the module-level _ms_snapshot /
# _hf_snapshot indirection points so tests can stub them out hermetically.
# ---------------------------------------------------------------------------

#: Transports understood by :func:`download` (excluding the ``"auto"`` mode
#: which chains both of these, ModelScope first).
SUPPORTED_SOURCES = ("modelscope", "hfmirror")

#: Mirror endpoint used by the hf-mirror transport.
_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

_ATTEMPTS_PER_SOURCE = 2  # initial try + one retry


def _ms_snapshot(model_id: str, local_dir: Path) -> None:
    """Fetch *model_id* into *local_dir* via the ModelScope SDK.

    Per the spike report, ``from modelscope import snapshot_download`` is the
    proven channel on this host (github.com / huggingface.co are unreachable,
    modelscope.cn works).  The import is deferred to call time so merely using
    the registry does not pay for importing the SDK.  Interrupted transfers
    resume natively on re-invocation.
    """
    from modelscope import snapshot_download

    snapshot_download(str(model_id), local_dir=str(local_dir))


def _hf_snapshot(repo_id: str, local_dir: Path) -> None:
    """Fetch *repo_id* into *local_dir* through the hf-mirror endpoint.

    ``HF_ENDPOINT`` is set to the mirror around the
    ``huggingface_hub.snapshot_download`` call and restored afterwards; the env
    var must be in place before huggingface_hub is imported because its
    constants module binds the endpoint at import time.
    """
    saved = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = _HF_MIRROR_ENDPOINT
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(str(repo_id), local_dir=str(local_dir))
    finally:
        if saved is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = saved


def _resolve_aliases(aliases: str | Iterable[str]) -> tuple[str, ...]:
    """Normalise the *aliases* argument into a validated alias tuple."""
    if aliases == "all":
        return ALIASES
    names: tuple[str, ...]
    if isinstance(aliases, str):
        names = (aliases,)
    else:
        names = tuple(aliases)
    for name in names:
        if name not in REPOS:
            raise KeyError(
                f"unknown alias {str(name)!r} (未知的模型别名); known aliases: {ALIASES}"
            )
    return names


def _target_for(alias: str, models_dir: str | Path | None) -> Path:
    """Target directory for an alias; an explicit *models_dir* wins over roots."""
    root = Path(models_dir).expanduser() if models_dir is not None else local_dir()
    return root / flatten(REPOS[alias])


def _download_via(repo_id: str, target: Path, sources: Iterable[str]) -> None:
    """Attempt *sources* (in order, one retry each); write ``.ok`` on success.

    Raises RuntimeError naming every attempt plus both manual download URLs
    once all attempts are exhausted.  Each transport is invoked through its
    indirection point with its canonical signature (repo id positional;
    ``local_dir`` as keyword for ModelScope) so test doubles can stub either.
    """
    failures: list[str] = []
    for source in sources:
        for attempt in range(1, _ATTEMPTS_PER_SOURCE + 1):
            try:
                if source == "modelscope":
                    _ms_snapshot(repo_id, local_dir=target)
                else:
                    _hf_snapshot(repo_id, target)
                # mark_ok() writes its .ok marker into the resolved target;
                # the existing-directory passthrough that accepts a Path ref
                # lives in resolve_path(), and a missing path falls through to
                # registry lookup and raises KeyError — so the target dir must
                # be in place before mark_ok() is called.
                target.mkdir(parents=True, exist_ok=True)
                mark_ok(target)
                return
            except Exception as exc:  # noqa: BLE001 - aggregated into RuntimeError below
                failures.append(f"{source} attempt {attempt}: {exc!r}")
    raise RuntimeError(
        f"download failed for {repo_id!r} after {', '.join(sources)}"
        f" (模型下载失败，已尝试 {', '.join(sources)}):\n  - "
        + "\n  - ".join(failures)
        + "\nManual fix (手动下载): place the files under "
        f"{target}\n  https://modelscope.cn/models/{repo_id}\n"
        f"  https://hf-mirror.com/{repo_id}"
    )


def download(
    aliases: str | Iterable[str] = "all",
    source: str = "auto",
    models_dir: str | Path | None = None,
    resume: bool = True,
) -> list[Path]:
    """Download one or more official Qwen3-TTS model repositories.

    Parameters
    ----------
    aliases:
        ``"all"`` for every registered repo, a single alias string such as
        ``"tokenizer"``, or an iterable of aliases (returned paths keep this
        order).  Unknown aliases raise :class:`KeyError`.
    source:
        ``"auto"`` (default) tries ModelScope first and falls back to hf-mirror
        per alias; or one of :data:`SUPPORTED_SOURCES`.  An explicit source is
        retried once but never falls back; exhausting every attempt raises
        :class:`RuntimeError` listing both manual URLs.
    models_dir:
        Override for the models root directory; takes precedence over
        ``$QWEN3_TTS_ROCM_MODELS_DIR`` and the cwd/cache default (see
        :func:`local_dir`).  Defaults to the ambient root when omitted.
    resume:
        Both transports resume interrupted transfers natively, so partial
        downloads continue where they left off.  Accepted for API stability;
        passing ``False`` currently behaves identically (reserved for a future
        clean-restart mode).

    Already-downloaded targets (:func:`is_downloaded`: ``config.json`` or
    ``.ok`` present) are skipped without touching the network.  Returns the
    list of target directories (downloaded or skipped) in requested order.
    """
    names = _resolve_aliases(aliases)
    if source == "auto":
        sources = SUPPORTED_SOURCES
    elif source in SUPPORTED_SOURCES:
        sources = (source,)
    else:
        raise ValueError(
            f"unknown source {source!r} (未知的下载源); "
            f"expected 'auto' or one of {SUPPORTED_SOURCES}"
        )

    results: list[Path] = []
    for name in names:
        target = _target_for(name, models_dir)
        # Only consult is_downloaded() on an existing directory: resolve_path()
        # passes existing dirs through verbatim, guaranteeing the check hits
        # the effective root instead of some unrelated env/cwd-derived location.
        if target.is_dir() and is_downloaded(target):
            results.append(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        _download_via(REPOS[name], target, sources)
        results.append(target)
    return results
