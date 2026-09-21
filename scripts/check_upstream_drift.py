#!/usr/bin/env python
"""Detect upstream Qwen3-TTS drift vs docs/upstream-baseline.json.

Checks: upstream main SHA, qwen-tts PyPI version, repo file tree
(checkpoints/finetuning), finetuning/sft_12hz.py blob hash, README.md
(vLLM instructions) hash, and the installed package's public generate API
signatures. Never marks new upstream capability as Radeon-compatible --
drift output only demands revalidation.

Exit-code convention (also stated in ``--help`` and pinned by
tests/test_check_upstream_drift.py):

    0  UNCHANGED -- baseline matches the current upstream state
    1  UPSTREAM DRIFT DETECTED -- revalidation required (never auto-green)
    2  NETWORK/COLLECTION ERROR -- drift could NOT be determined; retry later

Network policy (stdlib urllib only, no new dependencies):

* The GitHub API is accessed anonymously (no gh-token fallback -- anonymous
  access validated on the reference host; api.github.com rate limits apply
  and surface as exit 2, not as a false UNCHANGED).
* Every HTTP GET retries exactly once after a transient failure; the known
  github.com:443 GnuTLS flakiness of the reference host is why.
* Reference-host quirk, not a script defect: DNS for pypi.org returns IPv6
  addresses whose TLS handshake black-holes, so urllib spends its timeout on
  each address before the IPv4 fallback -- the PyPI probe can take ~2 min
  there. Slow is not failed; only a hard error after the retry exits 2.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import urllib.request
from pathlib import Path

UPSTREAM = "QwenLM/Qwen3-TTS"
PYPI = "https://pypi.org/pypi/qwen-tts/json"

EXIT_UNCHANGED = 0
EXIT_DRIFT = 1
EXIT_NETWORK = 2


class NetworkError(RuntimeError):
    """A network fetch failed even after its one in-script retry."""


def _get(url: str):
    """GET url -> bytes; one retry on transient failure, then NetworkError."""
    req = urllib.request.Request(url, headers={"User-Agent": "qwen3-tts-rocm-drift-check"})
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except OSError as exc:  # URLError/HTTPError/TimeoutError are all OSError
            last = exc
            if attempt == 1:
                time.sleep(2)
    raise NetworkError(f"GET {url} failed after retry: {type(last).__name__}: {last}") from last


def _gh_api(path: str):
    return json.loads(_get(f"https://api.github.com/repos/{UPSTREAM}/{path}"))


def upstream_sha() -> str:
    return _gh_api("commits/main")["sha"]


def pypi_version() -> str:
    return json.loads(_get(PYPI))["info"]["version"]


def repo_file_hashes() -> dict[str, str]:
    tree = _gh_api("git/trees/main?recursive=1")["tree"]
    return {e["path"]: e.get("sha", "") for e in tree if e["type"] == "blob"}


def generate_api_surface() -> dict[str, str]:
    from qwen_tts import Qwen3TTSModel
    out = {}
    for name in dir(Qwen3TTSModel):
        if name.startswith("generate") or name in {"create_voice_clone_prompt"}:
            try:
                out[name] = str(inspect.signature(getattr(Qwen3TTSModel, name)))
            except (TypeError, ValueError):
                out[name] = "<no-signature>"
    return out


def collect() -> dict:
    return {
        "upstream_sha": upstream_sha(),
        "pypi_version": pypi_version(),
        "file_hashes": repo_file_hashes(),
        "api_surface": generate_api_surface(),
    }


def diff_state(baseline: dict, current: dict) -> dict:
    d = {}
    for key in ("upstream_sha", "pypi_version"):
        if baseline.get(key) != current[key]:
            d[key] = {"baseline": baseline.get(key), "current": current[key]}
    for key in ("file_hashes", "api_surface"):
        old, new = baseline.get(key, {}), current[key]
        if set(old) != set(new) or any(old[k] != new[k] for k in old):
            d[key] = {
                "added": sorted(set(new) - set(old)),
                "removed": sorted(set(old) - set(new)),
                "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
            }
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Detect upstream Qwen3-TTS drift vs a committed baseline "
        "(SHA / PyPI version / repo tree hashes / installed generate-API signatures).",
        epilog="exit codes: 0 = UNCHANGED, 1 = UPSTREAM DRIFT DETECTED "
        "(revalidation required), 2 = network/collection error "
        "(drift undetermined -- retry later).",
    )
    ap.add_argument("--baseline", default="docs/upstream-baseline.json")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the CURRENT upstream state as the new baseline and exit 0")
    args = ap.parse_args(argv)
    try:
        current = collect()
    except NetworkError as exc:
        print(f"NETWORK ERROR (exit 2 -- drift NOT determined): {exc}", file=sys.stderr)
        return EXIT_NETWORK
    except Exception as exc:  # noqa: BLE001 - any collection failure must exit 2, never crash or false-green
        print(f"COLLECTION ERROR (exit 2 -- drift NOT determined): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_NETWORK
    if args.update_baseline:
        Path(args.baseline).write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print("BASELINE UPDATED")
        return EXIT_UNCHANGED
    try:
        baseline = json.loads(Path(args.baseline).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"BASELINE ERROR (exit 2 -- cannot compare): {args.baseline}: {exc}",
              file=sys.stderr)
        return EXIT_NETWORK
    d = diff_state(baseline, current)
    if not d:
        print("UNCHANGED")
        return EXIT_UNCHANGED
    print("UPSTREAM DRIFT DETECTED — revalidation required (new upstream capability is NOT auto-marked Radeon-compatible):")
    print(json.dumps(d, indent=2))
    return EXIT_DRIFT


if __name__ == "__main__":
    sys.exit(main())
