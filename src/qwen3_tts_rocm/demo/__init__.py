"""Headless demo-backend subpackage: :mod:`.backend` synthesis service + history.

Everything the Gradio/UI layer needs minus the browser: import here and call
plain Python, no process side effects, no downloads, official API only.
"""

from .backend import (
    DEFAULT_GEN_KWARGS,
    HistoryStore,
    SynthesisService,
    display_map,
    format_error,
    lookup_display,
)

__all__ = [
    "DEFAULT_GEN_KWARGS",
    "HistoryStore",
    "SynthesisService",
    "display_map",
    "format_error",
    "lookup_display",
]
