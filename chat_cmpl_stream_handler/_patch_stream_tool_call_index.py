"""Retired monkey-patch — kept so existing ``apply()`` calls keep working.

The fix it carried (providers such as Gemini's OpenAI-compatible endpoint sending
``tool_call_delta.index = None``) now lives in
:class:`chat_cmpl_stream_handler.streamers.OpenAIStreamer`, where it is local, testable,
and applies without mutating the OpenAI SDK.

``apply()`` is a no-op as of 0.6.0 and will be removed in a later release. Delete the
call; the loop normalises indices on its own.
"""

from __future__ import annotations

import warnings


def apply() -> None:
    """Do nothing. See the module docstring."""
    warnings.warn(
        "_patch_stream_tool_call_index.apply() is a no-op as of 0.6.0 — "
        "tool-call index normalisation now happens inside the stream adapter. "
        "Remove the call.",
        DeprecationWarning,
        stacklevel=2,
    )
