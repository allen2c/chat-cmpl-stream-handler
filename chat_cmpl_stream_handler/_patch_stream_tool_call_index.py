"""Monkey-patch for openai SDK streaming: fix providers that return
``tool_call_delta.index = None`` (e.g. Gemini OpenAI-compat endpoint).

The openai SDK assumes ``tool_call_delta.index`` is always an ``int``,
but some providers (notably Gemini's OpenAI-compatible API) send
``None``.  This patch normalises the index to its positional order
before the SDK processes the chunk.

Usage — call ``apply()`` once before any streaming request::

    from chat_cmpl_stream_handler._patch_stream_tool_call_index import apply
    apply()

It is safe to call ``apply()`` multiple times; only the first call
takes effect.  The patch must be applied before
``ChatCompletionStreamState`` instances are created (i.e. before
``openai_client.beta.chat.completions.stream()`` is called).
"""

from __future__ import annotations

import logging

from openai.lib.streaming.chat._completions import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk

logger = logging.getLogger(__name__)

_PATCHED = False
_original_handle_chunk = ChatCompletionStreamState.handle_chunk


def _fix_none_tool_call_indices(chunk: ChatCompletionChunk) -> None:
    """Mutate *chunk* in-place so every ``tool_call.index`` is an int.

    When a provider omits the index (sends ``None``), we fall back to the
    positional order of the tool-call deltas within the choice, which is the
    only sane default.
    """
    for choice in chunk.choices:
        if not choice.delta.tool_calls:
            continue
        for pos, tc in enumerate(choice.delta.tool_calls):
            if tc.index is None:
                tc.index = pos


def _patched_handle_chunk(self, chunk: ChatCompletionChunk):
    _fix_none_tool_call_indices(chunk)
    return _original_handle_chunk(self, chunk)


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    ChatCompletionStreamState.handle_chunk = _patched_handle_chunk
    _PATCHED = True
    logger.debug(
        "Patched ChatCompletionStreamState.handle_chunk" " for None tool_call index"
    )
