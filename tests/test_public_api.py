"""The public entry points reject keywords they do not accept.

Both used to end in `**kwargs` and never read it, so `max_iteration=5` — one letter
short — was silently dropped and the caller went on believing a cap was in place.
"""

import inspect
from typing import Any, Callable, Dict

import pytest

from chat_cmpl_stream_handler import (
    stream_until_user_input,
    stream_until_user_input_events,
)
from tests.scripted import ScriptedStreamer, text_turn

ENTRY_POINTS: list[Callable[..., Any]] = [stream_until_user_input, stream_until_user_input_events]

DOCUMENTED_KEYWORDS: Dict[str, Any] = {
    "tools": None,
    "tool_invokers": None,
    "stream_kwargs": None,
    "context": None,
    "max_iterations": 10,
    "fallback_invoker": None,
    "on_tool_error": "emit",
}


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda fn: fn.__name__)
@pytest.mark.parametrize("misspelling", ["max_iteration", "tool_invoker", "on_tool_errors", "streaming_kwargs"])
def test_a_misspelled_keyword_raises(entry_point: Callable[..., Any], misspelling: str):
    # The TypeError lands while binding arguments, so neither a coroutine nor an async
    # generator is ever created — nothing is left un-awaited by this call.
    with pytest.raises(TypeError, match=misspelling):
        entry_point(
            messages=[{"role": "user", "content": "hi"}],
            model="scripted",
            openai_client=ScriptedStreamer(text_turn("done")),
            **{misspelling: 5},
        )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda fn: fn.__name__)
def test_the_documented_keywords_are_still_accepted(entry_point: Callable[..., Any]):
    # Bound rather than called: proving the signature accepts these needs no stream.
    inspect.signature(entry_point).bind(
        messages=[{"role": "user", "content": "hi"}],
        model="scripted",
        openai_client=ScriptedStreamer(text_turn("done")),
        **DOCUMENTED_KEYWORDS,
    )
