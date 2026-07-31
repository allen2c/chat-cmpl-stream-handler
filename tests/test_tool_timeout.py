"""`tool_timeout` caps a single invoker.

Offline: a scripted turn asks for one tool call, and the invoker decides whether it
answers in time. Without a cap, one hanging invoker hangs the whole run with no way out,
and a hang is harder to diagnose than a crash.
"""

import asyncio
from typing import Any, List

import pytest
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from chat_cmpl_stream_handler import (
    OnToolError,
    RunCompleted,
    RunFailed,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallTimeout,
    stream_until_user_input,
    stream_until_user_input_events,
)
from tests.scripted import ScriptedStreamer, text_turn, tool_call_turn

SLEEP_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "sleep",
        "description": "Sleep.",
        "parameters": {"type": "object", "properties": {}},
    },
}

TIMEOUT = 0.02
LONGER_THAN_THE_TIMEOUT = 30.0


class SleepInvoker:
    """An invoker that sleeps, and remembers whether it was cancelled."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self.cancelled = False

    async def __call__(self, _tool_call: ChatCompletionMessageToolCall, _context: Any) -> str:
        try:
            await asyncio.sleep(self._seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "slept"


class SelfTimingOutInvoker:
    """An invoker whose own work times out, well inside the cap."""

    async def __call__(self, _tool_call: ChatCompletionMessageToolCall, _context: Any) -> str:
        raise TimeoutError("the invoker's own HTTP client gave up")


async def run_events(invoker: Any, *, on_tool_error: OnToolError = "emit") -> List[Any]:
    return [
        event
        async for event in stream_until_user_input_events(
            messages=[{"role": "user", "content": "sleep please"}],
            model="scripted",
            openai_client=ScriptedStreamer(tool_call_turn("sleep"), text_turn("awake")),
            tool_invokers={"sleep": invoker},
            stream_kwargs={"tools": [SLEEP_TOOL]},
            tool_timeout=TIMEOUT,
            on_tool_error=on_tool_error,
        )
    ]


@pytest.mark.asyncio
async def test_a_slow_invoker_is_cancelled_and_the_loop_continues():
    invoker = SleepInvoker(LONGER_THAN_THE_TIMEOUT)

    events = await run_events(invoker)

    failure = next(event for event in events if isinstance(event, ToolCallFailed))
    assert isinstance(failure.exception, ToolCallTimeout)
    assert "tool_timeout=0.02s" in str(failure.exception)
    assert invoker.cancelled, "the invoker was left running after the cap fired"

    # on_tool_error="emit": the model is told the tool failed and the run finishes.
    completed = next(event for event in events if isinstance(event, ToolCallCompleted))
    assert completed.result.content == "Tool invocation failed."
    assert isinstance(events[-1], RunCompleted)


@pytest.mark.asyncio
async def test_an_invoker_inside_the_cap_is_untouched():
    invoker = SleepInvoker(0.0)

    events = await run_events(invoker)

    assert not any(isinstance(event, ToolCallFailed) for event in events)
    assert not invoker.cancelled
    completed = next(event for event in events if isinstance(event, ToolCallCompleted))
    assert completed.result.content == "slept"


@pytest.mark.asyncio
async def test_on_tool_error_abort_ends_the_run_on_a_timeout():
    events = await run_events(SleepInvoker(LONGER_THAN_THE_TIMEOUT), on_tool_error="abort")

    assert isinstance(events[-1], RunFailed)
    assert isinstance(events[-1].exception, ToolCallTimeout)


@pytest.mark.asyncio
async def test_on_tool_error_raise_propagates_the_timeout():
    with pytest.raises(ToolCallTimeout):
        await run_events(SleepInvoker(LONGER_THAN_THE_TIMEOUT), on_tool_error="raise")


@pytest.mark.asyncio
async def test_the_invokers_own_timeout_error_is_not_relabelled():
    """A cap that did not fire must not be blamed for someone else's TimeoutError."""
    events = await run_events(SelfTimingOutInvoker())

    failure = next(event for event in events if isinstance(event, ToolCallFailed))
    assert isinstance(failure.exception, TimeoutError)
    assert not isinstance(failure.exception, ToolCallTimeout)
    assert "HTTP client" in str(failure.exception)


@pytest.mark.asyncio
async def test_no_timeout_by_default():
    """The cap is opt-in — omitting it leaves the old, uncapped behaviour in place."""
    invoker = SleepInvoker(0.05)

    result = await stream_until_user_input(
        messages=[{"role": "user", "content": "sleep please"}],
        model="scripted",
        openai_client=ScriptedStreamer(tool_call_turn("sleep"), text_turn("awake")),
        tool_invokers={"sleep": invoker},
        stream_kwargs={"tools": [SLEEP_TOOL]},
    )

    assert not invoker.cancelled
    assert [message["role"] for message in result.to_input_list()] == ["user", "assistant", "tool", "assistant"]
