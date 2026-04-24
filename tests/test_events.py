"""Pure unit tests for lifecycle event dataclasses."""

from dataclasses import FrozenInstanceError

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from chat_cmpl_stream_handler import StreamResult
from chat_cmpl_stream_handler.events import (
    IterationCompleted,
    IterationStarted,
    RunCompleted,
    RunFailed,
    StreamEvent,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    ToolResult,
)


def _tool_call() -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id="call_1",
        function=Function(arguments="{}", name="test_tool"),
        type="function",
    )


@pytest.mark.parametrize(
    "event",
    [
        ToolResult(content="ok", metadata={"x": 1}),
        IterationStarted(index=0, messages=[]),
        StreamEvent(event=object()),
        ToolCallStarted(iteration=0, tool_call=_tool_call()),
        ToolCallCompleted(
            iteration=0,
            tool_call=_tool_call(),
            result=ToolResult(content="ok"),
        ),
        ToolCallFailed(
            iteration=0,
            tool_call=_tool_call(),
            exception=RuntimeError("boom"),
        ),
        IterationCompleted(
            index=0,
            usage=None,
            assistant_message={"role": "assistant", "content": "hi"},
        ),
        RunCompleted(result=StreamResult([], "gpt-test", [])),
        RunFailed(exception=RuntimeError("boom")),
    ],
)
def test_lifecycle_events_are_frozen(event):
    field_name = next(iter(event.__dataclass_fields__))

    with pytest.raises(FrozenInstanceError):
        setattr(event, field_name, "changed")


def test_tool_result_fields_round_trip():
    result = ToolResult(content="hello", metadata={"trace_id": "abc"})

    assert result.content == "hello"
    assert result.metadata == {"trace_id": "abc"}
