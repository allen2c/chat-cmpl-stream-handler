from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from chat_cmpl_stream_handler import (
    IterationCompleted,
    IterationStarted,
    RunCompleted,
    RunFailed,
    StreamEvent,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    ToolResult,
    args_from_tool_call,
    stream_until_user_input_events,
)

GET_WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

FOO_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "foo__bar",
        "description": "Return a test payload.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
}


async def weather_result(
    tool_call: ChatCompletionMessageToolCall, _context: Any
) -> ToolResult:
    args = args_from_tool_call(tool_call)
    return ToolResult(
        content=f"weather:{args['city']}",
        metadata={"city": args["city"]},
    )


async def failing_tool(_tool_call: ChatCompletionMessageToolCall, _context: Any) -> str:
    raise RuntimeError("tool exploded")


async def fallback_tool(
    _tool_call: ChatCompletionMessageToolCall, _context: Any
) -> str:
    return "fallback ok"


@pytest.mark.asyncio
async def test_events_no_tools(openai_client: AsyncOpenAI, openai_model: str):
    events = await _events(
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        model=openai_model,
        openai_client=openai_client,
        stream_kwargs={"stream_options": {"include_usage": True}},
    )

    assert isinstance(events[0], IterationStarted)
    assert any(
        isinstance(e, StreamEvent) and e.event.type == "content.delta" for e in events
    )
    assert any(
        isinstance(e, StreamEvent) and e.event.type == "content.done" for e in events
    )
    assert any(isinstance(e, IterationCompleted) and e.usage for e in events)
    assert isinstance(events[-1], RunCompleted)
    assert not any(isinstance(e, ToolCallStarted) for e in events)


@pytest.mark.asyncio
async def test_events_tool_result_metadata(
    openai_client: AsyncOpenAI, openai_model: str
):
    events = await _events(
        messages=[
            {
                "role": "user",
                "content": "Call get_weather once with city Tokyo, then answer.",
            }
        ],
        model=openai_model,
        openai_client=openai_client,
        tool_invokers={"get_weather": weather_result},
        stream_kwargs={"tools": [GET_WEATHER_TOOL], "parallel_tool_calls": False},
    )

    completed = _one(events, ToolCallCompleted)
    assert completed.result == ToolResult(
        content="weather:Tokyo",
        metadata={"city": "Tokyo"},
    )
    assert [m["role"] for m in events[-1].result.to_input_list()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["emit", "abort", "raise"])
async def test_events_tool_error_modes(
    openai_client: AsyncOpenAI,
    openai_model: str,
    mode: str,
):
    stream_kwargs = {"tools": [GET_WEATHER_TOOL], "parallel_tool_calls": False}
    if mode != "emit":
        stream_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    kwargs = dict(
        messages=[
            {
                "role": "user",
                "content": (
                    "Call get_weather once with city Tokyo. If it reports an "
                    "error, answer without calling tools again."
                ),
            }
        ],
        model=openai_model,
        openai_client=openai_client,
        tool_invokers={"get_weather": failing_tool},
        stream_kwargs=stream_kwargs,
        on_tool_error=mode,
    )

    if mode == "raise":
        seen = []
        with pytest.raises(RuntimeError, match="tool exploded"):
            async for event in stream_until_user_input_events(**kwargs):
                seen.append(event)
        assert any(isinstance(e, ToolCallFailed) for e in seen)
        return

    events = await _events(**kwargs)
    assert isinstance(_one(events, ToolCallFailed).exception, RuntimeError)
    if mode == "emit":
        assert (
            _one(events, ToolCallCompleted).result.content == "Tool invocation failed."
        )
        assert isinstance(events[-1], RunCompleted)
    else:
        assert isinstance(events[-1], RunFailed)


@pytest.mark.asyncio
async def test_events_fallback_invoker(openai_client: AsyncOpenAI, openai_model: str):
    events = await _events(
        messages=[{"role": "user", "content": "Call foo__bar once, then answer."}],
        model=openai_model,
        openai_client=openai_client,
        stream_kwargs={"tools": [FOO_TOOL], "parallel_tool_calls": False},
        fallback_invoker=lambda name: fallback_tool if name == "foo__bar" else None,
    )

    assert _one(events, ToolCallCompleted).result.content == "fallback ok"


@pytest.mark.asyncio
async def test_events_missing_invoker_stays_strict(
    openai_client: AsyncOpenAI,
    openai_model: str,
):
    with pytest.raises(ValueError, match="No invoker"):
        await _events(
            messages=[{"role": "user", "content": "Call foo__bar."}],
            model=openai_model,
            openai_client=openai_client,
            stream_kwargs={"tools": [FOO_TOOL]},
        )


async def _events(**kwargs: Any) -> list[Any]:
    return [event async for event in stream_until_user_input_events(**kwargs)]


def _one(events: list[Any], event_type: type) -> Any:
    found = [event for event in events if isinstance(event, event_type)]
    assert len(found) == 1
    return found[0]
