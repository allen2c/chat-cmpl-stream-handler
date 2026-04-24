from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from chat_cmpl_stream_handler import (
    ChatCompletionStreamHandler,
    FunctionTool,
    StreamResult,
    ToolResult,
    args_from_tool_call,
    stream_until_user_input,
)
from tests.conftest import LLMProvider

GET_WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


async def get_weather_invoker(
    tool_call: ChatCompletionMessageToolCall, context: Any
) -> str:
    assert context == "test"
    args = args_from_tool_call(tool_call)
    return f"The weather in {args['city']} is sunny and 25°C."


async def get_weather_tool_result_invoker(
    tool_call: ChatCompletionMessageToolCall, context: Any
) -> ToolResult:
    assert context == "test"
    args = args_from_tool_call(tool_call)
    return ToolResult(
        content=f"hi from {args['city']}",
        metadata={"city": args["city"]},
    )


async def failing_weather_invoker(
    _tool_call: ChatCompletionMessageToolCall, _context: Any
) -> str:
    raise RuntimeError("weather unavailable")


@pytest.mark.parametrize("via", ["dict", "tool"])
@pytest.mark.asyncio
async def test_stream_until_user_input_with_tool_call(
    llm_provider: LLMProvider, via: str
):
    openai_client = llm_provider.client
    model = llm_provider.model

    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    if via == "dict":
        extra_kwargs: dict[str, Any] = dict(
            tool_invokers={"get_weather": get_weather_invoker},
            stream_kwargs={
                "tools": [GET_WEATHER_TOOL],
                "stream_options": {"include_usage": True},
            },
        )
    else:
        extra_kwargs = dict(
            tools=[
                FunctionTool(
                    tool_param=GET_WEATHER_TOOL,
                    invoker=get_weather_invoker,
                )
            ],
            stream_kwargs={"stream_options": {"include_usage": True}},
        )

    result = await stream_until_user_input(
        messages=messages,
        model=model,
        openai_client=openai_client,
        stream_handler=ChatCompletionStreamHandler(),
        context="test",
        **extra_kwargs,
    )

    assert isinstance(result, StreamResult)

    input_list = result.to_input_list()
    roles = [msg["role"] for msg in input_list]

    # user → assistant (tool_calls) → tool → assistant (final)
    assert roles == ["user", "assistant", "tool", "assistant"]

    # Check if the usages are not empty
    assert len(result.usages) > 0
    assert all(u.total_tokens for u in result.usages)


@pytest.mark.asyncio
async def test_stream_until_user_input_callback_receives_tool_result_content(
    openai_client: AsyncOpenAI,
    openai_model: str,
):
    callback_outputs: list[str] = []

    async def capture_output(
        _tool_call: ChatCompletionMessageToolCall, output: str
    ) -> None:
        callback_outputs.append(output)

    result = await stream_until_user_input(
        messages=[
            {
                "role": "user",
                "content": (
                    "Call get_weather once with city Tokyo, then answer using "
                    "the tool result."
                ),
            }
        ],
        model=openai_model,
        openai_client=openai_client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers={"get_weather": get_weather_tool_result_invoker},
        stream_kwargs={
            "tools": [GET_WEATHER_TOOL],
            "parallel_tool_calls": False,
        },
        context="test",
        tool_call_output_callback=capture_output,
    )

    input_list = result.to_input_list()
    tool_messages = [message for message in input_list if message["role"] == "tool"]

    assert callback_outputs == ["hi from Tokyo"]
    assert tool_messages[0]["content"] == "hi from Tokyo"


@pytest.mark.asyncio
async def test_stream_until_user_input_emit_tool_error_returns_result(
    openai_client: AsyncOpenAI,
    openai_model: str,
):
    result = await stream_until_user_input(
        messages=[
            {
                "role": "user",
                "content": (
                    "Call get_weather once with city Tokyo, then answer using "
                    "the tool result even if the tool reports an error."
                ),
            }
        ],
        model=openai_model,
        openai_client=openai_client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers={"get_weather": failing_weather_invoker},
        stream_kwargs={
            "tools": [GET_WEATHER_TOOL],
            "parallel_tool_calls": False,
        },
    )

    assert isinstance(result, StreamResult)
    assert any(
        message["role"] == "tool" and message["content"] == "Tool invocation failed."
        for message in result.to_input_list()
    )


@pytest.mark.asyncio
async def test_stream_until_user_input_raise_tool_error_propagates(
    openai_client: AsyncOpenAI,
    openai_model: str,
):
    with pytest.raises(RuntimeError, match="weather unavailable"):
        await stream_until_user_input(
            messages=[
                {
                    "role": "user",
                    "content": "Call get_weather once with city Tokyo.",
                }
            ],
            model=openai_model,
            openai_client=openai_client,
            stream_handler=ChatCompletionStreamHandler(),
            tool_invokers={"get_weather": failing_weather_invoker},
            stream_kwargs={
                "tools": [GET_WEATHER_TOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "parallel_tool_calls": False,
            },
            on_tool_error="raise",
        )
