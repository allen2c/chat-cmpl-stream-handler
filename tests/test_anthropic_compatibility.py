"""Tests for AnthropicOpenAI compatibility with stream_until_user_input."""

import os
from typing import Any

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from chat_cmpl_stream_handler import (
    ChatCompletionStreamHandler,
    StreamResult,
    args_from_tool_call,
    stream_until_user_input,
)
from chat_cmpl_stream_handler._anthropic import AnthropicOpenAI

GET_WEATHER_TOOL = {
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
        },
    },
}


async def get_weather_invoker(
    tool_call: ChatCompletionMessageToolCall, context: Any
) -> str:
    args = args_from_tool_call(tool_call)
    return f"The weather in {args['city']} is sunny and 25°C."


@pytest.fixture(scope="session")
def anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")
    return AnthropicOpenAI(api_key=api_key)


@pytest.mark.asyncio
async def test_anthropic_simple_text(anthropic_client: AnthropicOpenAI):
    """Basic text response — no tools."""
    result = await stream_until_user_input(
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        model="claude-haiku-4-5-20251001",
        openai_client=anthropic_client,
        stream_handler=ChatCompletionStreamHandler(),
    )

    assert isinstance(result, StreamResult)
    msgs = result.to_input_list()
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"]  # should have some text


@pytest.mark.asyncio
async def test_anthropic_tool_call(anthropic_client: AnthropicOpenAI):
    """Tool call loop: user → assistant (tool_call) → tool → assistant."""
    result = await stream_until_user_input(
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        model="claude-haiku-4-5-20251001",
        openai_client=anthropic_client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers={"get_weather": get_weather_invoker},
        stream_kwargs={"tools": [GET_WEATHER_TOOL]},
    )

    assert isinstance(result, StreamResult)
    msgs = result.to_input_list()
    roles = [m["role"] for m in msgs]

    # Expect: user → assistant (with tool_calls) → tool → assistant
    assert roles[0] == "user"
    assert "assistant" in roles
    assert "tool" in roles
    assert roles[-1] == "assistant"

    # Usage should be tracked
    assert len(result.usages) > 0
    assert all(u.total_tokens > 0 for u in result.usages)
