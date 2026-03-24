import json
from typing import Any

import pytest
from openai.types.chat import ChatCompletionToolParam

from chat_cmpl_stream_handler import (
    ChatCompletionStreamHandler,
    StreamResult,
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


async def get_weather_invoker(arguments: str, context: Any) -> str:
    assert context == "test"
    args = json.loads(arguments)
    return f"The weather in {args['city']} is sunny and 25°C."


@pytest.mark.asyncio
async def test_stream_until_user_input_with_tool_call(llm_provider: LLMProvider):
    openai_client = llm_provider.client
    model = llm_provider.model

    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    result = await stream_until_user_input(
        messages=messages,
        model=model,
        openai_client=openai_client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers={"get_weather": get_weather_invoker},
        stream_kwargs={
            "tools": [GET_WEATHER_TOOL],
            "stream_options": {"include_usage": True},
        },
        context="test",
    )

    assert isinstance(result, StreamResult)

    input_list = result.to_input_list()
    roles = [msg["role"] for msg in input_list]

    # user → assistant (tool_calls) → tool → assistant (final)
    assert roles == ["user", "assistant", "tool", "assistant"]

    # Check if the usages are not empty
    assert len(result.usages) > 0
    assert all(u.total_tokens for u in result.usages)
