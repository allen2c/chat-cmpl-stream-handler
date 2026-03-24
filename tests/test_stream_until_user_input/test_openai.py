import json
from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam

from chat_cmpl_stream_handler import (
    ChatCompletionStreamHandler,
    StreamResult,
    stream_until_user_input,
)

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
    args = json.loads(arguments)
    return f"The weather in {args['city']} is sunny and 25°C."


@pytest.mark.asyncio
async def test_stream_until_user_input_with_tool_call(
    openai_client: AsyncOpenAI, openai_model: str
):
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    result = await stream_until_user_input(
        messages=messages,
        model=openai_model,
        openai_client=openai_client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers={"get_weather": get_weather_invoker},
        stream_kwargs={"tools": [GET_WEATHER_TOOL]},
    )

    assert isinstance(result, StreamResult)

    input_list = result.to_input_list()
    roles = [msg["role"] for msg in input_list]

    # user → assistant (tool_calls) → tool → assistant (final)
    assert roles == ["user", "assistant", "tool", "assistant"]
