import os
from typing import Any

import pytest
from google import genai
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel

from chat_cmpl_stream_handler import args_from_tool_call, stream_until_user_input
from chat_cmpl_stream_handler.utils.get_strict_json_schema import get_strict_json_schema
from tests.conftest import as_dicts

MODEL = "gemini-3.1-flash-lite"

GET_WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name."}},
            "required": ["city"],
        },
    },
}


class Answer(BaseModel):
    city: str
    celsius: int


async def get_weather_invoker(tool_call: ChatCompletionMessageToolCall, _context: Any) -> str:
    args = args_from_tool_call(tool_call)
    return f"The weather in {args['city']} is sunny and 25°C."


@pytest.fixture(scope="session")
def genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


@pytest.mark.asyncio
async def test_genai_client_runs_the_tool_loop(genai_client: genai.Client):
    """The second turn only succeeds if the thought signature made it back to Gemini."""
    result = await stream_until_user_input(
        messages=[
            {"role": "system", "content": "you are a concise assistant"},
            {"role": "user", "content": "Weather in Tokyo?"},
        ],
        model=MODEL,
        openai_client=genai_client,
        tool_invokers={"get_weather": get_weather_invoker},
        stream_kwargs={"tools": [GET_WEATHER_TOOL]},
    )

    messages = as_dicts(result.to_input_list())
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]

    tool_call = messages[2]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert "__thought__" not in tool_call["id"], "native genai ids stay clean"
    assert tool_call["provider_specific_fields"]["thought_signature"]

    assert messages[-1]["content"]
    assert len(result.usages) > 0
    assert all(usage.total_tokens for usage in result.usages)


@pytest.mark.asyncio
async def test_genai_client_returns_structured_output(genai_client: genai.Client):
    result = await stream_until_user_input(
        messages=[
            {"role": "system", "content": "you are a concise assistant"},
            {"role": "user", "content": "Tokyo is 25 degrees celsius. Report it."},
        ],
        model=MODEL,
        openai_client=genai_client,
        stream_kwargs={"response_format": get_strict_json_schema(Answer)},
    )

    content = as_dicts(result.to_input_list())[-1]["content"]
    assert isinstance(content, str)
    assert Answer.model_validate_json(content).celsius == 25
