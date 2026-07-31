import os
from typing import Any

import pytest
from litellm.router import Router
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel

from chat_cmpl_stream_handler import (
    RunCompleted,
    StreamResult,
    ToolCallStarted,
    args_from_tool_call,
    stream_until_user_input,
    stream_until_user_input_events,
)
from chat_cmpl_stream_handler.utils.get_strict_json_schema import get_strict_json_schema
from tests.conftest import as_dicts

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


async def get_weather_invoker(tool_call: ChatCompletionMessageToolCall, _context: Any) -> str:
    args = args_from_tool_call(tool_call)
    return f"The weather in {args['city']} is sunny and 25°C."


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


@pytest.fixture(scope="session")
def gemini_router() -> Router:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not set")

    return Router(
        model_list=[
            {
                "model_name": "flash",
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-lite",
                    "api_key": api_key,
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_litellm_router_runs_the_tool_loop(gemini_router: Router):
    """A Router deployment drives the same loop an AsyncOpenAI client does."""
    tool_call_ids: list[str] = []
    result: StreamResult | None = None

    async for event in stream_until_user_input_events(
        messages=[
            {"role": "system", "content": "you are a concise assistant"},
            {"role": "user", "content": "Weather in Tokyo?"},
        ],
        model="flash",
        openai_client=gemini_router,
        tool_invokers={"get_weather": get_weather_invoker},
        stream_kwargs={"tools": [GET_WEATHER_TOOL]},
    ):
        if isinstance(event, ToolCallStarted):
            tool_call_ids.append(event.tool_call.id)
        elif isinstance(event, RunCompleted):
            result = event.result

    assert result is not None, "the run did not complete"
    messages = result.to_input_list()
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]

    # Gemini 3 signatures ride inside the tool-call id; the loop must not clean them up.
    assert tool_call_ids and all("__thought__" in call_id for call_id in tool_call_ids)
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == tool_call_ids


@pytest.mark.asyncio
async def test_litellm_router_passes_response_format_through(gemini_router: Router):
    """`response_format` is a passthrough on the Router path, as it is on the other two.

    LiteLLMStreamer forwards request kwargs untouched, so this needs no adapter code —
    which is exactly the kind of claim worth a test rather than an argument.
    """
    result = await stream_until_user_input(
        messages=[
            {"role": "system", "content": "You are a math tutor. Show the steps."},
            {"role": "user", "content": "how can I solve 8x + 7 = -23"},
        ],
        model="flash",
        openai_client=gemini_router,
        stream_kwargs={"response_format": get_strict_json_schema(MathReasoning)},
    )

    assistant_message = as_dicts(result.to_input_list())[-1]
    assert assistant_message["role"] == "assistant"
    assert isinstance(assistant_message["content"], str)

    reasoning = MathReasoning.model_validate_json(assistant_message["content"])
    assert reasoning.steps
    assert reasoning.final_answer
