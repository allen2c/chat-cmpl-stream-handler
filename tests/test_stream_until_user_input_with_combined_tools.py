import json
from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from chat_cmpl_stream_handler import (
    ChatCompletionStreamHandler,
    StreamResult,
    stream_until_user_input,
)
from chat_cmpl_stream_handler.utils.mcp import (
    MCPServerConfig,
    build_mcp_tools_and_invokers,
)
from chat_cmpl_stream_handler.utils.pydantic_to_tool import (
    PydanticToolConfig,
    build_pydantic_tools_and_invokers,
)

AWS_MCP_URL: str = "https://marketplace-mcp.us-east-1.api.aws/mcp"
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


class EchoRequest(BaseModel):
    text: str


async def echo_invoker(arguments: EchoRequest, context: Any) -> str:
    assert context == "test"
    return f"echo:{arguments.text}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_case",
    [
        "weather",
        "mcp",
        "pydantic",
    ],
    ids=["local-weather-tool", "mcp-tool", "pydantic-tool"],
)
async def test_stream_until_user_input_with_combined_tools(
    openai_client: AsyncOpenAI,
    openai_model: str,
    query_case: str,
):
    mcp_tools, mcp_tool_invokers = await build_mcp_tools_and_invokers(
        [
            MCPServerConfig(
                server_url=AWS_MCP_URL,
                server_label="aws",
            )
        ]
    )
    assert len(mcp_tools) > 0

    pydantic_tools, pydantic_tool_invokers = build_pydantic_tools_and_invokers(
        [
            PydanticToolConfig(
                model=EchoRequest,
                invoker=echo_invoker,
            )
        ]
    )

    all_tools = [GET_WEATHER_TOOL, *mcp_tools, *pydantic_tools]
    all_tool_invokers = {
        "get_weather": get_weather_invoker,
        **mcp_tool_invokers,
        **pydantic_tool_invokers,
    }

    target_mcp_tool_name = mcp_tools[0]["function"]["name"]
    target_pydantic_tool_name = pydantic_tools[0]["function"]["name"]
    if query_case == "weather":
        query = "What's the weather in Tokyo?"
        expected_tool_name = "get_weather"
    elif query_case == "mcp":
        query = (
            f"Call the tool `{target_mcp_tool_name}` with an empty JSON object, "
            "then summarize the result in one sentence."
        )
        expected_tool_name = target_mcp_tool_name
    else:
        query = (
            f"Call the tool `{target_pydantic_tool_name}` with "
            '{"text":"hello"} and then summarize the result in one sentence.'
        )
        expected_tool_name = target_pydantic_tool_name

    result = await stream_until_user_input(
        messages=[{"role": "user", "content": query}],
        model=openai_model,
        openai_client=openai_client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers=all_tool_invokers,
        stream_kwargs={
            "tools": all_tools,
            "stream_options": {"include_usage": True},
        },
        context="test",
    )

    assert isinstance(result, StreamResult)

    input_list = result.to_input_list()
    roles = [msg["role"] for msg in input_list]
    tool_call_names = _extract_tool_call_names(input_list)

    assert roles[0] == "user"
    assert "assistant" in roles
    assert "tool" in roles
    assert roles[-1] == "assistant"
    assert expected_tool_name in tool_call_names
    assert len(result.usages) > 0
    assert all(u.total_tokens for u in result.usages)


def _extract_tool_call_names(input_list: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in input_list:
        for tool_call in message.get("tool_calls") or []:
            names.append(tool_call["function"]["name"])
    return names
