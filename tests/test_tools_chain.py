import json
from typing import Any, Literal

import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from chat_cmpl_stream_handler import (
    ChatCompletionStreamHandler,
    StreamResult,
    stream_until_user_input,
)
from chat_cmpl_stream_handler.utils.pydantic_to_tool import (
    PydanticToolConfig,
    build_pydantic_tools_and_invokers,
)
from tests.conftest import LLMProvider


class BeginChainInput(BaseModel):
    kickoff: Literal["start"]


class UseAlphaInput(BaseModel):
    alpha_token: str


class UseBetaInput(BaseModel):
    beta_token: str


class UseGammaInput(BaseModel):
    gamma_token: str


def tool_output_of_next_step(tool_name: str, arguments: dict[str, str]) -> str:
    return json.dumps(
        {
            "status": "ok",
            "next_tool": tool_name,
            "next_arguments": arguments,
            "instruction": (
                f"Call `{tool_name}` next with the exact `next_arguments` object."
            ),
        }
    )


async def begin_chain_invoker(arguments: BeginChainInput, context: Any) -> str:
    assert context == "chain-test"
    assert arguments.kickoff == "start"
    return tool_output_of_next_step("use_alpha", {"alpha_token": "alpha-bridge"})


async def use_alpha_invoker(arguments: UseAlphaInput, context: Any) -> str:
    assert context == "chain-test"
    assert arguments.alpha_token == "alpha-bridge"
    return tool_output_of_next_step("use_beta", {"beta_token": "beta-lantern"})


async def use_beta_invoker(arguments: UseBetaInput, context: Any) -> str:
    assert context == "chain-test"
    assert arguments.beta_token == "beta-lantern"
    return tool_output_of_next_step("use_gamma", {"gamma_token": "gamma-harbor"})


async def use_gamma_invoker(arguments: UseGammaInput, context: Any) -> str:
    assert context == "chain-test"
    assert arguments.gamma_token == "gamma-harbor"
    return "CHAIN_COMPLETE"


def extract_tool_call_names(
    input_list: list[ChatCompletionMessageParam], model: str | None = None
) -> list[str]:
    names: list[str] = []
    for message in input_list:
        if "tool_calls" in message:
            content = message["content"]
            tool_calls = message["tool_calls"]
            if tool_calls and content:
                print(
                    f"The tool call from model '{model or 'not given'}' "
                    + f"has message content: {content}"
                )

            for tool_call in tool_calls or []:
                names.append(tool_call["function"]["name"])
    return names


@pytest.mark.asyncio
async def test_stream_until_user_input_with_dependent_tools_chain(
    llm_provider: LLMProvider,
):
    openai_client = llm_provider.client
    model = llm_provider.model

    tools, tool_invokers = build_pydantic_tools_and_invokers(
        [
            PydanticToolConfig(
                model=BeginChainInput,
                name="begin_chain",
                description="Start the chain with the kickoff value.",
                invoker=begin_chain_invoker,
            ),
            PydanticToolConfig(
                model=UseAlphaInput,
                name="use_alpha",
                description="Exchange the alpha token for the beta token.",
                invoker=use_alpha_invoker,
            ),
            PydanticToolConfig(
                model=UseBetaInput,
                name="use_beta",
                description="Exchange the beta token for the gamma token.",
                invoker=use_beta_invoker,
            ),
            PydanticToolConfig(
                model=UseGammaInput,
                name="use_gamma",
                description="Finish the chain with the gamma token.",
                invoker=use_gamma_invoker,
            ),
        ]
    )

    result = await stream_until_user_input(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are executing a strict tool chain test. "
                    "Call exactly one tool at a time. "
                    "Never invent the next tool arguments. "
                    "Read each tool result and copy its `next_arguments` exactly "
                    "into the next tool call. "
                    "When the final tool returns CHAIN_COMPLETE, answer with exactly "
                    "`CHAIN_COMPLETE`."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Start by calling `begin_chain` with "
                    '{"kickoff":"start"} and finish the full chain.'
                ),
            },
        ],
        model=model,
        openai_client=openai_client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers=tool_invokers,
        stream_kwargs={
            "tools": tools,
            "stream_options": {"include_usage": True},
        },
        context="chain-test",
        max_iterations=6,
    )

    assert isinstance(result, StreamResult)

    input_list = result.to_input_list()
    tool_call_names = extract_tool_call_names(input_list, model)

    assert tool_call_names == [
        "begin_chain",
        "use_alpha",
        "use_beta",
        "use_gamma",
    ]
    assert input_list[-1]["role"] == "assistant"
    assert input_list[-1]["content"] == "CHAIN_COMPLETE"
    assert len(result.usages) > 0
    assert all(u.total_tokens for u in result.usages)
