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


class BeginPipelineInput(BaseModel):
    phase: Literal["go"]


class UseStep1Input(BaseModel):
    step1_ticket: str


class UseStep2Input(BaseModel):
    step2_ticket: str


class UseStep3Input(BaseModel):
    step3_ticket: str


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


async def begin_pipeline_invoker(arguments: BeginPipelineInput, context: Any) -> str:
    assert context == "pipeline-test"
    assert arguments.phase == "go"
    return tool_output_of_next_step("use_step_1", {"step1_ticket": "slot-a-001"})


async def use_step_1_invoker(arguments: UseStep1Input, context: Any) -> str:
    assert context == "pipeline-test"
    assert arguments.step1_ticket == "slot-a-001"
    return tool_output_of_next_step("use_step_2", {"step2_ticket": "slot-b-002"})


async def use_step_2_invoker(arguments: UseStep2Input, context: Any) -> str:
    assert context == "pipeline-test"
    assert arguments.step2_ticket == "slot-b-002"
    return tool_output_of_next_step("use_step_3", {"step3_ticket": "slot-c-003"})


async def use_step_3_invoker(arguments: UseStep3Input, context: Any) -> str:
    assert context == "pipeline-test"
    assert arguments.step3_ticket == "slot-c-003"
    return "PIPELINE_DONE"


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
async def test_stream_until_user_input_with_dependent_tools_pipeline(
    llm_provider: LLMProvider,
):
    openai_client = llm_provider.client
    model = llm_provider.model

    tools, tool_invokers = build_pydantic_tools_and_invokers(
        [
            PydanticToolConfig(
                model=BeginPipelineInput,
                name="begin_pipeline",
                description="Start the ordered pipeline with the required phase value.",
                invoker=begin_pipeline_invoker,
            ),
            PydanticToolConfig(
                model=UseStep1Input,
                name="use_step_1",
                description=(
                    "Advance the pipeline using the step-1 ticket from the previous "
                    "tool result."
                ),
                invoker=use_step_1_invoker,
            ),
            PydanticToolConfig(
                model=UseStep2Input,
                name="use_step_2",
                description=(
                    "Advance the pipeline using the step-2 ticket from the previous "
                    "tool result."
                ),
                invoker=use_step_2_invoker,
            ),
            PydanticToolConfig(
                model=UseStep3Input,
                name="use_step_3",
                description=(
                    "Finish the pipeline using the step-3 ticket from the previous "
                    "tool result."
                ),
                invoker=use_step_3_invoker,
            ),
        ]
    )

    result = await stream_until_user_input(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are running a deterministic pipeline test. "
                    "Call exactly one tool at a time. "
                    "Never invent the next tool arguments. "
                    "Read each tool result and copy its `next_arguments` exactly "
                    "into the next tool call. "
                    "When the final tool returns PIPELINE_DONE, answer with exactly "
                    "`PIPELINE_DONE`."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Begin by calling `begin_pipeline` with "
                    '{"phase":"go"} and complete every step.'
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
        context="pipeline-test",
        max_iterations=6,
    )

    assert isinstance(result, StreamResult)

    input_list = result.to_input_list()
    tool_call_names = extract_tool_call_names(input_list, model)

    assert tool_call_names == [
        "begin_pipeline",
        "use_step_1",
        "use_step_2",
        "use_step_3",
    ]
    assert input_list[-1]["role"] == "assistant"
    assert input_list[-1]["content"] == "PIPELINE_DONE"
    assert len(result.usages) > 0
    assert all(u.total_tokens for u in result.usages)
