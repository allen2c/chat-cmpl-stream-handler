"""Offline tests for the genai translation, both directions.

The one that matters is `test_a_thought_signature_survives_a_full_round_trip` — every other
test here checks one hop of a path that is only useful end to end.
"""

import base64
import json
from typing import Any, AsyncIterator, Dict, List, Optional, cast

import pytest
from google import genai
from google.genai import types
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam

from chat_cmpl_stream_handler import _assistant_msg_to_param
from chat_cmpl_stream_handler._genai_streamer import GenAIStreamer, _to_config, _to_contents

SIGNATURE = b"\n\x1bsignature-bytes\xff"
SIGNATURE_B64 = base64.b64encode(SIGNATURE).decode()

WEATHER_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

GET_WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": WEATHER_PARAMETERS,
    },
}


# --- fake client ------------------------------------------------------------------


class _FakeModels:
    def __init__(self, responses: List[types.GenerateContentResponse]) -> None:
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[types.GenerateContentResponse]:
        self.calls.append(kwargs)

        async def iterator() -> AsyncIterator[types.GenerateContentResponse]:
            for response in self._responses:
                yield response

        return iterator()


class _FakeGenAIClient:
    def __init__(self, responses: List[types.GenerateContentResponse]) -> None:
        self.models = _FakeModels(responses)
        self.aio = self


def _fake_streamer(responses: List[types.GenerateContentResponse]) -> tuple[GenAIStreamer, _FakeModels]:
    client = _FakeGenAIClient(responses)
    return GenAIStreamer(cast(genai.Client, client)), client.models


def _response(
    *parts: types.Part,
    finish_reason: Optional[types.FinishReason] = None,
    usage: Optional[types.GenerateContentResponseUsageMetadata] = None,
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=usage,
    )


async def _collect(streamer: GenAIStreamer, **kwargs: Any) -> List[ChatCompletionChunk]:
    return [chunk async for chunk in streamer.stream(**kwargs)]


# --- response: genai -> OpenAI ----------------------------------------------------


@pytest.mark.asyncio
async def test_text_parts_become_content_deltas():
    streamer, _ = _fake_streamer(
        [
            _response(types.Part(text="sunny ")),
            _response(types.Part(text="and warm"), finish_reason=types.FinishReason.STOP),
        ]
    )

    chunks = await _collect(streamer, messages=[], model="gemini-3")

    assert [chunk.choices[0].delta.content for chunk in chunks] == ["sunny ", "and warm", None]
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_a_thought_part_is_not_replayed_as_assistant_content():
    streamer, _ = _fake_streamer(
        [
            _response(types.Part(text="the user wants weather", thought=True)),
            _response(types.Part(text="It is sunny."), finish_reason=types.FinishReason.STOP),
        ]
    )

    chunks = await _collect(streamer, messages=[], model="gemini-3")

    contents = [chunk.choices[0].delta.content for chunk in chunks]
    assert "the user wants weather" not in contents
    assert "It is sunny." in contents


@pytest.mark.asyncio
async def test_usage_and_finish_reason_land_on_the_final_chunk():
    streamer, _ = _fake_streamer(
        [
            _response(
                types.Part(text="hi"),
                finish_reason=types.FinishReason.MAX_TOKENS,
                usage=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=11,
                    candidates_token_count=5,
                    thoughts_token_count=3,
                    total_token_count=19,
                ),
            )
        ]
    )

    chunks = await _collect(streamer, messages=[], model="gemini-3")

    assert chunks[0].usage is None
    usage = chunks[-1].usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 8, 19)
    assert chunks[-1].choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_a_function_call_becomes_a_tool_call_and_finishes_as_tool_calls():
    streamer, _ = _fake_streamer(
        [
            _response(
                types.Part(
                    function_call=types.FunctionCall(name="get_weather", args={"city": "Tokyo"}),
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    chunks = await _collect(streamer, messages=[], model="gemini-3")

    tool_call = (chunks[0].choices[0].delta.tool_calls or [])[0]
    assert tool_call.index == 0
    assert tool_call.function is not None
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments or "") == {"city": "Tokyo"}
    assert tool_call.id is not None and tool_call.id.startswith("call_genai_")
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_provider_state_is_emitted_once_on_the_final_chunk():
    streamer, _ = _fake_streamer(
        [
            _response(types.Part(text="thinking", thought=True, thought_signature=SIGNATURE)),
            _response(
                types.Part(
                    function_call=types.FunctionCall(name="get_weather", args={}),
                    thought_signature=SIGNATURE,
                ),
                finish_reason=types.FinishReason.STOP,
            ),
        ]
    )

    chunks = await _collect(streamer, messages=[], model="gemini-3")

    carriers = [chunk for chunk in chunks if "provider_specific_fields" in (chunk.choices[0].delta.model_extra or {})]
    assert len(carriers) == 1 and carriers[0] is chunks[-1]

    delta_extra = chunks[-1].choices[0].delta.model_extra or {}
    assert delta_extra["provider_specific_fields"] == {"thought_signatures": [SIGNATURE_B64]}

    final_tool_call = (chunks[-1].choices[0].delta.tool_calls or [])[0]
    assert (final_tool_call.model_extra or {})["provider_specific_fields"] == {"thought_signature": SIGNATURE_B64}
    # Nothing else rides along: a repeated id or name would be concatenated, not replaced.
    # `function` is present but empty only because the OpenAI SDK asserts it is not None.
    assert final_tool_call.id is None
    assert final_tool_call.function is not None
    assert final_tool_call.function.name is None and final_tool_call.function.arguments is None


# --- request: OpenAI -> genai -----------------------------------------------------


def test_system_messages_become_a_system_instruction():
    contents, system_instruction = _to_contents(
        [
            {"role": "system", "content": "be concise"},
            {"role": "system", "content": "be kind"},
            {"role": "user", "content": "hi"},
        ]
    )

    assert system_instruction == "be concise\n\nbe kind"
    assert [content.role for content in contents] == ["user"]


def test_a_tool_message_becomes_a_function_response_named_after_its_call():
    contents, _ = _to_contents(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
    )

    assert [content.role for content in contents] == ["user", "model", "user"]
    function_response = (contents[2].parts or [])[0].function_response
    assert function_response is not None
    assert function_response.name == "get_weather"
    assert function_response.id == "call_1"
    assert function_response.response == {"result": "sunny"}


def test_a_tool_message_with_an_unknown_call_id_is_rejected():
    with pytest.raises(ValueError, match="unknown tool_call_id"):
        _to_contents([{"role": "tool", "tool_call_id": "nope", "content": "sunny"}])


def test_a_synthetic_tool_call_id_is_never_sent_back():
    contents, _ = _to_contents(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_genai_deadbeef",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_genai_deadbeef", "content": "sunny"},
        ]
    )

    function_call = (contents[0].parts or [])[0].function_call
    function_response = (contents[1].parts or [])[0].function_response
    assert function_call is not None and function_call.id is None
    assert function_response is not None and function_response.id is None


def test_tools_pass_through_as_raw_json_schema():
    config = _to_config({"tools": [GET_WEATHER_TOOL]}, None)

    tool = cast(types.Tool, (config.tools or [])[0])
    declaration = (tool.function_declarations or [])[0]
    assert declaration.name == "get_weather"
    assert declaration.parameters_json_schema == WEATHER_PARAMETERS


@pytest.mark.parametrize(
    "tool_choice, expected_mode, expected_names",
    [
        ("auto", types.FunctionCallingConfigMode.AUTO, None),
        ("none", types.FunctionCallingConfigMode.NONE, None),
        ("required", types.FunctionCallingConfigMode.ANY, None),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            types.FunctionCallingConfigMode.ANY,
            ["get_weather"],
        ),
    ],
)
def test_tool_choice_maps_onto_a_function_calling_mode(
    tool_choice: Any,
    expected_mode: types.FunctionCallingConfigMode,
    expected_names: Optional[List[str]],
):
    config = _to_config({"tool_choice": tool_choice}, None)

    function_calling_config = (config.tool_config or types.ToolConfig()).function_calling_config
    assert function_calling_config is not None
    assert function_calling_config.mode == expected_mode
    assert function_calling_config.allowed_function_names == expected_names


def test_a_json_schema_response_format_is_close_to_a_passthrough():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    config = _to_config(
        {"response_format": {"type": "json_schema", "json_schema": {"name": "Answer", "schema": schema}}},
        None,
    )

    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema


@pytest.mark.parametrize(
    "openai_option, genai_field, value",
    [
        ("max_tokens", "max_output_tokens", 64),
        ("stop", "stop_sequences", ["\n"]),
        ("temperature", "temperature", 0.5),
    ],
)
def test_request_options_are_renamed_not_dropped(openai_option: str, genai_field: str, value: Any):
    config = _to_config({openai_option: value}, None)

    assert getattr(config, genai_field) == value


def test_options_with_no_genai_equivalent_are_ignored():
    config = _to_config({"parallel_tool_calls": False, "stream_options": {"include_usage": True}}, None)

    assert config.temperature is None  # nothing blew up, nothing was invented


# --- the round trip ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_thought_signature_survives_a_full_round_trip():
    """genai bytes -> chunk -> accumulator -> message param -> JSON -> genai bytes."""
    streamer, models = _fake_streamer(
        [
            _response(
                types.Part(
                    function_call=types.FunctionCall(id="call_real", name="get_weather", args={"city": "Tokyo"}),
                    thought_signature=SIGNATURE,
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    first_turn: List[ChatCompletionMessageParam] = [{"role": "user", "content": "weather?"}]
    chunks = await _collect(streamer, messages=first_turn, model="gemini-3")

    state = ChatCompletionStreamState()
    for chunk in chunks:
        state.handle_chunk(chunk)
    assistant_param = _assistant_msg_to_param(state.get_final_completion().choices[0].message)

    # It survives the accumulator...
    tool_call_params = cast(List[Dict[str, Any]], assistant_param.get("tool_calls"))
    assert tool_call_params[0]["provider_specific_fields"] == {"thought_signature": SIGNATURE_B64}

    # ...and the JSON round trip `to_input_list()` performs.
    replayed: List[ChatCompletionMessageParam] = json.loads(json.dumps([first_turn[0], assistant_param]))

    # ...and comes back out as the exact bytes Gemini handed us.
    contents, _ = _to_contents(replayed)
    part = (contents[1].parts or [])[0]
    assert part.thought_signature == SIGNATURE
    assert part.function_call is not None and part.function_call.id == "call_real"
    assert models.calls[0]["contents"][0].role == "user"
