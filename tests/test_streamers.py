from typing import Any, AsyncIterator, Dict, List, cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from chat_cmpl_stream_handler import (
    ChunkStreamer,
    MaxIterationsReached,
    RunFailed,
    stream_until_user_input,
    stream_until_user_input_events,
)
from chat_cmpl_stream_handler.streamers import OpenAIStreamer
from tests.scripted import ScriptedStreamer, chunk, text_turn, tool_call_turn

PING_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Ping.",
        "parameters": {"type": "object", "properties": {}},
    },
}


async def ping_invoker(_tool_call: ChatCompletionMessageToolCall, _context: Any) -> str:
    return "pong"


# --- ScriptedStreamer: loop logic without a provider ------------------------------


def test_scripted_streamer_satisfies_the_protocol():
    assert isinstance(ScriptedStreamer(), ChunkStreamer)


@pytest.mark.parametrize("max_iterations", [1, 3])
@pytest.mark.asyncio
async def test_max_iterations_reached_is_reported_as_run_failed(max_iterations: int):
    streamer = ScriptedStreamer(*([tool_call_turn("ping")] * max_iterations))

    events = [
        event
        async for event in stream_until_user_input_events(
            messages=[{"role": "user", "content": "loop forever"}],
            model="scripted",
            openai_client=streamer,
            tool_invokers={"ping": ping_invoker},
            stream_kwargs={"tools": [PING_TOOL]},
            max_iterations=max_iterations,
        )
    ]

    failure = events[-1]
    assert isinstance(failure, RunFailed)
    assert isinstance(failure.exception, MaxIterationsReached)
    assert f"max_iterations={max_iterations}" in str(failure.exception)
    assert len(streamer.requests) == max_iterations


@pytest.mark.asyncio
async def test_max_iterations_reached_raises_through_the_callback_api():
    streamer = ScriptedStreamer(tool_call_turn("ping"))

    with pytest.raises(MaxIterationsReached):
        await stream_until_user_input(
            messages=[{"role": "user", "content": "loop forever"}],
            model="scripted",
            openai_client=streamer,
            tool_invokers={"ping": ping_invoker},
            stream_kwargs={"tools": [PING_TOOL]},
            max_iterations=1,
        )


@pytest.mark.asyncio
async def test_the_loop_replays_the_growing_history_to_every_request():
    streamer = ScriptedStreamer(tool_call_turn("ping"), text_turn("done"))

    result = await stream_until_user_input(
        messages=[{"role": "user", "content": "ping please"}],
        model="scripted",
        openai_client=streamer,
        tool_invokers={"ping": ping_invoker},
        stream_kwargs={"tools": [PING_TOOL]},
    )

    first, second = streamer.requests
    assert [message["role"] for message in first["messages"]] == ["user"]
    assert [message["role"] for message in second["messages"]] == ["user", "assistant", "tool"]
    assert first["tools"] == [PING_TOOL]
    assert [message["role"] for message in result.to_input_list()] == ["user", "assistant", "tool", "assistant"]


# --- OpenAIStreamer: the quirks that used to need a global monkey-patch -----------


class _FakeAsyncStream:
    """Stands in for openai's AsyncStream — iterable once, closable."""

    def __init__(self, chunks: List[ChatCompletionChunk]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        for item in self._chunks:
            yield item

    async def close(self) -> None:
        self.closed = True


class _FakeAsyncOpenAI:
    """The two attribute hops OpenAIStreamer walks, and nothing else."""

    def __init__(self, stream: _FakeAsyncStream) -> None:
        self._stream = stream
        self.requests: List[Dict[str, Any]] = []

        completions = type("_Completions", (), {"create": self._create})()
        self.chat = type("_Chat", (), {"completions": completions})()

    async def _create(self, **kwargs: Any) -> _FakeAsyncStream:
        self.requests.append(kwargs)
        return self._stream


def _tool_call_chunk_without_index(count: int) -> ChatCompletionChunk:
    """A chunk as Gemini's OpenAI-compatible endpoint sends it: index is None."""
    built = chunk(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "index": position,
                    "id": f"call_{position}",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
                for position in range(count)
            ],
        }
    )
    for tool_call in cast(Any, built.choices[0].delta.tool_calls):
        tool_call.index = None
    return built


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.asyncio
async def test_openai_streamer_normalizes_a_missing_tool_call_index(count: int):
    stream = _FakeAsyncStream([_tool_call_chunk_without_index(count)])
    streamer = OpenAIStreamer(cast(AsyncOpenAI, _FakeAsyncOpenAI(stream)))

    chunks = [item async for item in streamer.stream(messages=[], model="scripted")]

    indices = [tool_call.index for tool_call in chunks[0].choices[0].delta.tool_calls or []]
    assert indices == list(range(count))


@pytest.mark.asyncio
async def test_openai_streamer_closes_the_stream_when_the_consumer_stops_early():
    stream = _FakeAsyncStream([chunk({"role": "assistant", "content": "a"}), chunk({"content": "b"})])
    streamer = OpenAIStreamer(cast(AsyncOpenAI, _FakeAsyncOpenAI(stream)))

    chunks = streamer.stream(messages=[], model="scripted")
    await chunks.__anext__()
    await chunks.aclose()

    assert stream.closed
