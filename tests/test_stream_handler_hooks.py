"""`ChatCompletionStreamHandler.handle()` routes each event to its own hook.

The refusal and logprobs branches of that dispatch chain are public API and had no
coverage. Driven offline through the real path — scripted chunks, the OpenAI SDK's
accumulator, `StreamEvent`, `handle()` — rather than by handing the handler a
hand-built event, so the wiring is what gets tested.
"""

from typing import TYPE_CHECKING, Any, Dict, List

import pytest
from openai.lib.streaming.chat._events import (
    ContentDeltaEvent,
    ContentDoneEvent,
    LogprobsContentDeltaEvent,
    LogprobsContentDoneEvent,
    LogprobsRefusalDeltaEvent,
    LogprobsRefusalDoneEvent,
    RefusalDeltaEvent,
    RefusalDoneEvent,
)

from chat_cmpl_stream_handler import ChatCompletionStreamHandler, stream_until_user_input
from tests.scripted import ScriptedStreamer, chunk

if TYPE_CHECKING:
    # A Union alias, so it is subscriptable to a type checker but not at runtime.
    from openai.lib.streaming.chat._events import ChatCompletionStreamEvent


def token(text: str) -> Dict[str, Any]:
    """One log-probability token, as a provider sends it."""
    return {"token": text, "logprob": -0.5, "bytes": list(text.encode()), "top_logprobs": []}


def refusal_turn() -> List[Any]:
    """A turn where the model refuses, with per-token refusal logprobs."""
    return [
        chunk({"role": "assistant", "refusal": "I cannot"}, logprobs={"refusal": [token("I"), token(" cannot")]}),
        chunk({"refusal": " help"}, logprobs={"refusal": [token(" help")]}),
        chunk({}, finish_reason="stop"),
    ]


def content_turn() -> List[Any]:
    """A turn where the model answers, with per-token content logprobs."""
    return [
        chunk({"role": "assistant", "content": "hi"}, logprobs={"content": [token("hi")]}),
        chunk({"content": " there"}, logprobs={"content": [token(" there")]}),
        chunk({}, finish_reason="stop"),
    ]


class RecordingHandler(ChatCompletionStreamHandler[None]):
    """Records which hook fired, in order, and the last event each one saw."""

    def __init__(self) -> None:
        self.event_types: List[str] = []
        self.hooks: List[str] = []
        self.by_hook: Dict[str, Any] = {}

    def _record(self, name: str, event: Any) -> None:
        self.hooks.append(name)
        self.by_hook[name] = event

    async def on_event(self, event: "ChatCompletionStreamEvent[None]") -> None:
        self.event_types.append(event.type)

    async def on_content_delta(self, event: ContentDeltaEvent) -> None:
        self._record("on_content_delta", event)

    async def on_content_done(self, event: ContentDoneEvent[None]) -> None:
        self._record("on_content_done", event)

    async def on_refusal_delta(self, event: RefusalDeltaEvent) -> None:
        self._record("on_refusal_delta", event)

    async def on_refusal_done(self, event: RefusalDoneEvent) -> None:
        self._record("on_refusal_done", event)

    async def on_logprobs_content_delta(self, event: LogprobsContentDeltaEvent) -> None:
        self._record("on_logprobs_content_delta", event)

    async def on_logprobs_content_done(self, event: LogprobsContentDoneEvent) -> None:
        self._record("on_logprobs_content_done", event)

    async def on_logprobs_refusal_delta(self, event: LogprobsRefusalDeltaEvent) -> None:
        self._record("on_logprobs_refusal_delta", event)

    async def on_logprobs_refusal_done(self, event: LogprobsRefusalDoneEvent) -> None:
        self._record("on_logprobs_refusal_done", event)


async def run(turn: List[Any]) -> RecordingHandler:
    handler = RecordingHandler()
    await stream_until_user_input(
        messages=[{"role": "user", "content": "anything"}],
        model="scripted",
        openai_client=ScriptedStreamer(turn),
        stream_handler=handler,
    )
    return handler


@pytest.mark.asyncio
async def test_a_refusal_reaches_the_refusal_hooks():
    handler = await run(refusal_turn())

    assert handler.hooks == [
        "on_refusal_delta",
        "on_logprobs_refusal_delta",
        "on_refusal_delta",
        "on_logprobs_refusal_delta",
        "on_refusal_done",
        "on_logprobs_refusal_done",
    ]
    assert handler.by_hook["on_refusal_delta"].delta == " help"
    assert handler.by_hook["on_refusal_done"].refusal == "I cannot help"
    assert [entry.token for entry in handler.by_hook["on_logprobs_refusal_done"].refusal] == ["I", " cannot", " help"]


@pytest.mark.asyncio
async def test_content_logprobs_reach_the_logprobs_hooks():
    handler = await run(content_turn())

    assert handler.hooks == [
        "on_content_delta",
        "on_logprobs_content_delta",
        "on_content_delta",
        "on_logprobs_content_delta",
        "on_content_done",
        "on_logprobs_content_done",
    ]
    assert handler.by_hook["on_content_done"].content == "hi there"
    assert [entry.token for entry in handler.by_hook["on_logprobs_content_done"].content] == ["hi", " there"]


@pytest.mark.asyncio
async def test_on_event_sees_every_event_including_the_raw_chunks():
    handler = await run(refusal_turn())

    # on_event runs before the typed hooks and is not filtered by them: `chunk` has its
    # own hook and no logprobs/refusal flavour, yet it still shows up here.
    assert handler.event_types.count("chunk") == 3
    assert set(handler.event_types) == {
        "chunk",
        "refusal.delta",
        "refusal.done",
        "logprobs.refusal.delta",
        "logprobs.refusal.done",
    }
