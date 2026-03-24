import logging
from typing import TYPE_CHECKING, Final, Generic, Text

from openai.lib._parsing._completions import ResponseFormatT
from openai.lib.streaming.chat._events import (
    ChunkEvent,
    ContentDeltaEvent,
    ContentDoneEvent,
    FunctionToolCallArgumentsDeltaEvent,
    FunctionToolCallArgumentsDoneEvent,
    LogprobsContentDeltaEvent,
    LogprobsContentDoneEvent,
    LogprobsRefusalDeltaEvent,
    LogprobsRefusalDoneEvent,
    RefusalDeltaEvent,
    RefusalDoneEvent,
)

if TYPE_CHECKING:
    from openai.lib.streaming.chat._events import ChatCompletionStreamEvent

__version__: Final[Text] = "0.1.0"


logger = logging.getLogger(__name__)


class ChatCompletionStreamHandler(Generic[ResponseFormatT]):
    async def handle(self, event: "ChatCompletionStreamEvent[ResponseFormatT]") -> None:
        """Internal dispatcher — routes each stream event to the right hook."""
        await self.on_event(event)

        if event.type == "chunk":
            await self.on_chunk(event)
        elif event.type == "content.delta":
            await self.on_content_delta(event)
        elif event.type == "content.done":
            await self.on_content_done(event)
        elif event.type == "refusal.delta":
            await self.on_refusal_delta(event)
        elif event.type == "refusal.done":
            await self.on_refusal_done(event)
        elif event.type == "tool_calls.function.arguments.delta":
            await self.on_tool_calls_function_arguments_delta(event)
        elif event.type == "tool_calls.function.arguments.done":
            await self.on_tool_calls_function_arguments_done(event)
        elif event.type == "logprobs.content.delta":
            await self.on_logprobs_content_delta(event)
        elif event.type == "logprobs.content.done":
            await self.on_logprobs_content_done(event)
        elif event.type == "logprobs.refusal.delta":
            await self.on_logprobs_refusal_delta(event)
        elif event.type == "logprobs.refusal.done":
            await self.on_logprobs_refusal_done(event)
        else:
            logger.warning(f"Unknown event type: {event.type}")

    async def on_event(
        self, event: "ChatCompletionStreamEvent[ResponseFormatT]"
    ) -> None:
        """Called for every stream event before more specific hooks."""
        pass

    async def on_chunk(self, event: ChunkEvent) -> None:
        """Called for every raw SSE chunk received from the API."""
        pass

    async def on_content_delta(self, event: ContentDeltaEvent) -> None:
        """Called each time a new content token arrives."""
        pass

    async def on_content_done(self, event: ContentDoneEvent[ResponseFormatT]) -> None:
        """Called once when the full content string is complete."""
        pass

    async def on_refusal_delta(self, event: RefusalDeltaEvent) -> None:
        """Called each time a new refusal token arrives."""
        pass

    async def on_refusal_done(self, event: RefusalDoneEvent) -> None:
        """Called once when the full refusal string is complete."""
        pass

    async def on_tool_calls_function_arguments_delta(
        self, event: FunctionToolCallArgumentsDeltaEvent
    ) -> None:
        """Called for each incremental JSON fragment of a tool-call's arguments."""
        pass

    async def on_tool_calls_function_arguments_done(
        self, event: FunctionToolCallArgumentsDoneEvent
    ) -> None:
        """Called once when a tool call's full argument JSON is available."""
        pass

    async def on_logprobs_content_delta(self, event: LogprobsContentDeltaEvent) -> None:
        """Called for each incremental list of content log-probability tokens."""
        pass

    async def on_logprobs_content_done(self, event: LogprobsContentDoneEvent) -> None:
        """Called once with the complete list of content log-probability tokens."""
        pass

    async def on_logprobs_refusal_delta(self, event: LogprobsRefusalDeltaEvent) -> None:
        """Called for each incremental list of refusal log-probability tokens."""
        pass

    async def on_logprobs_refusal_done(self, event: LogprobsRefusalDoneEvent) -> None:
        """Called once with the complete list of refusal log-probability tokens."""
        pass
