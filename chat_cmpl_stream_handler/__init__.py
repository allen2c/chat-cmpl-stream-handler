import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Final,
    Generic,
    Iterable,
    List,
    Optional,
    Text,
    Union,
)

from openai import AsyncOpenAI
from openai.lib._parsing._completions import ResponseFormatT
from openai.lib.streaming.chat import ChatCompletionStreamState
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
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.shared.chat_model import ChatModel

from chat_cmpl_stream_handler.utils.tool_call import (  # noqa: F401
    args_from_tool_call as args_from_tool_call,
)

if TYPE_CHECKING:
    from openai.lib.streaming.chat._events import ChatCompletionStreamEvent

__version__: Final[Text] = "0.5.0"


logger = logging.getLogger(__name__)


ToolInvokerFn = Callable[[ChatCompletionMessageToolCall, Any], Awaitable[str]]


async def stream_until_user_input(
    messages: Iterable[ChatCompletionMessageParam],
    model: Union[str, ChatModel],
    openai_client: AsyncOpenAI,
    *,
    stream_handler: "ChatCompletionStreamHandler[ResponseFormatT] | None" = None,
    tool_invokers: Dict[str, ToolInvokerFn] | None = None,
    stream_kwargs: Dict[Text, Any] | None = None,
    context: Any | None = None,
    max_iterations: int = 10,
    tool_call_output_callback: Optional[
        Callable[[ChatCompletionMessageFunctionToolCall, str], Awaitable[None]]
    ] = None,
    **kwargs,
) -> "StreamResult":
    current_messages = list(messages)
    usages: List["CompletionUsage"] = []
    active_stream_handler = stream_handler or ChatCompletionStreamHandler()

    for _ in range(max_iterations):
        # 1. stream the response
        state = ChatCompletionStreamState()
        stream = await openai_client.chat.completions.create(
            messages=current_messages,
            model=model,
            stream=True,
            **{
                k: v
                for k, v in (stream_kwargs or {}).items()
                if k not in ("messages", "model", "stream")
            },
        )

        async for chunk in stream:
            for event in state.handle_chunk(chunk):
                await active_stream_handler.handle(event)

        final = state.get_final_completion()
        if final.usage:
            usages.append(
                CompletionUsage.model_validate_json(final.usage.model_dump_json())
            )

        assistant_msg = final.choices[0].message
        current_messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=assistant_msg.content,
                **(
                    {
                        "tool_calls": [
                            ChatCompletionMessageFunctionToolCallParam(
                                id=tc.id,
                                type="function",
                                function={
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "{}",
                                },
                                **(
                                    {"extra_content": tc.model_extra["extra_content"]}
                                    if "extra_content" in getattr(tc, "model_extra", {})
                                    else {}
                                ),
                            )
                            for tc in assistant_msg.tool_calls
                        ]
                    }
                    if assistant_msg.tool_calls
                    else {}
                ),
            )
        )  # Add assistant message to history

        # 2. Check if there are tool calls
        if not assistant_msg.tool_calls:
            return StreamResult(current_messages, model, usages=usages)  # End

        # 3. Execute tool calls, and add the results back to messages
        for tool_call in assistant_msg.tool_calls:
            invoker = (tool_invokers or {}).get(tool_call.function.name)

            if invoker is None:
                raise ValueError(f"No invoker for tool: {tool_call.function.name}")

            tool_call_output = await invoker(tool_call, context)

            current_messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=tool_call_output,
                )
            )

            if tool_call_output_callback is not None:
                await tool_call_output_callback(tool_call, tool_call_output)

    raise MaxIterationsReached(
        f"Reached max_iterations={max_iterations} without waiting for user input."
    )


class StreamResult:
    def __init__(
        self,
        messages: List[ChatCompletionMessageParam],
        model: Union[str, ChatModel],
        usages: List["CompletionUsage"],
    ):
        self._messages = messages
        self._model = model

        self.usages = usages

    def to_input_list(self) -> List[ChatCompletionMessageParam]:
        return json.loads(json.dumps(self._messages, default=str))


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


class MaxIterationsReached(Exception):
    """Raised when stream_until_user_input exceeds the maximum iteration limit."""

    pass
