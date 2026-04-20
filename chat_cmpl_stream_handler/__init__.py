"""Streaming chat completion utilities with tool-call orchestration.

Public surface:

* ``stream_until_user_invoker`` / ``merge_tools_and_invokers`` — the loop
  and its tool-source merger.
* ``Tool`` / ``FunctionTool`` — optional ergonomic packaging for
  ``(schema, invoker)`` pairs.
* ``ChatCompletionStreamHandler`` / ``StreamResult`` /
  ``MaxIterationsReached`` — observation, result, and error types.
"""

import json
import logging
from dataclasses import dataclass
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
    Protocol,
    Sequence,
    Text,
    Tuple,
    Union,
    runtime_checkable,
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
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
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


def merge_tools_and_invokers(
    tools: "Sequence[Union[Tool, ChatCompletionToolParam]] | None" = None,
    tool_invokers: Dict[str, ToolInvokerFn] | None = None,
    stream_tools: Iterable[ChatCompletionToolParam] | None = None,
) -> Tuple[List[ChatCompletionToolParam], Dict[str, ToolInvokerFn]]:
    """Merge tool sources into the primitive ``(schemas, invokers)`` pair.

    Sources, applied in order (later wins for schemas; explicit
    ``tool_invokers`` always wins for invokers):

    1. ``stream_tools`` — raw schemas already in ``stream_kwargs["tools"]``.
    2. ``tools`` — ``Tool`` objects and/or raw schemas.
    3. ``tool_invokers`` — explicit name → callable overlay.

    Raises ``ValueError`` if any schema name lacks a matching invoker.
    """
    schemas_by_name: Dict[str, ChatCompletionToolParam] = {}
    invokers: Dict[str, ToolInvokerFn] = {}

    for schema in stream_tools or ():
        schemas_by_name[schema["function"]["name"]] = schema

    for item in tools or ():
        if isinstance(item, Tool):
            param = item.tool_param
            name = param["function"]["name"]
            schemas_by_name[name] = param
            invokers[name] = item.invoke
        else:
            schemas_by_name[item["function"]["name"]] = item

    for name, fn in (tool_invokers or {}).items():
        if name in invokers:
            logger.warning(
                f"tool_invokers[{name!r}] overrides Tool.invoke from `tools=`"
            )
        invokers[name] = fn

    missing = [name for name in schemas_by_name if name not in invokers]
    if missing:
        raise ValueError(f"No invoker for tool(s): {missing}")

    return list(schemas_by_name.values()), invokers


async def stream_until_user_input(
    messages: Iterable[ChatCompletionMessageParam],
    model: Union[str, ChatModel],
    openai_client: AsyncOpenAI,
    *,
    stream_handler: "ChatCompletionStreamHandler[ResponseFormatT] | None" = None,
    tools: "Sequence[Union[Tool, ChatCompletionToolParam]] | None" = None,
    tool_invokers: Dict[str, ToolInvokerFn] | None = None,
    stream_kwargs: Dict[Text, Any] | None = None,
    context: Any | None = None,
    max_iterations: int = 10,
    tool_call_output_callback: Optional[
        Callable[[ChatCompletionMessageFunctionToolCall, str], Awaitable[None]]
    ] = None,
    **kwargs,
) -> "StreamResult":
    merged_stream_kwargs: Dict[Text, Any] = dict(stream_kwargs or {})
    resolved_tools, resolved_invokers = merge_tools_and_invokers(
        tools=tools,
        tool_invokers=tool_invokers,
        stream_tools=merged_stream_kwargs.pop("tools", None),
    )
    if resolved_tools:
        merged_stream_kwargs["tools"] = resolved_tools

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
                for k, v in merged_stream_kwargs.items()
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
            invoker = resolved_invokers.get(tool_call.function.name)

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


@runtime_checkable
class Tool(Protocol):
    """A self-contained tool: schema + invoker travel together.

    Any object exposing ``tool_param`` and ``invoke`` qualifies — no base
    class required. Pass instances via ``stream_until_user_input(tools=...)``.
    """

    tool_param: ChatCompletionToolParam

    async def invoke(
        self, tool_call: ChatCompletionMessageToolCall, context: Any
    ) -> str: ...


@dataclass(frozen=True)
class FunctionTool(Tool):
    """Trivial ``Tool`` implementation for users who don't want a subclass."""

    tool_param: ChatCompletionToolParam
    invoker: ToolInvokerFn

    async def invoke(
        self, tool_call: ChatCompletionMessageToolCall, context: Any
    ) -> str:
        return await self.invoker(tool_call, context)


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
