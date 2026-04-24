"""Streaming chat completion helpers with tool-call orchestration.

The generator API yields lifecycle events. The callback API keeps the older
handler-based flow. Tools can be passed as raw schemas with invokers or as
small objects that carry both together.
"""

import json
import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Final,
    Generic,
    Iterable,
    List,
    Literal,
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

from chat_cmpl_stream_handler.events import (  # noqa: F401
    IterationCompleted,
    IterationStarted,
    LifecycleEvent,
    RunCompleted,
    RunFailed,
    StreamEvent,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    ToolResult,
)
from chat_cmpl_stream_handler.utils.tool_call import (  # noqa: F401
    args_from_tool_call as args_from_tool_call,
)

if TYPE_CHECKING:
    from openai.lib.streaming.chat._events import ChatCompletionStreamEvent


__version__: Final[Text] = "0.5.0"

logger = logging.getLogger(__name__)

ToolInvokerFn = Callable[
    [ChatCompletionMessageToolCall, Any],
    Awaitable[Union[str, ToolResult]],
]

OnToolError = Literal["emit", "raise", "abort"]

_GENERIC_TOOL_ERROR_MESSAGE: Final[str] = "Tool invocation failed."


def merge_tools_and_invokers(
    tools: "Sequence[Union[Tool, ChatCompletionToolParam]] | None" = None,
    tool_invokers: Dict[str, ToolInvokerFn] | None = None,
    stream_tools: Iterable[ChatCompletionToolParam] | None = None,
) -> Tuple[List[ChatCompletionToolParam], Dict[str, ToolInvokerFn]]:
    """Merge tool schemas and invokers.

    Schemas are collected from stream tools first, then tools. Later schemas
    with the same name replace earlier schemas. Explicit invokers replace
    invokers from packaged tools.

    Raises ValueError when any schema has no invoker.
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


async def stream_until_user_input_events(
    messages: Iterable[ChatCompletionMessageParam],
    model: Union[str, ChatModel],
    openai_client: AsyncOpenAI,
    *,
    tools: Optional[Sequence[Union["Tool", ChatCompletionToolParam]]] = None,
    tool_invokers: Optional[Dict[str, ToolInvokerFn]] = None,
    stream_kwargs: Optional[Dict[Text, Any]] = None,
    context: Optional[Any] = None,
    max_iterations: int = 10,
    fallback_invoker: Optional[Callable[[str], Optional[ToolInvokerFn]]] = None,
    on_tool_error: OnToolError = "emit",
    **kwargs,
) -> AsyncIterator["LifecycleEvent"]:
    """Run the stream loop and yield lifecycle events.

    Events mark iteration starts, raw stream events, completed model
    responses, tool invocation starts and finishes, and terminal success or
    failure.

    on_tool_error controls invoker failures. "emit" yields a failure event,
    sends a generic tool error message, and continues. "raise" yields the
    failure event and raises the original exception. "abort" yields the
    failure event, then a terminal run failure.
    """
    _validate_on_tool_error(on_tool_error)

    merged_stream_kwargs: Dict[Text, Any] = dict(stream_kwargs or {})
    stream_tools = list(merged_stream_kwargs.pop("tools", None) or [])
    resolved_invoker_input = _add_fallback_invokers(
        tools=tools,
        tool_invokers=tool_invokers,
        stream_tools=stream_tools,
        fallback_invoker=fallback_invoker,
    )
    resolved_tools, resolved_invokers = merge_tools_and_invokers(
        tools=tools,
        tool_invokers=resolved_invoker_input,
        stream_tools=stream_tools,
    )
    if resolved_tools:
        merged_stream_kwargs["tools"] = resolved_tools

    current_messages: List[ChatCompletionMessageParam] = list(messages)
    usages: List["CompletionUsage"] = []

    for index in range(max_iterations):
        yield IterationStarted(index=index, messages=list(current_messages))

        try:
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
                    yield StreamEvent(event=event)

            final = state.get_final_completion()
        except Exception as exc:
            yield RunFailed(exception=exc)
            return

        iteration_usage: Optional[CompletionUsage] = None
        if final.usage:
            iteration_usage = CompletionUsage.model_validate_json(
                final.usage.model_dump_json()
            )
            usages.append(iteration_usage)

        assistant_msg = final.choices[0].message
        assistant_param = _assistant_msg_to_param(assistant_msg)
        current_messages.append(assistant_param)

        yield IterationCompleted(
            index=index,
            usage=iteration_usage,
            assistant_message=assistant_param,
        )

        if not assistant_msg.tool_calls:
            yield RunCompleted(result=StreamResult(current_messages, model, usages))
            return

        for tool_call in assistant_msg.tool_calls:
            invoker = resolved_invokers.get(tool_call.function.name)
            if invoker is None and fallback_invoker is not None:
                # Defensive path for provider-returned names outside the schemas.
                invoker = fallback_invoker(tool_call.function.name)
            if invoker is None:
                yield RunFailed(
                    exception=ValueError(
                        f"No invoker for tool: {tool_call.function.name}"
                    )
                )
                return

            yield ToolCallStarted(iteration=index, tool_call=tool_call)

            try:
                raw_output = await invoker(tool_call, context)
            except Exception as exc:
                yield ToolCallFailed(
                    iteration=index, tool_call=tool_call, exception=exc
                )
                if on_tool_error == "raise":
                    raise
                if on_tool_error == "abort":
                    yield RunFailed(exception=exc)
                    return
                result = ToolResult(
                    content=_GENERIC_TOOL_ERROR_MESSAGE,
                    metadata={"error": repr(exc)},
                )
            else:
                result = (
                    raw_output
                    if isinstance(raw_output, ToolResult)
                    else ToolResult(content=str(raw_output))
                )

            current_messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=result.content,
                )
            )
            yield ToolCallCompleted(iteration=index, tool_call=tool_call, result=result)

    yield RunFailed(
        exception=MaxIterationsReached(
            f"Reached max_iterations={max_iterations} without waiting for user input."
        )
    )


async def stream_until_user_input(
    messages: Iterable[ChatCompletionMessageParam],
    model: Union[str, ChatModel],
    openai_client: AsyncOpenAI,
    *,
    stream_handler: Optional["ChatCompletionStreamHandler[ResponseFormatT]"] = None,
    tools: Optional[Sequence[Union["Tool", ChatCompletionToolParam]]] = None,
    tool_invokers: Optional[Dict[str, ToolInvokerFn]] = None,
    stream_kwargs: Optional[Dict[Text, Any]] = None,
    context: Optional[Any] = None,
    max_iterations: int = 10,
    tool_call_output_callback: Optional[
        Callable[[ChatCompletionMessageFunctionToolCall, str], Awaitable[None]]
    ] = None,
    fallback_invoker: Optional[Callable[[str], Optional[ToolInvokerFn]]] = None,
    on_tool_error: OnToolError = "emit",
    **kwargs,
) -> "StreamResult":
    """Run the stream loop through the callback-style API.

    Raw stream events are sent to the stream handler. Tool outputs are sent
    to the optional tool callback as plain strings. The final stream result
    is returned.
    """
    active_stream_handler = stream_handler or ChatCompletionStreamHandler()

    async for event in stream_until_user_input_events(
        messages,
        model,
        openai_client,
        tools=tools,
        tool_invokers=tool_invokers,
        stream_kwargs=stream_kwargs,
        context=context,
        max_iterations=max_iterations,
        fallback_invoker=fallback_invoker,
        on_tool_error=on_tool_error,
    ):
        if isinstance(event, StreamEvent):
            await active_stream_handler.handle(event.event)
        elif isinstance(event, ToolCallCompleted):
            if tool_call_output_callback is not None:
                await tool_call_output_callback(event.tool_call, event.result.content)
        elif isinstance(event, RunCompleted):
            return event.result
        elif isinstance(event, RunFailed):
            raise event.exception

    raise RuntimeError("stream_until_user_input_events exited without a terminal event")


@runtime_checkable
class Tool(Protocol):
    """A tool schema paired with its invoker.

    Any object with tool_param and invoke can be used as a tool. Invokers may
    return a string or ToolResult. ToolResult.content becomes the tool message
    content, and ToolResult.metadata is available on lifecycle events.
    """

    tool_param: ChatCompletionToolParam

    async def invoke(
        self, tool_call: ChatCompletionMessageToolCall, context: Any
    ) -> Union[str, ToolResult]: ...


@dataclass(frozen=True)
class FunctionTool(Tool):
    """Small tool container for a schema and invoker function."""

    tool_param: ChatCompletionToolParam
    invoker: ToolInvokerFn

    async def invoke(
        self, tool_call: ChatCompletionMessageToolCall, context: Any
    ) -> Union[str, ToolResult]:
        return await self.invoker(tool_call, context)


class StreamResult:
    """Final message history and usage data for a completed stream loop."""

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
    """Callback hooks for observing raw stream events."""

    async def handle(self, event: "ChatCompletionStreamEvent[ResponseFormatT]") -> None:
        """Route a stream event to the matching hook."""
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
    """Raised when the tool loop reaches the iteration limit."""


def _assistant_msg_to_param(assistant_msg: Any) -> ChatCompletionAssistantMessageParam:
    tool_calls_param: Dict[str, Any] = (
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
    )
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        content=assistant_msg.content,
        **tool_calls_param,
    )


def _add_fallback_invokers(
    *,
    tools: Optional[Sequence[Union["Tool", ChatCompletionToolParam]]],
    tool_invokers: Optional[Dict[str, ToolInvokerFn]],
    stream_tools: Iterable[ChatCompletionToolParam],
    fallback_invoker: Optional[Callable[[str], Optional[ToolInvokerFn]]],
) -> Optional[Dict[str, ToolInvokerFn]]:
    if fallback_invoker is None:
        return tool_invokers

    invokers = dict(tool_invokers or {})
    covered = set(invokers)
    for item in tools or ():
        if isinstance(item, Tool):
            covered.add(item.tool_param["function"]["name"])

    for name in _tool_schema_names(tools=tools, stream_tools=stream_tools):
        if name in covered:
            continue
        invoker = fallback_invoker(name)
        if invoker is not None:
            invokers[name] = invoker
            covered.add(name)

    return invokers


def _tool_schema_names(
    *,
    tools: Optional[Sequence[Union["Tool", ChatCompletionToolParam]]],
    stream_tools: Iterable[ChatCompletionToolParam],
) -> List[str]:
    names = [schema["function"]["name"] for schema in stream_tools]
    for item in tools or ():
        schema = item.tool_param if isinstance(item, Tool) else item
        names.append(schema["function"]["name"])
    return names


def _validate_on_tool_error(on_tool_error: OnToolError) -> None:
    if on_tool_error not in ("emit", "raise", "abort"):
        raise ValueError("on_tool_error must be one of: 'emit', 'raise', or 'abort'")
