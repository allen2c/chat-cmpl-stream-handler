"""Lifecycle events for ``stream_until_user_input_events``.

The async-generator API yields instances of :class:`LifecycleEvent` so
consumers can drive SSE/streaming UIs with a single ``async for`` loop
instead of juggling a callback handler plus a background task.

All events are plain frozen dataclasses — dispatch via ``isinstance`` or
``match``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Union

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.completion_usage import CompletionUsage

if TYPE_CHECKING:
    from openai.lib.streaming.chat._events import ChatCompletionStreamEvent

    from chat_cmpl_stream_handler import StreamResult


@dataclass(frozen=True)
class ToolResult:
    """Structured return type for ``Tool.invoke`` / tool invokers.

    ``content`` is the string placed into the tool message returned to the
    model. ``metadata`` is a free-form payload that rides along in
    :class:`ToolCallCompleted` events — callers use it to carry
    domain-specific objects (AFS models, MCP labels, etc.) through the
    stream without stringifying them.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IterationStarted:
    """Emitted at the top of each tool-loop iteration, before the stream begins."""

    index: int
    messages: List[ChatCompletionMessageParam]


@dataclass(frozen=True)
class StreamEvent:
    """Wraps a raw OpenAI ``ChatCompletionStreamEvent`` (content.delta, tool_calls.*, etc.)."""  # noqa: E501

    event: "ChatCompletionStreamEvent"


@dataclass(frozen=True)
class ToolCallStarted:
    """Emitted just before an invoker runs for a given tool call."""

    iteration: int
    tool_call: ChatCompletionMessageFunctionToolCall


@dataclass(frozen=True)
class ToolCallCompleted:
    """Emitted after a tool invoker returns (including the ``on_tool_error='emit'`` fallback)."""  # noqa: E501

    iteration: int
    tool_call: ChatCompletionMessageFunctionToolCall
    result: ToolResult


@dataclass(frozen=True)
class ToolCallFailed:
    """Emitted when an invoker raises. Always emitted before any error handling."""

    iteration: int
    tool_call: ChatCompletionMessageFunctionToolCall
    exception: BaseException


@dataclass(frozen=True)
class IterationCompleted:
    """Emitted after the model response for an iteration is fully received."""

    index: int
    usage: CompletionUsage | None
    assistant_message: ChatCompletionAssistantMessageParam


@dataclass(frozen=True)
class RunCompleted:
    """Terminal success event. No further events will be yielded after this."""

    result: "StreamResult"


@dataclass(frozen=True)
class RunFailed:
    """Terminal failure event. No further events will be yielded after this."""

    exception: BaseException


LifecycleEvent = Union[
    IterationStarted,
    StreamEvent,
    ToolCallStarted,
    ToolCallCompleted,
    ToolCallFailed,
    IterationCompleted,
    RunCompleted,
    RunFailed,
]
