"""Lifecycle event types for streaming runs.

The generator API yields these frozen dataclasses as a run progresses.
Consumers can inspect the event type and fields directly.
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
    """Structured output from a tool invoker.

    content is sent back as the tool message. metadata is kept on lifecycle
    events for callers that need extra structured data.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IterationStarted:
    """A tool-loop iteration is starting."""

    index: int
    messages: List[ChatCompletionMessageParam]


@dataclass(frozen=True)
class StreamEvent:
    """A raw stream event was received."""

    event: "ChatCompletionStreamEvent"


@dataclass(frozen=True)
class ToolCallStarted:
    """A tool invoker is about to run."""

    iteration: int
    tool_call: ChatCompletionMessageFunctionToolCall


@dataclass(frozen=True)
class ToolCallCompleted:
    """A tool invoker returned a result."""

    iteration: int
    tool_call: ChatCompletionMessageFunctionToolCall
    result: ToolResult


@dataclass(frozen=True)
class ToolCallFailed:
    """A tool invoker raised an exception."""

    iteration: int
    tool_call: ChatCompletionMessageFunctionToolCall
    exception: BaseException


@dataclass(frozen=True)
class IterationCompleted:
    """The model response for an iteration is complete."""

    index: int
    usage: CompletionUsage | None
    assistant_message: ChatCompletionAssistantMessageParam


@dataclass(frozen=True)
class RunCompleted:
    """The run completed successfully."""

    result: "StreamResult"


@dataclass(frozen=True)
class RunFailed:
    """The run ended with an exception."""

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
