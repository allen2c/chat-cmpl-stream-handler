# chat-cmpl-stream-handler

[![PyPI version](https://img.shields.io/pypi/v/chat-cmpl-stream-handler.svg)](https://pypi.org/project/chat-cmpl-stream-handler/)
[![Python Version](https://img.shields.io/pypi/pyversions/chat-cmpl-stream-handler.svg)](https://pypi.org/project/chat-cmpl-stream-handler/)
[![License](https://img.shields.io/pypi/l/chat-cmpl-stream-handler.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/allen2c/chat-cmpl-stream-handler/actions/workflows/test.yml/badge.svg)](https://github.com/allen2c/chat-cmpl-stream-handler/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://allen2c.github.io/chat-cmpl-stream-handler/)

You've reimplemented the tool call loop for the fifth time. So have I. Never again.

## Why

OpenAI Responses API? Still evolving. Agents SDK? Promising — frameworks always are, at first. Chat Completions API? Boring, stable, everywhere.

This library does exactly two things that everyone keeps copy-pasting across projects:

1. Stream a chat completion and handle events
2. Keep looping tool calls until the model is done

That's it. No magic. No framework. Just the loop.

## Installation

```bash
pip install chat-cmpl-stream-handler
```

## Quick Start

```python
import asyncio
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from chat_cmpl_stream_handler import args_from_tool_call, stream_until_user_input

client = AsyncOpenAI(api_key="...")

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


async def get_weather(tool_call: ChatCompletionMessageToolCall, context) -> str:
    args = args_from_tool_call(tool_call)
    return f"The weather in {args['city']} is sunny and 25°C."


async def main():
    result = await stream_until_user_input(
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        model="gpt-4.1-nano",
        openai_client=client,
        tool_invokers={"get_weather": get_weather},
        stream_kwargs={
            "tools": [GET_WEATHER_TOOL],
            "stream_options": {"include_usage": True},
        },
    )

    # user → assistant (tool_calls) → tool → assistant (final answer)
    for msg in result.to_input_list():
        print(msg["role"], "->", msg.get("content", ""))

    for usage in result.usages:
        print(f"total tokens: {usage.total_tokens}")


asyncio.run(main())
```

### Listening to stream events

Subclass `ChatCompletionStreamHandler` and override whatever you care about:

```python
from chat_cmpl_stream_handler import ChatCompletionStreamHandler
from openai.lib.streaming.chat._events import ContentDeltaEvent, FunctionToolCallArgumentsDoneEvent


class PrintingHandler(ChatCompletionStreamHandler):
    async def on_content_delta(self, event: ContentDeltaEvent) -> None:
        print(event.delta, end="", flush=True)

    async def on_tool_calls_function_arguments_done(
        self, event: FunctionToolCallArgumentsDoneEvent
    ) -> None:
        print(f"\n[calling] {event.name}({event.arguments})")
```

If you want one async stream of everything that happens in the loop, use
`stream_until_user_input_events`:

```python
from chat_cmpl_stream_handler import (
    RunCompleted,
    StreamEvent,
    ToolCallCompleted,
    stream_until_user_input_events,
)


async for event in stream_until_user_input_events(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="gpt-4.1-nano",
    openai_client=client,
    tool_invokers={"get_weather": get_weather},
    stream_kwargs={"tools": [GET_WEATHER_TOOL]},
):
    if isinstance(event, StreamEvent) and event.event.type == "content.delta":
        print(event.event.delta, end="")
    elif isinstance(event, ToolCallCompleted):
        print("tool result:", event.result.content)
    elif isinstance(event, RunCompleted):
        result = event.result
```

### Building tools from MCP servers

If you already expose capabilities through an MCP server, you can turn them into
OpenAI-compatible `tools` plus `tool_invokers` in one step:

```python
from chat_cmpl_stream_handler.utils.mcp import MCPServerConfig, build_mcp_tools_and_invokers


mcp_tools, mcp_tool_invokers = await build_mcp_tools_and_invokers(
    [
        MCPServerConfig(
            server_url="https://marketplace-mcp.us-east-1.api.aws/mcp",
            server_label="aws",
        )
    ]
)

result = await stream_until_user_input(
    messages=[{"role": "user", "content": "Use aws__get_cost_and_usage and summarize it."}],
    model="gpt-4.1",
    openai_client=client,
    tool_invokers=mcp_tool_invokers,
    stream_kwargs={"tools": mcp_tools},
)
```

Notes:

- `server_label="aws"` prefixes discovered tools like `aws__tool_name`
- if you pass an initialized `ClientSession` into `MCPServerConfig(session=...)`,
  tool discovery and tool calls reuse that session without reconnecting
- runtime `context` from `stream_until_user_input(..., context=...)` is forwarded
  into MCP `meta["context"]`

### Building tools from Pydantic models

For local tools with typed inputs, use the Pydantic helpers directly from
`chat_cmpl_stream_handler.utils`:

```python
from typing import Any

from pydantic import BaseModel

from chat_cmpl_stream_handler.utils.pydantic_to_tool import (
    PydanticToolConfig,
    build_pydantic_tools_and_invokers,
)


class EchoRequest(BaseModel):
    """Echo text back to the user."""

    text: str


async def echo_tool(arguments: EchoRequest, context: Any) -> str:
    return f"{context}: {arguments.text}"


pydantic_tools, pydantic_tool_invokers = build_pydantic_tools_and_invokers(
    [
        PydanticToolConfig(
            model=EchoRequest,
            invoker=echo_tool,
        )
    ]
)

result = await stream_until_user_input(
    messages=[{"role": "user", "content": "Call echo_request with text=hello"}],
    model="gpt-4.1",
    openai_client=client,
    tool_invokers=pydantic_tool_invokers,
    stream_kwargs={"tools": pydantic_tools},
    context="demo",
)
```

The generated invoker validates the tool arguments with
`model_validate_json(...)` before calling your handler.

## API Reference

### `stream_until_user_input`

```python
async def stream_until_user_input(
    messages: Iterable[ChatCompletionMessageParam],
    model: str | ChatModel,
    openai_client: AsyncOpenAI,
    *,
    stream_handler: ChatCompletionStreamHandler[ResponseFormatT] | None = None,
    tools: Sequence[Tool | ChatCompletionToolParam] | None = None,
    tool_invokers: dict[str, ToolInvokerFn] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    context: Any | None = None,
    max_iterations: int = 10,
    tool_call_output_callback: Callable[[ChatCompletionMessageFunctionToolCall, str], Awaitable[None]] | None = None,
    fallback_invoker: Callable[[str], ToolInvokerFn | None] | None = None,
    on_tool_error: Literal["emit", "raise", "abort"] = "emit",
) -> StreamResult
```

Streams a completion, executes tool calls, feeds results back, repeats — until the model stops asking for tools. Raises `MaxIterationsReached` if you've somehow ended up in an infinite tool call loop.

| Parameter                   | Description                                                                                                           |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `messages`                  | Initial message list                                                                                                  |
| `model`                     | Model name                                                                                                            |
| `openai_client`             | `AsyncOpenAI` instance                                                                                                |
| `stream_handler`            | Receives raw stream events. Default: a no-op `ChatCompletionStreamHandler()`                                          |
| `tools`                     | Optional `Tool` objects or raw tool schemas                                                                           |
| `tool_invokers`             | `{"tool_name": async_fn}`. Each function takes `(tool_call, context)` and returns `str` or `ToolResult`               |
| `stream_kwargs`             | Passed directly to `chat.completions.create()`                                                                        |
| `context`                   | Forwarded to every tool invoker as-is                                                                                 |
| `max_iterations`            | Safety cap. Default: 10                                                                                               |
| `tool_call_output_callback` | Receives each completed tool output as a plain string                                                                 |
| `fallback_invoker`          | Resolves a tool invoker by name when the normal invoker map misses                                                    |
| `on_tool_error`             | `"emit"` continues with a generic tool error, `"raise"` re-raises, `"abort"` stops and raises through the adapter     |

### `stream_until_user_input_events`

```python
async def stream_until_user_input_events(
    messages: Iterable[ChatCompletionMessageParam],
    model: str | ChatModel,
    openai_client: AsyncOpenAI,
    *,
    tools: Sequence[Tool | ChatCompletionToolParam] | None = None,
    tool_invokers: dict[str, ToolInvokerFn] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    context: Any | None = None,
    max_iterations: int = 10,
    fallback_invoker: Callable[[str], ToolInvokerFn | None] | None = None,
    on_tool_error: Literal["emit", "raise", "abort"] = "emit",
) -> AsyncIterator[LifecycleEvent]
```

Yields lifecycle events as the loop runs:

- `IterationStarted`
- `StreamEvent`
- `IterationCompleted`
- `ToolCallStarted`
- `ToolCallCompleted`
- `ToolCallFailed`
- `RunCompleted`
- `RunFailed`

### `ToolInvokerFn`

```python
ToolInvokerFn = Callable[[ChatCompletionMessageToolCall, Any], Awaitable[str | ToolResult]]
```

Each tool invoker receives the full `ChatCompletionMessageToolCall` object from the OpenAI response. This gives you access to `tool_call.id`, `tool_call.function.name`, and `tool_call.function.arguments` — useful for tracing, logging, or emitting SSE events with the real tool call id.

### `ToolResult`

```python
ToolResult(content: str, metadata: dict[str, Any])
```

Return `ToolResult` when the tool message should be a string but the caller also needs structured metadata in `ToolCallCompleted`. `metadata` defaults to an empty dict. The callback API only exposes `content`.

### `args_from_tool_call`

```python
def args_from_tool_call(tool_call: ChatCompletionMessageToolCall) -> dict[str, Any]
```

Convenience helper that parses `tool_call.function.arguments` into a dictionary. Handles empty arguments gracefully.

### `StreamResult`

| Attribute / Method | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `.to_input_list()` | Full message history as a JSON-serializable list, ready for the next round  |
| `.usages`          | `list[CompletionUsage]` — one per iteration, so you can watch the bill grow |

### `ChatCompletionStreamHandler`

All methods are no-ops by default. Override only what you need.

| Method                                          | When it fires                           |
|-------------------------------------------------|-----------------------------------------|
| `on_event(event)`                               | Every event, before more specific hooks |
| `on_chunk(event)`                               | Every raw SSE chunk                     |
| `on_content_delta(event)`                       | Each content token                      |
| `on_content_done(event)`                        | Full content string complete            |
| `on_refusal_delta(event)`                       | Each refusal token                      |
| `on_refusal_done(event)`                        | Full refusal string complete            |
| `on_tool_calls_function_arguments_delta(event)` | Each incremental tool argument fragment |
| `on_tool_calls_function_arguments_done(event)`  | Full tool argument JSON available       |
| `on_logprobs_content_delta(event)`              | Each logprobs content token             |
| `on_logprobs_content_done(event)`               | All logprobs content tokens done        |
| `on_logprobs_refusal_delta(event)`              | Each logprobs refusal token             |
| `on_logprobs_refusal_done(event)`               | All logprobs refusal tokens done        |

## Provider Compatibility

Works with any OpenAI-compatible endpoint. Some providers are more compatible than others.

### Anthropic

Anthropic exposes an OpenAI-compatible endpoint — no adapter needed. Use a plain `AsyncOpenAI` with the Anthropic base URL:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-ant-...", base_url="https://api.anthropic.com/v1")
result = await stream_until_user_input(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="claude-haiku-4-5-20251001",
    openai_client=client,
    tool_invokers={"get_weather": get_weather},
    stream_kwargs={
        "tools": [GET_WEATHER_TOOL],
        "stream_options": {"include_usage": True},
    },
)
```

### Gemini

Gemini's streaming API sends `tool_call_delta.index = None`, which the OpenAI SDK does not appreciate. Apply the included patch once at startup:

```python
from chat_cmpl_stream_handler._patch_stream_tool_call_index import apply
apply()  # safe to call multiple times
```

Put it at the top of `main.py`, or in `conftest.py` if you're testing. This is opt-in — the library won't silently monkey-patch anything on import.

**Gemini 3 thought signatures:** Gemini 3 models require a `thought_signature` to be echoed back during multi-turn function calling. `stream_until_user_input` preserves these signatures automatically — no action needed on your side.

## License

MIT
