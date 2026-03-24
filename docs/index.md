# chat-cmpl-stream-handler

[![PyPI version](https://img.shields.io/pypi/v/chat-cmpl-stream-handler.svg)](https://pypi.org/project/chat-cmpl-stream-handler/)
[![Python Version](https://img.shields.io/pypi/pyversions/chat-cmpl-stream-handler.svg)](https://pypi.org/project/chat-cmpl-stream-handler/)
[![License](https://img.shields.io/pypi/l/chat-cmpl-stream-handler.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/allen2c/chat-cmpl-stream-handler/actions/workflows/test.yml/badge.svg)](https://github.com/allen2c/chat-cmpl-stream-handler/actions/workflows/test.yml)

A lightweight Python library for handling OpenAI-compatible chat completion streams with automatic tool call execution.

## Features

- **Automatic tool call loop** — `stream_until_user_input` keeps streaming and executing tool calls until the model has no more tool calls to make
- **Event hooks** — subclass `ChatCompletionStreamHandler` to react to any stream event (content delta, tool call arguments, refusals, logprobs, etc.)
- **Usage tracking** — aggregates `CompletionUsage` across all iterations in the loop
- **Provider agnostic** — works with any OpenAI-compatible endpoint (OpenAI, Groq, Mistral, Gemini, DeepSeek, Moonshot, HuggingFace, etc.)

## Installation

```bash
pip install chat-cmpl-stream-handler
```

## Quick Start

### Basic streaming with tool calls

```python
import asyncio
import json
from openai import AsyncOpenAI
from chat_cmpl_stream_handler import (
    ChatCompletionStreamHandler,
    stream_until_user_input,
)

client = AsyncOpenAI(api_key="...")

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


async def get_weather(arguments: str, context) -> str:
    args = json.loads(arguments)
    return f"The weather in {args['city']} is sunny and 25°C."


async def main():
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    result = await stream_until_user_input(
        messages=messages,
        model="gpt-4.1-nano",
        openai_client=client,
        stream_handler=ChatCompletionStreamHandler(),
        tool_invokers={"get_weather": get_weather},
        stream_kwargs={
            "tools": [GET_WEATHER_TOOL],
            "stream_options": {"include_usage": True},
        },
    )

    # Full message history including tool calls
    for msg in result.to_input_list():
        print(msg["role"], "->", msg.get("content", ""))

    # Token usage across all iterations
    for usage in result.usages:
        print(f"tokens: {usage.total_tokens}")


asyncio.run(main())
```

### Custom event handler

Subclass `ChatCompletionStreamHandler` and override any hook you need:

```python
from chat_cmpl_stream_handler import ChatCompletionStreamHandler
from openai.lib.streaming.chat._events import (
    ContentDeltaEvent,
    FunctionToolCallArgumentsDoneEvent,
)


class MyHandler(ChatCompletionStreamHandler):
    async def on_content_delta(self, event: ContentDeltaEvent) -> None:
        print(event.delta, end="", flush=True)

    async def on_tool_calls_function_arguments_done(
        self, event: FunctionToolCallArgumentsDoneEvent
    ) -> None:
        print(f"\n[tool call] {event.name}({event.arguments})")
```

## API Reference

### `stream_until_user_input`

```python
async def stream_until_user_input(
    messages: Iterable[ChatCompletionMessageParam],
    model: str | ChatModel,
    openai_client: AsyncOpenAI,
    *,
    stream_handler: ChatCompletionStreamHandler[ResponseFormatT],
    tool_invokers: dict[str, ToolInvokerFn] | None = None,
    stream_kwargs: dict[str, Any] | None = None,
    context: Any | None = None,
    max_iterations: int = 10,
) -> StreamResult
```

Streams a chat completion, automatically executing any tool calls and feeding results back into the conversation. Returns once the model produces a response with no tool calls.

| Parameter        | Description                                                                              |
|------------------|------------------------------------------------------------------------------------------|
| `messages`       | Initial message list                                                                     |
| `model`          | Model name                                                                               |
| `openai_client`  | `AsyncOpenAI` instance                                                                   |
| `stream_handler` | Handler that receives stream events                                                      |
| `tool_invokers`  | Map of tool name → async callable `(arguments: str, context) -> str`                     |
| `stream_kwargs`  | Extra kwargs passed to `beta.chat.completions.stream()` (e.g. `tools`, `stream_options`) |
| `context`        | Arbitrary value forwarded to every tool invoker                                          |
| `max_iterations` | Raises `MaxIterationsReached` if exceeded                                                |

### `StreamResult`

| Attribute / Method | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `.to_input_list()` | Returns the full message history as a JSON-serializable list |
| `.usages`          | `list[CompletionUsage]` — one entry per streaming iteration  |

### `ChatCompletionStreamHandler`

Base class. Override any of the following async methods:

| Method                                          | Trigger                                  |
|-------------------------------------------------|------------------------------------------|
| `on_event(event)`                               | Every event (before more specific hooks) |
| `on_chunk(event)`                               | Every raw SSE chunk                      |
| `on_content_delta(event)`                       | Each content token                       |
| `on_content_done(event)`                        | Full content string complete             |
| `on_refusal_delta(event)`                       | Each refusal token                       |
| `on_refusal_done(event)`                        | Full refusal string complete             |
| `on_tool_calls_function_arguments_delta(event)` | Each incremental tool argument fragment  |
| `on_tool_calls_function_arguments_done(event)`  | Full tool argument JSON available        |
| `on_logprobs_content_delta(event)`              | Each logprobs content token              |
| `on_logprobs_content_done(event)`               | All logprobs content tokens complete     |
| `on_logprobs_refusal_delta(event)`              | Each logprobs refusal token              |
| `on_logprobs_refusal_done(event)`               | All logprobs refusal tokens complete     |

## Provider Compatibility

This library works with any OpenAI-compatible endpoint. Some providers have known quirks:

### Gemini (OpenAI-compatible endpoint)

Gemini's streaming API may return `tool_call_delta.index = None`, which causes the OpenAI SDK to crash. Apply the included opt-in patch before making any streaming requests:

```python
from chat_cmpl_stream_handler._patch_stream_tool_call_index import apply
apply()
```

Call `apply()` once at startup (e.g., at the top of your `main.py` or in your test `conftest.py`). It is safe to call multiple times.

## License

MIT
