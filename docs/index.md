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
import json
from openai import AsyncOpenAI
from chat_cmpl_stream_handler import ChatCompletionStreamHandler, stream_until_user_input

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


async def get_weather(arguments: str, context) -> str:
    args = json.loads(arguments)
    return f"The weather in {args['city']} is sunny and 25°C."


async def main():
    result = await stream_until_user_input(
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        model="gpt-4.1-nano",
        openai_client=client,
        stream_handler=ChatCompletionStreamHandler(),
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
from openai.lib.streaming.chat._events import (
    ContentDeltaEvent,
    FunctionToolCallArgumentsDoneEvent,
)


class PrintingHandler(ChatCompletionStreamHandler):
    async def on_content_delta(self, event: ContentDeltaEvent) -> None:
        print(event.delta, end="", flush=True)

    async def on_tool_calls_function_arguments_done(
        self, event: FunctionToolCallArgumentsDoneEvent
    ) -> None:
        print(f"\n[calling] {event.name}({event.arguments})")
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

Streams a completion, executes tool calls, feeds results back, repeats — until the model stops asking for tools. Raises `MaxIterationsReached` if you've somehow ended up in an infinite tool call loop (it happens).

| Parameter        | Description                                                                             |
|------------------|-----------------------------------------------------------------------------------------|
| `messages`       | Initial message list                                                                    |
| `model`          | Model name                                                                              |
| `openai_client`  | `AsyncOpenAI` instance                                                                  |
| `stream_handler` | Receives stream events                                                                  |
| `tool_invokers`  | `{"tool_name": async_fn}` — each fn takes `(arguments: str, context)` and returns `str` |
| `stream_kwargs`  | Passed directly to `beta.chat.completions.stream()` (e.g. `tools`, `stream_options`)    |
| `context`        | Forwarded to every tool invoker as-is                                                   |
| `max_iterations` | Safety cap. Default: 10                                                                 |

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

Anthropic's Messages API is not OpenAI-compatible. Use the included `AnthropicOpenAI` adapter — a drop-in `AsyncOpenAI` subclass that translates requests under the hood (no extra dependencies required):

```python
from chat_cmpl_stream_handler._anthropic import AnthropicOpenAI

client = AnthropicOpenAI(api_key="sk-ant-...")
result = await stream_until_user_input(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="claude-haiku-4-5-20251001",
    openai_client=client,
    stream_handler=ChatCompletionStreamHandler(),
    tool_invokers={"get_weather": get_weather},
    stream_kwargs={"tools": [GET_WEATHER_TOOL]},
)
```

A few differences from OpenAI to be aware of:

- Usage is always returned — no need to pass `stream_options: {"include_usage": True}`.
- The `strict` field in tool definitions is silently ignored (Anthropic doesn't support it).
- OpenAI-only keys (`stream_options`, `response_format`) are stripped before the request is sent.

### Gemini

Gemini's streaming API sends `tool_call_delta.index = None`, which the OpenAI SDK does not appreciate. Apply the included patch once at startup:

```python
from chat_cmpl_stream_handler._patch_stream_tool_call_index import apply
apply()  # safe to call multiple times
```

Put it at the top of `main.py`, or in `conftest.py` if you're testing. This is opt-in — the library won't silently monkey-patch anything on import.

## License

MIT
