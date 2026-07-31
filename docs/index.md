# chat-cmpl-stream-handler

[![PyPI version](https://img.shields.io/pypi/v/chat-cmpl-stream-handler.svg)](https://pypi.org/project/chat-cmpl-stream-handler/)
[![Python Version](https://img.shields.io/pypi/pyversions/chat-cmpl-stream-handler.svg)](https://pypi.org/project/chat-cmpl-stream-handler/)
[![License](https://img.shields.io/pypi/l/chat-cmpl-stream-handler.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/allen2c/chat-cmpl-stream-handler/actions/workflows/test.yml/badge.svg)](https://github.com/allen2c/chat-cmpl-stream-handler/actions/workflows/test.yml)

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

Requires Python 3.12+.

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

## Not just OpenAI

`openai_client=` takes three kinds of client. Two of them are not OpenAI-compatible
endpoints — the loop adapts them:

```python
from google import genai
from litellm.router import Router

openai_client=AsyncOpenAI(api_key="...")        # or any compatible base_url
openai_client=Router(model_list=[...])          # model= is the deployment name
openai_client=genai.Client(api_key="...")       # translated in both directions
```

Tool calls, streaming, structured output and Gemini 3 thought signatures work on all
three. Bring a fourth by implementing [`ChunkStreamer`](api.md#chunkstreamer) — it is one
method, and the loop needs nothing else from a provider.

## Two ways to observe the loop

Both APIs run the same loop; pick whichever fits your call site.

| API                              | Style           | Use when                                                           |
|----------------------------------|-----------------|--------------------------------------------------------------------|
| `stream_until_user_input`        | Callback        | You want the final `StreamResult` and only care about some events  |
| `stream_until_user_input_events` | Async generator | You want every lifecycle event in one stream (SSE relays, tracing) |

### Callbacks

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

### Lifecycle events

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

## Next

- [API Reference](api.md) — every public function, type, and hook
- [Building Tools](tools.md) — MCP servers and Pydantic models as tools
- [Provider Compatibility](providers.md) — litellm, native genai, and per-provider quirks
- [Development](development.md) — toolchain and conventions for contributors

## License

MIT
