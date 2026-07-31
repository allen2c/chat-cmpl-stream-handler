# API Reference

Everything below is importable from the package root:

```python
from chat_cmpl_stream_handler import stream_until_user_input, ToolResult, ...
```

Both entry points below take exactly the keywords listed. An unknown one raises
`TypeError` — a typo such as `max_iteration=` is a loud error, not a silently dropped cap.
Provider-bound options belong in `stream_kwargs`.

## `stream_until_user_input`

```python
async def stream_until_user_input(
    messages: Iterable[ChatCompletionMessageParam],
    model: str,
    openai_client: AsyncOpenAI | Router | genai.Client | ChunkStreamer,
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

| Parameter                   | Description                                                                                                       |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------|
| `messages`                  | Initial message list                                                                                              |
| `model`                     | Model name                                                                                                        |
| `openai_client`             | `AsyncOpenAI`, a litellm `Router`, a `genai.Client`, or any [`ChunkStreamer`](#chunkstreamer)                     |
| `stream_handler`            | Receives raw stream events. Default: a no-op `ChatCompletionStreamHandler()`                                      |
| `tools`                     | Optional `Tool` objects or raw tool schemas                                                                       |
| `tool_invokers`             | `{"tool_name": async_fn}`. Each function takes `(tool_call, context)` and returns `str` or `ToolResult`           |
| `stream_kwargs`             | Passed straight through to the provider request                                                                    |
| `context`                   | Forwarded to every tool invoker as-is                                                                             |
| `max_iterations`            | Safety cap. Default: 10                                                                                           |
| `tool_call_output_callback` | Receives each completed tool output as a plain string                                                             |
| `fallback_invoker`          | Resolves a tool invoker by name when the normal invoker map misses                                                |
| `on_tool_error`             | `"emit"` continues with a generic tool error, `"raise"` re-raises, `"abort"` stops and raises through the adapter |

## `stream_until_user_input_events`

```python
async def stream_until_user_input_events(
    messages: Iterable[ChatCompletionMessageParam],
    model: str,
    openai_client: AsyncOpenAI | Router | genai.Client | ChunkStreamer,
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

Yields lifecycle events as the loop runs. Every event is a frozen dataclass.

| Event                | Fields                                | When                                              |
|----------------------|---------------------------------------|---------------------------------------------------|
| `IterationStarted`   | `index`, `messages`                   | A loop iteration begins                           |
| `StreamEvent`        | `event`                               | One raw stream event from the OpenAI SDK          |
| `IterationCompleted` | `index`, `usage`, `assistant_message` | The model response for this iteration is complete |
| `ToolCallStarted`    | `iteration`, `tool_call`              | A tool invoker is about to run                    |
| `ToolCallCompleted`  | `iteration`, `tool_call`, `result`    | An invoker returned a `ToolResult`                |
| `ToolCallFailed`     | `iteration`, `tool_call`, `exception` | An invoker raised                                 |
| `RunCompleted`       | `result`                              | Terminal success — carries the `StreamResult`     |
| `RunFailed`          | `exception`                           | Terminal failure                                  |

`LifecycleEvent` is the union of all of the above.

## `Tool` and `FunctionTool`

`Tool` is a runtime-checkable `Protocol` — any object carrying both a schema and its invoker qualifies:

```python
class Tool(Protocol):
    tool_param: ChatCompletionToolParam

    async def invoke(self, tool_call: ChatCompletionMessageToolCall, context: Any) -> str | ToolResult: ...
```

`FunctionTool(tool_param=..., invoker=...)` is the ready-made container. Pass either through `tools=` and skip `tool_invokers=` entirely.

When the same tool name arrives from both `tools=` and `tool_invokers=`, the explicit `tool_invokers=` entry wins and a warning is logged.

## `ToolInvokerFn`

```python
ToolInvokerFn = Callable[[ChatCompletionMessageToolCall, Any], Awaitable[str | ToolResult]]
```

Each tool invoker receives the full `ChatCompletionMessageToolCall` object from the OpenAI response. This gives you access to `tool_call.id`, `tool_call.function.name`, and `tool_call.function.arguments` — useful for tracing, logging, or emitting SSE events with the real tool call id.

## `ToolResult`

```python
ToolResult(content: str, metadata: dict[str, Any])
```

Return `ToolResult` when the tool message should be a string but the caller also needs structured metadata in `ToolCallCompleted`. `metadata` defaults to an empty dict. The callback API only exposes `content`.

## `args_from_tool_call`

```python
def args_from_tool_call(tool_call: ChatCompletionMessageToolCall) -> dict[str, Any]
```

Convenience helper that parses `tool_call.function.arguments` into a dictionary. Handles empty arguments gracefully.

## `merge_tools_and_invokers`

```python
def merge_tools_and_invokers(
    tools: Sequence[Tool | ChatCompletionToolParam] | None = None,
    tool_invokers: dict[str, ToolInvokerFn] | None = None,
    stream_tools: Iterable[ChatCompletionToolParam] | None = None,
) -> tuple[list[ChatCompletionToolParam], dict[str, ToolInvokerFn]]
```

The resolution step the loop runs internally, exposed for callers who want to inspect or pre-validate their tool wiring. Raises `ValueError` when a schema has no matching invoker.

## `ChunkStreamer`

```python
@runtime_checkable
class ChunkStreamer(Protocol):
    def stream(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]: ...
```

The only thing the loop needs from a provider: a stream of OpenAI chunks. `openai_client=`
accepts an `AsyncOpenAI`, a litellm `Router`, a `genai.Client`, or anything matching this protocol — pass a
raw client and the loop wraps it for you.

Implement it to add a provider the library doesn't know about, or to replay canned chunks
in tests. `kwargs` carries OpenAI-named request options (`tools`, `response_format`, ...)
straight through; translate shape, don't filter.

```python
class EchoStreamer:
    async def stream(self, *, messages, model, **kwargs):
        yield ChatCompletionChunk.model_validate({...})
```

## `StreamResult`

| Attribute / Method | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `.to_input_list()` | Full message history as a JSON-serializable list, ready for the next round  |
| `.usages`          | `list[CompletionUsage]` — one per iteration, so you can watch the bill grow |

`to_input_list()` raises `TypeError` on anything that is not JSON-serializable. It used
to stringify silently, which corrupted provider state instead of reporting it.

## `ChatCompletionStreamHandler`

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

## Exceptions

| Exception              | Raised when                                                                |
|------------------------|----------------------------------------------------------------------------|
| `MaxIterationsReached` | The loop hit `max_iterations` without the model settling on a final answer |
| `ValueError`           | A tool schema was passed with no invoker to match it                       |
