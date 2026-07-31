# Building Tools

The loop needs two things per tool: an OpenAI tool schema and an async invoker. You can
hand-write both, or generate them from something you already have.

| Source         | Helper                              | Module                                            |
|----------------|-------------------------------------|---------------------------------------------------|
| MCP server     | `build_mcp_tools_and_invokers`      | `chat_cmpl_stream_handler.utils.mcp`              |
| Pydantic model | `build_pydantic_tools_and_invokers` | `chat_cmpl_stream_handler.utils.pydantic_to_tool` |
| Hand-written   | `FunctionTool`                      | `chat_cmpl_stream_handler`                        |

Both builders return the same `(tools, tool_invokers)` pair, so you can merge several
sources into one call.

## From MCP servers

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

`MCPServerConfig` fields:

| Field           | Purpose                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------|
| `server_url`    | Base URL. Transport (Streamable HTTP vs SSE) is auto-detected and cached                   |
| `server_label`  | Prefixes discovered tools as `{label}__{tool_name}`; stripped again before the remote call |
| `meta`          | Static MCP `meta` sent with every call                                                     |
| `extra_headers` | Auth or custom headers; works across both transports                                       |
| `session`       | An already-initialized `ClientSession` to reuse instead of reconnecting                    |
| `filter_tool`   | `(ToolParam) -> bool` predicate to narrow what gets exposed                                |

Notes:

- Runtime `context` from `stream_until_user_input(..., context=...)` is forwarded into MCP `meta["context"]`. If `meta` already has a `context` key, it is overwritten and a warning is logged.
- Discovered schemas are rewritten for OpenAI strict mode: `title` stripped, `additionalProperties: false` injected, all properties listed in `required`.
- Lower-level entry points are available if you don't want the builder: `list_mcp_tools()`, `call_mcp_tool()`, and `clear_endpoint_cache()`.

## From Pydantic models

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

The generated invoker validates the tool arguments with `model_validate_json(...)` before
calling your handler, so `arguments` arrives typed.

`PydanticToolConfig` is generic over the model, so your handler's first parameter can be
the concrete model type — type checkers will hold you to it:

```python
PydanticToolConfig(model=EchoRequest, invoker=echo_tool)   # echo_tool: (EchoRequest, Any) -> str
```

Defaults: the tool name is the model name in snake_case (`EchoRequest` → `echo_request`),
and the description is the model docstring. Override with `name=` / `description=`.

## Hand-written tools

Pair a schema with an invoker using `FunctionTool` and pass it through `tools=`:

```python
from chat_cmpl_stream_handler import FunctionTool

result = await stream_until_user_input(
    messages=[...],
    model="gpt-4.1-nano",
    openai_client=client,
    tools=[FunctionTool(tool_param=GET_WEATHER_TOOL, invoker=get_weather)],
)
```

Or keep them apart — `stream_kwargs={"tools": [...]}` for schemas, `tool_invokers={...}`
for the functions. Both styles can be mixed in one call.

## Resolving unknown tool names

Some providers return tool names that were never in your schema list. `fallback_invoker`
gets a last shot at resolving them:

```python
result = await stream_until_user_input(
    messages=[...],
    model=...,
    openai_client=client,
    tools=[...],
    fallback_invoker=lambda name: my_registry.get(name),
)
```

A `Tool` passed via `tools=` always wins over the fallback — the fallback only fills gaps.
