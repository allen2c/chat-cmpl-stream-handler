---
id: CCSH-001
title: Expose full tool_call object to ToolInvokerFn
status: done
priority: high
created: 2026-04-01
updated: 2026-04-01
---

# Expose full tool_call object to ToolInvokerFn

## Description

`stream_until_user_input(...)` receives the provider-generated `tool_call.id` but the invoker
seam dropped it. Downstream integrations could not trace tool execution, emit SSE events with
the true id, or correlate tool traces with assistant/tool messages.

The fix passes the entire `ChatCompletionMessageToolCall` object as the first argument to
`ToolInvokerFn`, giving invokers full access to `id`, `function.name`, and `function.arguments`.

## Acceptance Criteria

- [x] `ToolInvokerFn` receives `ChatCompletionMessageToolCall` as first argument
- [x] `stream_until_user_input(...)` passes the full tool call object into invokers
- [x] `args_from_tool_call` helper added for ergonomic argument extraction
- [x] MCP-generated invokers (`_make_mcp_tool_invoker`) updated and working
- [x] Pydantic tool invokers (`_make_pydantic_tool_invoker`) updated and working
- [x] All 28 tests pass

## Action Log

### 2026-04-01
- [DECISION] ToolInvokerFn signature design
  - Chose passing entire `ChatCompletionMessageToolCall` as first arg: one-time breaking change that is future-proof; any field OpenAI adds is automatically available to invokers
  - Rejected arity inspection (2-arg vs 3-arg dispatch): fragile with decorators/partial, type system can't express the union, sets bad precedent
  - Rejected adding `tool_call_id: str` as 3rd positional arg: solves only today's need, next metadata request requires another breaking change
  - Rejected custom `ToolCallContext` dataclass: adds indirection; library is intentionally OpenAI-bound, so no need for abstraction layer
- [API] `ToolInvokerFn` changed from `Callable[[str, Any], Awaitable[str]]` to `Callable[[ChatCompletionMessageToolCall, Any], Awaitable[str]]`
- [API] Added `args_from_tool_call(tool_call) -> Dict[str, Any]` convenience helper
- Updated `ToolInvokerFn` type alias in `__init__.py`, `utils/mcp.py`, `utils/pydantic_to_tool.py`
- Updated `stream_until_user_input` call site: `await invoker(tool_call, context)`
- Updated `_make_mcp_tool_invoker` and `_make_pydantic_tool_invoker` internal closures
- Updated all test invoker signatures to use new API with `args_from_tool_call`
- All 28 tests passed

## Completion Note

2026-04-01

Original issue requested passing only `tool_call_id: str` with backward-compatible arity
inspection. Final implementation passes the full `ChatCompletionMessageToolCall` object instead,
which is a broader breaking change but eliminates future API churn. Added `args_from_tool_call`
helper to preserve ergonomics for the common case.

Deliverables:
- `chat_cmpl_stream_handler/__init__.py` — new type, helper, updated call site
- `chat_cmpl_stream_handler/utils/mcp.py` — updated invoker closure
- `chat_cmpl_stream_handler/utils/pydantic_to_tool.py` — updated invoker closure
- `tests/test_stream_until_user_input.py` — updated invoker
- `tests/test_stream_until_user_input_with_combined_tools.py` — updated invoker
- `tests/test_anthropic_compatibility.py` — updated invoker
