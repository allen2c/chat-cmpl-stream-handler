---
id: CCSH-003
title: "stream_until_user_input rejects non-strict tools due to beta API validation"
status: done
resolved: 2026-04-03
resolved-by: CCSH-004
priority: high
created: 2026-04-03
updated: 2026-04-03
---

# stream_until_user_input rejects non-strict tools due to beta API validation

## Description

`stream_until_user_input()` uses `openai_client.beta.chat.completions.stream()` (line 89 of
`__init__.py`). The beta streaming API internally calls `validate_input_tools()` from
`openai.lib._parsing._completions`, which **requires every tool in the `tools` list to have
`"strict": true`**. If any tool has `strict` absent or set to a non-`True` value, the SDK
raises:

```
ValueError: `<tool_name>` is not strict. Only `strict` function tools can be auto-parsed
```

This makes it impossible to use MCP tools (or any externally-defined tool) whose JSON Schema
contains constructs incompatible with OpenAI strict mode — for example, free-form objects
(`"type": "object"` without `"properties"`), `patternProperties`, or open
`additionalProperties`.

## Reproduction

In `aao` (downstream consumer), `ChatRunner.stream()` calls `stream_until_user_input` with
a mix of AFS tools (strict-compatible) and MCP tools. One MCP tool
(`aiellobackendmcp_tms_task__create_task`) has a `fields` parameter defined as a free-form
object by the MCP server. The caller (`aao/chat_runner/tools.py`) already detects this and
correctly removes `"strict"` from that tool's schema via `normalize_chat_tool_schema()`.
However, the beta API rejects the entire request because of this single non-strict tool.

**Traceback:**

```
File "chat_cmpl_stream_handler/__init__.py", line 89, in stream_until_user_input
    async with openai_client.beta.chat.completions.stream(
File "openai/resources/chat/completions/completions.py", line 3012, in stream
    _validate_input_tools(tools)
File "openai/lib/_parsing/_completions.py", line 79, in validate_input_tools
    raise ValueError(
ValueError: `aiellobackendmcp_tms_task__create_task` is not strict.
  Only `strict` function tools can be auto-parsed
```

## Analysis

The beta API (`openai_client.beta.chat.completions.stream()`) provides two features over the
non-beta API (`openai_client.chat.completions.create(stream=True)`):

1. **Auto-parsing** of structured output / tool call arguments into Pydantic models
2. **Typed event iteration** via `async for event in stream`

`stream_until_user_input` does **not** use feature (1) — tool call arguments are passed as
raw strings to `ToolInvokerFn`. The typed event iteration (2) is valuable, but the strict
validation is an unwanted side effect.

### Options considered

| # | Approach | Pros | Cons |
|---|----------|------|------|
| A | Switch to non-beta `client.chat.completions.create(stream=True)` | Removes strict constraint entirely | Loses typed event stream; need to manually iterate SSE chunks and reconstruct `ChatCompletionStreamEvent` |
| B | Keep beta API but split tools: pass only strict tools to `beta.stream()`, handle non-strict tools separately | Preserves typed events for strict tools | Complex; tool calls can reference any tool, splitting doesn't work |
| **C** | **Keep beta API, strip `strict` from all tools before passing to SDK, skip auto-parse validation** | Minimal change; preserves typed events; no behavioral change | Relies on SDK internals not changing the validation path |

## Proposed Solution (Option C)

Before passing `tools` to `openai_client.beta.chat.completions.stream()`, remove the
`strict` key from every tool's function schema. The beta API only validates `strict` tools —
tools without the key are not validated at all (the check is `strict is not True`).

Wait — re-reading `validate_input_tools`: it checks `if strict is not True: raise`. This
means tools **without** `strict` are also rejected. So stripping `strict` from all tools
won't work.

**Revised approach:** Use the non-beta streaming API. The non-beta
`client.chat.completions.create(model=..., stream=True)` returns an `AsyncStream` of raw
`ChatCompletionChunk` objects. The `ChatCompletionStreamState` helper (used internally by
the beta API) can be instantiated manually to get the same typed event stream without the
strict validation.

Alternatively, the simplest fix: check whether all tools are strict. If yes, use beta. If
any tool is non-strict, fall back to `client.chat.completions.create(stream=True)` with
manual chunk handling.

## Acceptance Criteria

1. `stream_until_user_input` accepts a mix of strict and non-strict tools without raising
2. Typed stream events (`ContentDeltaEvent`, `ContentDoneEvent`, etc.) still work for all
   code paths
3. Tool invocation behavior is unchanged (raw string arguments, same `ToolInvokerFn` contract)
4. No breaking changes to the public API
5. Existing tests continue to pass
