---
id: CCSH-004
title: "Migrate from beta stream API to standard chat completions API"
status: done
priority: high
created: 2026-04-03
updated: 2026-04-03
closes: CCSH-003
---

# Migrate from beta stream API to standard chat completions API

## Description

Replace `openai_client.beta.chat.completions.stream()` with
`openai_client.chat.completions.create(stream=True)` combined with manual
`ChatCompletionStreamState` event emission. This removes the strict tool
validation imposed by the beta API while preserving the typed event stream
interface (`ChatCompletionStreamHandler`).

## Changes

- `stream_until_user_input()` now uses `create(stream=True)` + `ChatCompletionStreamState`
- Typed events are manually emitted via `state.handle_chunk(chunk)`
- No changes to the public API surface
- Directly resolves CCSH-003 (non-strict tools rejected by beta API)

## Acceptance Criteria

1. `stream_until_user_input` works with both strict and non-strict tools
2. All typed stream events still fire correctly
3. Existing tests pass
4. No breaking changes to public API
