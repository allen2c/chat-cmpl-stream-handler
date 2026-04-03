# Project Status

Last reviewed: 2026-04-03

## Active

(none)

## Recently Done

| ID       | Title                                                                     | Completed  |
|----------|---------------------------------------------------------------------------|------------|
| CCSH-004 | Migrate from beta stream API to standard chat completions API             | 2026-04-03 |
| CCSH-003 | stream_until_user_input rejects non-strict tools (resolved by CCSH-004)   | 2026-04-03 |
| CCSH-002 | Move args_from_tool_call to utils with re-export                          | 2026-04-01 |
| CCSH-001 | Expose full tool_call object to ToolInvokerFn                             | 2026-04-01 |

## Notes

v0.4.0 also removed `AnthropicOpenAI` wrapper — Anthropic now works via plain
`AsyncOpenAI(base_url="https://api.anthropic.com/v1")`.
