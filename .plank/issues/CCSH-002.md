---
id: CCSH-002
title: Move args_from_tool_call to utils with re-export
status: done
priority: medium
created: 2026-04-01
updated: 2026-04-01
---

# Move args_from_tool_call to utils with re-export

## Description

Extract `args_from_tool_call` from `__init__.py` into `utils/tool_call.py` and re-export from `__init__.py` to maintain backward compatibility. This follows the "convenience re-export" pattern used by major Python packages.

## Acceptance Criteria

- [x] `args_from_tool_call` defined in `utils/tool_call.py`
- [x] `__init__.py` re-exports `args_from_tool_call` from utils
- [x] All existing imports (`from chat_cmpl_stream_handler import args_from_tool_call`) still work
- [x] All tests pass

## Action Log

### 2026-04-01
- Created `chat_cmpl_stream_handler/utils/tool_call.py` with `args_from_tool_call` implementation
- Updated `__init__.py` to import and re-export via `as` alias for explicit re-export
- Verified both import paths work: top-level and `utils.tool_call` direct
- All tests pass (5 passed, 20 skipped due to missing API keys in CI, 3 errors pre-existing from missing `OPENAI_API_KEY`)
- [DECISION] Convenience re-export pattern
  - Chose re-export from `__init__.py`: frequently used by consumers, backward compatible
  - Rejected utils-only access: would break all existing imports across tests, docs, README
