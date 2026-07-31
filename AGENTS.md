# AGENTS.md

A single-purpose library: stream an OpenAI-compatible chat completion and loop tool calls
until the model stops asking. No framework, no magic — just the loop.

## Layout

| Path                              | What lives there                                   |
|-----------------------------------|----------------------------------------------------|
| `chat_cmpl_stream_handler/`       | The loop, the stream handler, the lifecycle events |
| `chat_cmpl_stream_handler/utils/` | Tool builders (MCP, Pydantic) and small helpers    |
| `tests/`                          | Pytest suite — most cases hit real provider APIs   |
| `docs/`                           | Everything else. Start here.                       |

## Before you commit

```bash
make fmt     # isort → black → ruff --fix
make check   # ruff + pyright, both must be clean
make pytest  # needs API keys; see docs/development.md
```

Non-negotiable: line length is **120** everywhere, and `make check` reports **0 pyright errors**.
Do not add per-line `# type: ignore` to get there — fix the type.

## Read next

- [docs/index.md](docs/index.md) — overview and quick start
- [docs/api.md](docs/api.md) — public API reference
- [docs/tools.md](docs/tools.md) — building tools from MCP servers and Pydantic models
- [docs/providers.md](docs/providers.md) — provider quirks (Anthropic, Gemini)
- [docs/development.md](docs/development.md) — toolchain, conventions, testing, release

`README.md` is the standalone PyPI/GitHub landing page and intentionally repeats
much of `docs/`. Change public API → update both.
