# AGENTS.md

Stream an OpenAI-compatible chat completion and loop tool calls until the model stops
asking. No framework, no magic — just the loop.

**Everything is in [docs/](docs/index.md). Start there.**

- [docs/index.md](docs/index.md) — overview and quick start
- [docs/api.md](docs/api.md) — public API reference
- [docs/tools.md](docs/tools.md) — tools from MCP servers and Pydantic models
- [docs/providers.md](docs/providers.md) — provider quirks
- [docs/development.md](docs/development.md) — toolchain, conventions, testing, release

## Layout

| Path                              | What lives there                                   |
|-----------------------------------|----------------------------------------------------|
| `chat_cmpl_stream_handler/`       | The loop, the stream handler, the lifecycle events |
| `chat_cmpl_stream_handler/utils/` | Tool builders (MCP, Pydantic) and small helpers    |
| `tests/`                          | Pytest suite — most cases hit real provider APIs   |

## Before you commit

```bash
make fmt     # isort → black → ruff --fix
make check   # ruff + pyright, both must be clean
make pytest  # needs API keys; see docs/development.md
```

Line length is **120** everywhere and `make check` reports **0 pyright errors**.
Do not reach for `# type: ignore` — fix the type.

## Conventions worth knowing up front

- `docs/` describes **shipped** behaviour only. Designs for unreleased work go in `tmp/`,
  and `HANDOFF.md` tracks in-flight state — both are gitignored. Check `HANDOFF.md` first
  if work is already underway.
- `README.md` is the standalone PyPI/GitHub landing page and intentionally repeats much of
  `docs/`. Change public API → update both.
