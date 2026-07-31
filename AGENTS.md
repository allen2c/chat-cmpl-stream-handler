# AGENTS.md

Stream an OpenAI-compatible chat completion and loop tool calls until the model stops
asking. No framework, no magic — just the loop.

**Everything is in [docs/](docs/index.md). Start there.**

| Doc                                        | Read it when                                     |
|--------------------------------------------|--------------------------------------------------|
| [index](docs/index.md)                     | You need the overview or a quick start           |
| [api](docs/api.md)                         | You are changing or calling the public API       |
| [tools](docs/tools.md)                     | The task involves MCP servers or Pydantic tools  |
| [providers](docs/providers.md)             | A provider behaves oddly, or you are adding one  |
| [development](docs/development.md)         | Before you write code — toolchain and conventions |

## Layout

| Path                              | What lives there                                     |
|-----------------------------------|------------------------------------------------------|
| `chat_cmpl_stream_handler/`       | The loop, the stream adapters, the lifecycle events  |
| `chat_cmpl_stream_handler/utils/` | Tool builders (MCP, Pydantic) and small helpers      |
| `tests/`                          | Pytest suite — real provider APIs and offline tests  |

## Non-negotiables

```bash
make fmt     # isort → black → ruff --fix
make check   # ruff + pyright — must report 0 errors
make pytest  # needs API keys; see docs/development.md
```

- Python **3.12+**, line length **120**.
- Fix the type. `# type: ignore` is not an option.
- `docs/` describes **shipped** behaviour only. A doc for an unimplemented API is a doc
  that lies — keep design notes for unreleased work out of the repo.
- `README.md` is the standalone PyPI/GitHub landing page and intentionally repeats much of
  `docs/`. Change the public API → update both.
