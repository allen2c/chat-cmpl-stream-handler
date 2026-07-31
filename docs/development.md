# Development

## Setup

```bash
make install   # poetry install --all-extras --all-groups
```

Python 3.11+ (`requires-python = ">=3.11,<4"`).

Copy `.env.example` to `.env` and fill in whichever provider keys you have. Missing keys
make the matching tests skip, not fail.

## Commands

| Command       | What it does                                   |
|---------------|------------------------------------------------|
| `make fmt`    | isort → black → `ruff check --fix`             |
| `make check`  | `ruff check` + `pyright` — both must be clean  |
| `make pytest` | Full suite (most tests hit real provider APIs) |
| `make mkdocs` | Serve the docs site at `0.0.0.0:8000`          |
| `make update` | `poetry update`                                |

## Style

Line length is **120** across the whole toolchain. Four tools, one number:

| Tool   | Config                                 | Role                                                                 |
|--------|----------------------------------------|----------------------------------------------------------------------|
| isort  | `[tool.isort]` in `pyproject.toml`     | Import order (`profile = "black"`, `combine_as_imports = true`)      |
| black  | `[tool.black]` in `pyproject.toml`     | Formatting (`target-version = ["py311"]`)                            |
| ruff   | `[tool.ruff.lint]` in `pyproject.toml` | Linting — `B`, `E`, `F`, `W`; `E203` ignored for black compatibility |
| flake8 | `.flake8`                              | Editor-side linting only; not installed as a dev dependency          |

`.flake8` exists because flake8 does not read `pyproject.toml`. It is the single source of
truth for the VS Code flake8 extension — do not duplicate its settings into
`.vscode/settings.json`.

Ruff deliberately does not enable `I` (would fight isort) or `UP` (would rewrite the
codebase's `typing.Dict` / `Optional[...]` style).

## Type checking

`pyright` runs in `standard` mode over `chat_cmpl_stream_handler` and `tests`, configured
in `[tool.pyright]`. The tree is at **0 errors** and should stay there.

Fix the type, don't silence it. A few conventions that keep it that way:

- `ToolInvokerFn` is defined once in `chat_cmpl_stream_handler/utils/tool_call.py` and imported everywhere else. Redefining a structurally-identical alias per module makes `dict[str, ToolInvokerFn]` values incompatible across module boundaries — `Dict` is invariant.
- `StreamResult.to_input_list()` is typed `list[ChatCompletionMessageParam]` so its output can be fed straight back into `messages=`. Indexing optional keys like `["content"]` on that union is a type error by design; tests use the `as_dicts()` helper in `tests/conftest.py` to get a plain-dict view instead.
- `PydanticToolConfig` is generic over its model, so an invoker's first parameter is checked against the concrete model type.

## Testing

```bash
make pytest                      # everything
python -m pytest tests/test_events.py -q   # offline unit tests only
```

Offline (no network, no keys): `test_events.py`, `test_merge_tools_and_invokers.py`,
`test_tool_protocol.py`. Everything else calls a live provider.

The `llm_provider` fixture is parametrized over every entry in `PROVIDER_CONFIGS`, so one
test function becomes one run per configured provider. Prefer `@pytest.mark.parametrize`
over near-duplicate test functions — a test file should be scannable in a couple of
minutes.

Don't fan out parametrized cases against a rate-limited external MCP server; one
representative integration case is enough.

## Design docs and work in flight

`docs/` documents **shipped** behaviour only. Designs for unreleased work do not belong
here — a doc describing an unimplemented API is a doc that lies.

| Where          | What                                                    | In git |
|----------------|---------------------------------------------------------|--------|
| `docs/`        | Behaviour that exists in the released package            | yes    |
| `tmp/`         | Specs and design notes for work not yet implemented      | no     |
| `HANDOFF.md`   | Current branch state, decisions made, what's next        | no     |

When a design ships, move the relevant parts of its spec into `docs/` and drop the
scratch copy. If you are picking up an existing branch, read `HANDOFF.md` first.

## Releasing

1. Bump `version` in `pyproject.toml` **and** `__version__` in `chat_cmpl_stream_handler/__init__.py` — they must match.
2. `make fmt && make check && make pytest`.
3. Update `README.md` and `docs/` if the public API moved.
