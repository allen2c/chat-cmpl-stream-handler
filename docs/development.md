# Development

## Setup

```bash
make install   # poetry install --all-extras --all-groups
```

Python 3.12+ (`requires-python = ">=3.12"`).

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
| black  | `[tool.black]` in `pyproject.toml`     | Formatting (`target-version = ["py312"]`)                            |
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
- The `ChunkStreamer` protocol returns `AsyncIterator`, but the adapters that implement it are annotated `AsyncGenerator`. That is what makes `.aclose()` type-check where it is called; the protocol stays at the loosest contract an implementation has to meet.

## Testing

```bash
make pytest                      # everything
python -m pytest tests/test_events.py -q   # offline unit tests only
```

Offline (no network, no keys): `test_events.py`, `test_merge_tools_and_invokers.py`,
`test_tool_protocol.py`, `test_streamers.py`, `test_genai_streamer.py`. Everything else
calls a live provider.

**Real providers test provider behaviour; a scripted streamer tests loop logic.** Anything
about the loop itself — iteration limits, message history, error routing — belongs in an
offline test built on `tests/scripted.py`, which implements `ChunkStreamer` by replaying
canned chunks. Burning API calls to prove a `for` loop counts to ten is a bad trade.

A missing key skips; so does an **HTTP 402** from any provider. A depleted account is a
billing state no commit can fix, and failing on it leaves CI red until someone tops up —
`pytest_runtest_call` in `tests/conftest.py` turns it into a skip. Rate limits (429) still
fail on purpose: those say something about the code or the request.

The `llm_provider` fixture is parametrized over every entry in `PROVIDER_CONFIGS`, so one
test function becomes one run per configured provider. Prefer `@pytest.mark.parametrize`
over near-duplicate test functions — a test file should be scannable in a couple of
minutes.

Don't fan out parametrized cases against a rate-limited external MCP server; one
representative integration case is enough.

## Design docs

`docs/` documents **shipped** behaviour only — behaviour that exists in the released
package. A doc describing an unimplemented API is a doc that lies, so keep design notes and
in-flight state out of the repo entirely. When a design ships, that is when it earns a page
here.

## Releasing

1. Bump `version` in `pyproject.toml` **and** `__version__` in `chat_cmpl_stream_handler/__init__.py` — they must match.
2. `make fmt && make check && make pytest`.
3. Update `README.md` and `docs/` if the public API moved.
