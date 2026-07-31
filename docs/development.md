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
in `[tool.pyright]`. The tree is at **0 errors** and should stay there — the `Check`
workflow runs `make check` on every PR into `main`, so it is enforced, not just intended.
`pyright` is a dev dependency for that reason; a locally-installed one (homebrew, npm) will
shadow it and may be a different version than CI resolves.

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
`test_tool_protocol.py`, `test_streamers.py`, `test_genai_streamer.py`,
`test_public_api.py`, `test_stream_handler_hooks.py`, `test_tool_timeout.py`,
`test_out_of_credit.py`. Everything else calls a live provider.

**Real providers test provider behaviour; a scripted streamer tests loop logic.** Anything
about the loop itself — iteration limits, message history, error routing — belongs in an
offline test built on `tests/scripted.py`, which implements `ChunkStreamer` by replaying
canned chunks. Burning API calls to prove a `for` loop counts to ten is a bad trade.

A missing key skips, and so does **an account that cannot pay**. That is a billing state
no commit can fix; failing on it leaves CI red until someone tops up, and a permanently red
CI is one nobody reads. A rate limit the account *could* pay for still fails on purpose.

Telling the two apart needs the provider's words, not its status code: Gemini bills a
project spend cap as **429 `RESOURCE_EXHAUSTED`**, and OpenAI reports being out of credit
as a 429 too. `is_out_of_credit()` in `tests/conftest.py` matches a short, deliberately
narrow list of phrases — `tests/test_out_of_credit.py` pins it against real response text
from all three client paths, and against the rate limits and outages it must *not* swallow.
Widen that list only with a real message in hand.

Two entry points, because a provider failure takes two shapes:

| Shape                              | Caught by                                    |
|------------------------------------|----------------------------------------------|
| The exception reaches the test      | `pytest_runtest_call`, automatically          |
| The events API returns `RunFailed`  | `skip_if_out_of_credit(events)`, called by hand |

The events API reports provider failures as a terminal `RunFailed` instead of raising, so
nothing reaches the hook. A test that collects lifecycle events from a live provider has to
pass them through `skip_if_out_of_credit` itself — otherwise a depleted account surfaces as
whatever that test asserted next.

The `llm_provider` fixture is parametrized over every entry in `PROVIDER_CONFIGS`, so one
test function becomes one run per configured provider. Prefer `@pytest.mark.parametrize`
over near-duplicate test functions — a test file should be scannable in a couple of
minutes.

Don't fan out parametrized cases against a rate-limited external MCP server; one
representative integration case is enough.

## CI

Two workflows, both on push to `main` and on PRs into `main`:

| Workflow | Runs         | Needs keys                        |
|----------|--------------|-----------------------------------|
| `Check`  | `make check` | No — so it is readable in isolation |
| `Tests`  | `pytest -v`  | Yes, from repository secrets       |

They are separate on purpose. `Tests` calls live providers and can go red for reasons that
have nothing to do with the diff; `Check` never leaves the runner, so a red `Check` always
means the code.

## Design docs

`docs/` documents **shipped** behaviour only — behaviour that exists in the released
package. A doc describing an unimplemented API is a doc that lies, so keep design notes and
in-flight state out of the repo entirely. When a design ships, that is when it earns a page
here.

## Releasing

1. Bump `version` in `pyproject.toml` **and** `__version__` in `chat_cmpl_stream_handler/__init__.py` — they must match.
2. `make fmt && make check && make pytest`.
3. Update `README.md` and `docs/` if the public API moved.
