# Provider Compatibility

Works with any OpenAI-compatible endpoint. Some providers are more compatible than others.

The test suite runs against these providers when the matching key is set — see
`PROVIDER_CONFIGS` in `tests/conftest.py`:

| Provider     | Env var             | Base URL                                                   |
|--------------|---------------------|------------------------------------------------------------|
| OpenAI       | `OPENAI_API_KEY`    | default                                                    |
| Groq         | `GROQ_API_KEY`      | `https://api.groq.com/openai/v1`                           |
| Mistral      | `MISTRAL_API_KEY`   | `https://api.mistral.ai/v1`                                |
| Moonshot     | `MOONSHOT_API_KEY`  | `https://api.moonshot.ai/v1`                               |
| DeepSeek     | `DEEPSEEK_API_KEY`  | `https://api.deepseek.com`                                 |
| Gemini       | `GEMINI_API_KEY`    | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Hugging Face | `HF_TOKEN`          | `https://router.huggingface.co/v1`                         |
| Anthropic    | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1`                             |
| xAI          | `XAI_API_KEY`       | `https://api.x.ai/v1`                                      |

## Usage data

The standard Chat Completions API does not return usage on streamed responses unless you
ask for it. If `result.usages` is empty, add:

```python
stream_kwargs={"stream_options": {"include_usage": True}}
```

## Anthropic

Anthropic exposes an OpenAI-compatible endpoint — no adapter needed. Use a plain
`AsyncOpenAI` with the Anthropic base URL:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-ant-...", base_url="https://api.anthropic.com/v1")
result = await stream_until_user_input(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="claude-haiku-4-5-20251001",
    openai_client=client,
    tool_invokers={"get_weather": get_weather},
    stream_kwargs={
        "tools": [GET_WEATHER_TOOL],
        "stream_options": {"include_usage": True},
    },
)
```

## Gemini

Gemini's streaming API sends `tool_call_delta.index = None`, which the OpenAI SDK does not
appreciate. Apply the included patch once at startup:

```python
from chat_cmpl_stream_handler._patch_stream_tool_call_index import apply
apply()  # safe to call multiple times
```

Put it at the top of `main.py`, or in `conftest.py` if you're testing. This is opt-in — the
library won't silently monkey-patch anything on import.

**Gemini 3 thought signatures:** Gemini 3 models require a `thought_signature` to be echoed
back during multi-turn function calling. `stream_until_user_input` preserves these
signatures automatically — no action needed on your side.

## DeepSeek

No structured output support at time of writing; the `response_format` JSON-schema test is
skipped for DeepSeek models.

## Strict mode

The loop calls `chat.completions.create(stream=True)` rather than the beta streaming
helper, precisely so non-strict tool schemas (MCP servers, hand-written tools without
`strict: True`) are accepted. If a provider rejects a schema, that rejection is the
provider's, not this library's.
