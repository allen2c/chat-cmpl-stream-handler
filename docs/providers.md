# Provider Compatibility

Works with any OpenAI-compatible endpoint — some providers are more compatible than others
— plus two clients that are not compatible at all and get an adapter instead: a litellm
`Router` and a native `genai.Client`.

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

## Native google-genai

A `genai.Client` works too, and it is the only path that is not an OpenAI-compatible
endpoint underneath — the loop translates in both directions. Tool calls, streaming,
structured output and Gemini 3 thought signatures all work; tool-call ids stay clean.

```python
from google import genai

client = genai.Client(api_key="...")
result = await stream_until_user_input(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="gemini-3.1-flash-lite-preview",
    openai_client=client,
    tool_invokers={"get_weather": get_weather},
    stream_kwargs={"tools": [GET_WEATHER_TOOL]},
)
```

`stream_kwargs` keeps its OpenAI spelling: `max_tokens`, `stop`, `response_format`,
`tool_choice` are renamed onto their genai equivalents, and anything the mapping does not
recognise is forwarded to `GenerateContentConfig` untouched — so genai-native options such
as `thinking_config` work, and a typo is rejected instead of silently ignored.
`stream_options`, `parallel_tool_calls` and `user` have no genai equivalent and are dropped.

Structured output is close to a passthrough: `response_format` becomes
`response_json_schema` plus `response_mime_type="application/json"`, with no JSON Schema
dialect rewrite.

**Thought signatures.** Gemini 3 requires the signature attached to a part to come back on
the next turn. The loop carries it in `provider_specific_fields` — on the tool call for a
function-call part, on the message for anything else — base64-encoded so it survives
`to_input_list()`. Replay the history the loop gives you and it just works; strip those
fields and Gemini rejects the next turn.

## litellm Router

A litellm `Router` can be passed straight to `openai_client=`. The loop wraps it and
re-validates each chunk into a `ChatCompletionChunk`; everything litellm puts in
`provider_specific_fields` rides through untouched.

```python
from litellm.router import Router

router = Router(
    model_list=[
        {
            "model_name": "flash",
            "litellm_params": {"model": "gemini/gemini-3.1-flash-lite-preview", "api_key": "..."},
        }
    ]
)
result = await stream_until_user_input(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    model="flash",  # the deployment name, not the upstream model id
    openai_client=router,
    tool_invokers={"get_weather": get_weather},
    stream_kwargs={"tools": [GET_WEATHER_TOOL]},
)
```

**Tool-call ids look strange on Gemini 3.** litellm smuggles the thought signature into the
id itself:

```
call_abc__thought__EjQKMgERTTIP5bg...
```

That id reaches your invokers, your logs, and `ToolCallStarted.tool_call.id`. It is
deliberate — litellm strips it again on the way back in, and that round-trip is what keeps
multi-turn function calling working. Pass it through verbatim; do not clean it up before
handing messages back to the loop.

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
appreciate. The loop normalises the index to its positional order before the SDK ever sees
the chunk — nothing to do on your side.

Before 0.6.0 this needed an opt-in monkey-patch
(`_patch_stream_tool_call_index.apply()`). That function is now a no-op; delete the call.

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
