import os
from dataclasses import dataclass
from typing import Any, Iterable

import pytest
from openai import APIStatusError, AsyncOpenAI

from chat_cmpl_stream_handler import RunFailed
from chat_cmpl_stream_handler._patch_stream_tool_call_index import apply

apply()


# Providers say "you cannot pay" in their own words, and not always with a 402. Gemini
# bills a spend cap as 429 RESOURCE_EXHAUSTED, which is indistinguishable from a rate
# limit by status code alone. Matching the words is precise where the code is not.
#
# Deliberately narrow. A false positive here turns a real failure into a green run, which
# is worse than a red CI — bare "quota" and "billing" are out, because a per-minute quota
# message and a billing help-link both contain them.
_CANNOT_PAY_MARKERS: tuple[str, ...] = (
    "spending cap",  # Gemini / AI Studio project spend cap
    "spend cap",
    "insufficient_quota",  # OpenAI, out of credit — also arrives as a 429
    "exceeded your current quota",
    "included credits",  # Hugging Face
    "payment required",
)


def is_out_of_credit(exc: BaseException) -> bool:
    """True when a provider says the account cannot pay, not that it is going too fast.

    Text matching on purpose: the three client paths raise three unrelated exception
    types — ``openai.APIStatusError``, ``google.genai.errors.ClientError`` and litellm's
    own — and the only thing they share is that the provider's words survive into
    ``str(exc)``. A 402 is unambiguous without them.
    """
    if isinstance(exc, APIStatusError) and exc.status_code == 402:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _CANNOT_PAY_MARKERS)


def skip_if_out_of_credit(events: Iterable[Any]) -> None:
    """Skip when a collected ``RunFailed`` carries a billing error.

    The events API reports provider failures as a terminal ``RunFailed`` instead of
    raising, so nothing reaches :func:`pytest_runtest_call`. Without this, a depleted
    account surfaces as whatever the test asserted next — "the run did not complete".
    """
    for event in events:
        if isinstance(event, RunFailed) and is_out_of_credit(event.exception):
            pytest.skip(f"provider account cannot pay: {event.exception}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Skip, rather than fail, when a provider account is out of credit.

    Being unable to pay is a billing state, never a code defect — no commit can turn it
    green, so letting it fail leaves CI permanently red, and a permanently red CI is one
    nobody reads. A rate limit the account *could* pay for still fails on purpose: that
    says something about the code or the request.
    """
    try:
        return (yield)
    except Exception as exc:
        if not is_out_of_credit(exc):
            raise
        pytest.skip(f"{item.name}: provider account cannot pay: {exc}")


@dataclass(frozen=True)
class ProviderConfig:
    """Maps a provider's env var, base URL, and default model."""

    env_var: str
    default_model: str
    base_url: str | None = None


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        env_var="OPENAI_API_KEY",
        default_model="gpt-4.1-nano",
    ),
    "groq": ProviderConfig(
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-120b",
    ),
    "mistral": ProviderConfig(
        env_var="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
    ),
    "moonshot": ProviderConfig(
        env_var="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        default_model="moonshot-v1-8k",
    ),
    "deepseek": ProviderConfig(
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
    ),
    "gemini": ProviderConfig(
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-3.1-flash-lite",
    ),
    "huggingface": ProviderConfig(
        env_var="HF_TOKEN",
        base_url="https://router.huggingface.co/v1",
        default_model="openai/gpt-oss-120b",
    ),
    "anthropic": ProviderConfig(
        env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-haiku-4-5-20251001",
    ),
    "xai": ProviderConfig(
        env_var="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_model="grok-4-1-fast-non-reasoning",
    ),
}


@pytest.fixture(scope="session")
def openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=api_key)


@pytest.fixture(scope="session")
def openai_model():
    return "gpt-4.1-nano"


@dataclass(frozen=True)
class LLMProvider:
    """A fully resolved provider ready for testing."""

    name: str
    client: AsyncOpenAI
    model: str


@pytest.fixture(
    scope="session",
    params=list(PROVIDER_CONFIGS.keys()),
)
def llm_provider(request: pytest.FixtureRequest) -> LLMProvider:
    """Parametrized fixture — one test run per configured provider."""
    name: str = request.param
    config = PROVIDER_CONFIGS[name]
    api_key = os.getenv(config.env_var)

    if not api_key:
        pytest.skip(f"{config.env_var} is not set")

    client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)
    return LLMProvider(name=name, client=client, model=config.default_model)


def as_dicts(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """View message params as plain dicts for assertion-friendly indexing."""
    return [dict(message) for message in messages]
