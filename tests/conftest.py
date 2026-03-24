import os
from dataclasses import dataclass

import pytest
from openai import AsyncOpenAI


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
        default_model="gemini-2.5-flash",
    ),
    "huggingface": ProviderConfig(
        env_var="HF_TOKEN",
        base_url="https://router.huggingface.co/v1",
        default_model="openai/gpt-oss-120b",
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

    client = AsyncOpenAI(
        api_key=api_key,
        **({"base_url": config.base_url} if config.base_url else {}),
    )
    return LLMProvider(name=name, client=client, model=config.default_model)
