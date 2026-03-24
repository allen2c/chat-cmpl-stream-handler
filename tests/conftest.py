import os

import pytest
from openai import AsyncOpenAI


@pytest.fixture(scope="session")
def openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=api_key)


@pytest.fixture(scope="session")
def openai_model():
    return "gpt-4.1-mini"
