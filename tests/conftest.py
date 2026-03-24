import agents
import openai
import pytest


@pytest.fixture(scope="module")
def model_name():
    return "gpt-4.1-nano"


@pytest.fixture(scope="module")
def openai_client():
    return openai.AsyncOpenAI()


@pytest.fixture(scope="module")
def chat_model(model_name: str, openai_client: openai.AsyncOpenAI):
    return agents.OpenAIResponsesModel(model=model_name, openai_client=openai_client)


@pytest.fixture(scope="module")
def model_settings():
    return agents.ModelSettings(temperature=0.0)


@pytest.fixture(scope="module")
def agent():
    return agents.Agent(name="Test Agent")


@pytest.fixture(scope="module")
def agents_run_config():
    return agents.RunConfig(tracing_disabled=True)
