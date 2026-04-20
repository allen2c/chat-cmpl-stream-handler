import pytest
from pydantic import BaseModel

from chat_cmpl_stream_handler import (
    stream_until_user_input,
)
from chat_cmpl_stream_handler.utils.get_strict_json_schema import get_strict_json_schema
from tests.conftest import LLMProvider


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


@pytest.mark.asyncio
async def test_structured_output(llm_provider: LLMProvider):
    openai_client = llm_provider.client
    model = llm_provider.model

    if "deepseek" in model:
        pytest.skip("DeepSeek does not support structured output now")

    result = await stream_until_user_input(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",  # noqa: E501
            },
            {"role": "user", "content": "how can I solve 8x + 7 = -23"},
        ],
        model=model,
        openai_client=openai_client,
        stream_kwargs={"response_format": get_strict_json_schema(MathReasoning)},
    )

    messages = result.to_input_list()
    assistant_message = messages[-1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] is not None
    assert isinstance(assistant_message["content"], str)
    assert MathReasoning.model_validate_json(assistant_message["content"]) is not None
