import pytest

from chat_cmpl_stream_handler import ChatCompletionStreamHandler
from tests.conftest import LLMProvider


@pytest.mark.asyncio
async def test_chat_completion_stream_handler(llm_provider: LLMProvider):
    handler = ChatCompletionStreamHandler()

    openai_client = llm_provider.client
    model = llm_provider.model

    stream_manager = openai_client.chat.completions.stream(
        model=model,
        messages=[
            {"role": "system", "content": "You are a truncated response assistant."},
            {"role": "user", "content": "Repeat `hello world` directly."},
        ],
    )

    async with stream_manager as stream:
        async for chunk in stream:
            await handler.handle(chunk)
