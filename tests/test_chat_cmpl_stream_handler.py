import pytest
from openai import AsyncOpenAI

from chat_cmpl_stream_handler import ChatCompletionStreamHandler


@pytest.mark.asyncio
async def test_chat_completion_stream_handler(
    openai_client: AsyncOpenAI, openai_model: str
):
    handler = ChatCompletionStreamHandler()

    stream_manager = openai_client.chat.completions.stream(
        model=openai_model,
        messages=[
            {"role": "system", "content": "You are a truncated response assistant."},
            {"role": "user", "content": "Repeat `hello world` directly."},
        ],
    )

    async with stream_manager as stream:
        async for chunk in stream:
            await handler.handle(chunk)
