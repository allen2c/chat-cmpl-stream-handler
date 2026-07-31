"""Adapters that reduce a provider client to a stream of OpenAI chunks.

The loop needs exactly one thing from a client: an async iterator of
``ChatCompletionChunk``. :class:`ChunkStreamer` is that contract. Bring your own
implementation and the loop will take it — ``openai_client=`` accepts either a raw
provider client or anything matching the protocol.
"""

from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, List, Protocol, Union, cast, runtime_checkable

from google import genai
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router import Router as LiteLLMRouter
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from chat_cmpl_stream_handler._genai_streamer import GenAIStreamer

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues


@runtime_checkable
class ChunkStreamer(Protocol):
    """One streaming request, in OpenAI chunk terms.

    ``kwargs`` carries OpenAI-named request options (``tools``, ``response_format``,
    ``temperature``, ...) straight through to the provider. Implementations translate
    shape only; they do not filter or validate.
    """

    def stream(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]: ...


class OpenAIStreamer:
    """Streams from OpenAI and any OpenAI-compatible endpoint."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def stream(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream = await self._client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            **kwargs,
        )
        try:
            async for chunk in stream:
                _normalize_tool_call_indices(chunk)
                yield chunk
        finally:
            await stream.close()


class LiteLLMStreamer:
    """Streams from a litellm ``Router`` deployment.

    litellm already speaks the OpenAI chunk shape, so this is a re-validation, not a
    translation. That includes provider state: ``provider_specific_fields`` and the
    ``__thought__`` suffix litellm splices into tool-call ids both ride through verbatim.
    """

    def __init__(self, router: LiteLLMRouter) -> None:
        self._router = router

    async def stream(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream = cast(
            CustomStreamWrapper,
            await self._router.acompletion(
                messages=cast("List[AllMessageValues]", messages),
                model=model,
                stream=True,
                **kwargs,
            ),
        )
        try:
            async for chunk in stream:
                yield ChatCompletionChunk.model_validate(chunk.model_dump())
        finally:
            await stream.aclose()


def as_streamer(client: Union[AsyncOpenAI, LiteLLMRouter, "genai.Client", ChunkStreamer]) -> ChunkStreamer:
    """Wrap a provider client in its adapter. Pass a ``ChunkStreamer`` through untouched."""
    if isinstance(client, AsyncOpenAI):
        return OpenAIStreamer(client)
    if isinstance(client, LiteLLMRouter):
        return LiteLLMStreamer(client)
    if isinstance(client, genai.Client):
        return GenAIStreamer(client)
    if isinstance(client, ChunkStreamer):
        return client
    raise TypeError(f"Cannot stream from {type(client).__name__}: expected a known client or a ChunkStreamer")


def _normalize_tool_call_indices(chunk: ChatCompletionChunk) -> None:
    """Mutate *chunk* in place so every ``tool_call.index`` is an int.

    Gemini's OpenAI-compatible endpoint sends ``index = None``, which the OpenAI SDK's
    delta accumulator cannot merge. Positional order is the only sane fallback.
    """
    for choice in chunk.choices:
        for position, tool_call in enumerate(choice.delta.tool_calls or ()):
            if tool_call.index is None:
                tool_call.index = position
