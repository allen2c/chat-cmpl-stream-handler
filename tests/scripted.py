"""An offline ChunkStreamer for loop-logic tests — no provider, no network.

Real provider APIs test provider behaviour; this tests the loop. One scripted turn is
replayed per loop iteration, and every request the loop makes is recorded for assertions.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam


class ScriptedStreamer:
    """Replays one canned turn per loop iteration."""

    def __init__(self, *turns: Sequence[ChatCompletionChunk]) -> None:
        self._turns = turns
        self.requests: List[Dict[str, Any]] = []

    async def stream(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        turn_index = len(self.requests)
        self.requests.append({"messages": list(messages), "model": model, **kwargs})

        if turn_index >= len(self._turns):
            raise AssertionError(f"ScriptedStreamer was scripted for {len(self._turns)} turn(s), asked for more")

        for chunk in self._turns[turn_index]:
            yield chunk


def text_turn(text: str) -> List[ChatCompletionChunk]:
    """A turn where the model answers and stops."""
    return [
        chunk({"role": "assistant", "content": text}),
        chunk({}, finish_reason="stop"),
    ]


def tool_call_turn(name: str, arguments: str = "{}", *, call_id: str = "call_1") -> List[ChatCompletionChunk]:
    """A turn where the model asks for one tool call."""
    return [
        chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        ),
        chunk({}, finish_reason="tool_calls"),
    ]


def chunk(delta: Dict[str, Any], *, finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    """Build one chunk from a raw delta, the way a provider would send it."""
    return ChatCompletionChunk.model_validate(
        {
            "id": "scripted",
            "created": 0,
            "model": "scripted",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )
