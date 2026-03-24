"""Anthropic Messages API wrapped as an AsyncOpenAI-compatible client.

Only ``beta.chat.completions.stream()`` is implemented — just enough for
:func:`stream_until_user_input` to work with Claude models.

Usage::

    from chat_cmpl_stream_handler._anthropic import AnthropicOpenAI

    client = AnthropicOpenAI(api_key="sk-ant-...")
    result = await stream_until_user_input(
        messages=[{"role": "user", "content": "Hi"}],
        model="claude-haiku-4-5-20251001",
        openai_client=client,
        stream_handler=ChatCompletionStreamHandler(),
    )
"""

from __future__ import annotations

import json
import logging
import time
from functools import cached_property
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from openai.lib.streaming.chat._events import (
    ContentDeltaEvent,
    ContentDoneEvent,
    FunctionToolCallArgumentsDeltaEvent,
    FunctionToolCallArgumentsDoneEvent,
)
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096
_STREAM_TIMEOUT = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------


class AnthropicOpenAI(AsyncOpenAI):
    """Drop-in ``AsyncOpenAI`` subclass that targets the Anthropic Messages API.

    Only ``client.beta.chat.completions.stream()`` is supported.
    Everything else falls through to the parent (and will likely fail
    against the Anthropic endpoint — that's fine, we don't need it).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        anthropic_version: str = _ANTHROPIC_VERSION,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ):
        # Store *before* super().__init__ so the cached_property can use them.
        self._anthropic_api_key = api_key or ""
        self._anthropic_version = anthropic_version
        self._anthropic_base_url = base_url or "https://api.anthropic.com/v1"
        self._default_max_tokens = default_max_tokens

        # AsyncOpenAI needs *something* for api_key; the value is never used
        # because we override the only code-path that makes HTTP calls.
        super().__init__(
            api_key=api_key or "unused",
            base_url=self._anthropic_base_url,
            **kwargs,
        )

    # Override the beta property so our chain is used instead of the SDK's.
    @cached_property
    def beta(self) -> _AnthropicBeta:  # type: ignore[override]
        return _AnthropicBeta(self)


# ---------------------------------------------------------------------------
# Duck-typed beta.chat.completions chain
# ---------------------------------------------------------------------------


class _AnthropicBeta:
    def __init__(self, client: AnthropicOpenAI):
        self.chat = _AnthropicChat(client)


class _AnthropicChat:
    def __init__(self, client: AnthropicOpenAI):
        self.completions = _AnthropicCompletions(client)


class _AnthropicCompletions:
    def __init__(self, client: AnthropicOpenAI):
        self._client = client

    def stream(
        self, *, messages: Any, model: Any, **kwargs: Any
    ) -> _AnthropicStreamManager:
        return _AnthropicStreamManager(
            client=self._client,
            messages=list(messages),
            model=str(model),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Async context manager  (mimics AsyncChatCompletionStreamManager)
# ---------------------------------------------------------------------------


class _AnthropicStreamManager:
    def __init__(
        self,
        client: AnthropicOpenAI,
        messages: List[dict],
        model: str,
        **kwargs: Any,
    ):
        self._client = client
        self._messages = messages
        self._model = model
        self._kwargs = dict(kwargs)
        self._stream: Optional[_AnthropicStream] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> _AnthropicStream:
        system, anthropic_messages = _translate_messages(self._messages)
        anthropic_tools = _translate_tools(self._kwargs.pop("tools", None))
        max_tokens = self._kwargs.pop("max_tokens", self._client._default_max_tokens)

        # Strip OpenAI-only keys
        for key in ("stream_options", "stream", "response_format"):
            self._kwargs.pop(key, None)

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system:
            body["system"] = system
        if anthropic_tools:
            body["tools"] = anthropic_tools

        # Forward remaining kwargs (temperature, top_p, …)
        for k, v in self._kwargs.items():
            if k not in ("messages", "model"):
                body[k] = v

        url = self._client._anthropic_base_url.rstrip("/") + "/messages"
        headers = {
            "x-api-key": self._client._anthropic_api_key,
            "anthropic-version": self._client._anthropic_version,
            "content-type": "application/json",
        }

        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(_STREAM_TIMEOUT))
        response = await self._http_client.send(
            self._http_client.build_request("POST", url, json=body, headers=headers),
            stream=True,
        )

        if response.status_code != 200:
            error_body = await response.aread()
            await response.aclose()
            await self._http_client.aclose()
            raise RuntimeError(
                f"Anthropic API error {response.status_code}: {error_body.decode()}"
            )

        self._stream = _AnthropicStream(response, self._model)
        return self._stream

    async def __aexit__(self, *args: Any) -> None:
        if self._stream and self._stream._response:
            await self._stream._response.aclose()
        if self._http_client:
            await self._http_client.aclose()


# ---------------------------------------------------------------------------
# Async stream  (mimics AsyncChatCompletionStream)
# ---------------------------------------------------------------------------


class _AnthropicStream:
    """Yields OpenAI-shaped ``ChatCompletionStreamEvent`` objects from an
    Anthropic SSE stream, and exposes ``get_final_completion()``.
    """

    def __init__(self, response: httpx.Response, model: str):
        self._response = response
        self._model = model
        self._consumed = False

        # Accumulation state
        self._message_id: str = ""
        self._content: str = ""
        self._tool_calls: List[Dict[str, Any]] = []
        self._block_types: Dict[int, str] = {}  # block_index → "text"|"tool_use"
        self._block_to_tc: Dict[int, int] = {}  # block_index → tool_calls list index
        self._tc_count: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._stop_reason: Optional[str] = None

    # -- async iteration -----------------------------------------------------

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        event_name: Optional[str] = None
        async for raw_line in self._response.aiter_lines():
            line = raw_line.strip()
            if not line:
                event_name = None
                continue
            if line.startswith("event: "):
                event_name = line[7:]
                continue
            if not line.startswith("data: "):
                continue

            data = json.loads(line[6:])
            async for event in self._translate(event_name, data):
                yield event

        self._consumed = True

    async def _translate(  # noqa: C901 — intentionally flat for readability
        self, event_name: Optional[str], data: dict
    ) -> AsyncIterator[Any]:
        msg_type = data.get("type", "")

        # -- message_start: extract id & input tokens -----------------------
        if msg_type == "message_start":
            msg = data.get("message", {})
            self._message_id = msg.get("id", "")
            self._input_tokens = msg.get("usage", {}).get("input_tokens", 0)

        # -- content_block_start: register block type -----------------------
        elif msg_type == "content_block_start":
            index = data["index"]
            block = data["content_block"]
            block_type = block["type"]
            self._block_types[index] = block_type

            if block_type == "tool_use":
                tc_idx = self._tc_count
                self._tc_count += 1
                self._block_to_tc[index] = tc_idx
                self._tool_calls.append(
                    {"id": block["id"], "name": block["name"], "arguments": ""}
                )

        # -- content_block_delta: emit streaming events ---------------------
        elif msg_type == "content_block_delta":
            index = data["index"]
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                self._content += text
                yield ContentDeltaEvent(
                    type="content.delta",
                    delta=text,
                    snapshot=self._content,
                    parsed=None,
                )

            elif delta_type == "input_json_delta":
                tc_idx = self._block_to_tc[index]
                partial = delta.get("partial_json", "")
                self._tool_calls[tc_idx]["arguments"] += partial
                yield FunctionToolCallArgumentsDeltaEvent(
                    type="tool_calls.function.arguments.delta",
                    name=self._tool_calls[tc_idx]["name"],
                    index=tc_idx,
                    arguments=self._tool_calls[tc_idx]["arguments"],
                    parsed_arguments=None,
                    arguments_delta=partial,
                )

        # -- content_block_stop: emit "done" events -------------------------
        elif msg_type == "content_block_stop":
            index = data["index"]
            block_type = self._block_types.get(index)

            if block_type == "text":
                yield ContentDoneEvent(
                    type="content.done",
                    content=self._content,
                    parsed=None,
                )

            elif block_type == "tool_use":
                tc_idx = self._block_to_tc[index]
                tc = self._tool_calls[tc_idx]
                parsed = None
                try:
                    parsed = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    pass
                yield FunctionToolCallArgumentsDoneEvent(
                    type="tool_calls.function.arguments.done",
                    name=tc["name"],
                    index=tc_idx,
                    arguments=tc["arguments"],
                    parsed_arguments=parsed,
                )

        # -- message_delta: capture stop_reason & output tokens -------------
        elif msg_type == "message_delta":
            self._stop_reason = data.get("delta", {}).get("stop_reason")
            self._output_tokens = data.get("usage", {}).get("output_tokens", 0)

        # -- ping / message_stop: ignored -----------------------------------

    # -- final completion ----------------------------------------------------

    async def get_final_completion(self) -> ChatCompletion:
        if not self._consumed:
            async for _ in self._iterate():
                pass

        if self._stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif self._stop_reason == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        tool_calls_out = None
        if self._tool_calls:
            tool_calls_out = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=Function(name=tc["name"], arguments=tc["arguments"]),
                )
                for tc in self._tool_calls
            ]

        return ChatCompletion(
            id=self._message_id or f"msg_{int(time.time())}",
            choices=[
                Choice(
                    finish_reason=finish_reason,  # type: ignore[arg-type]
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=self._content or None,
                        tool_calls=tool_calls_out,
                    ),
                )
            ],
            created=int(time.time()),
            model=self._model,
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=self._input_tokens,
                completion_tokens=self._output_tokens,
                total_tokens=self._input_tokens + self._output_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# Message / tool translation  (OpenAI → Anthropic)
# ---------------------------------------------------------------------------


def _translate_messages(messages: List[dict]) -> tuple[Optional[str], List[dict]]:
    """Convert OpenAI-format messages to ``(system, anthropic_messages)``."""

    system_parts: List[str] = []
    anthropic_msgs: List[dict] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                system_parts.append(
                    " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                )
            else:
                system_parts.append(str(content))

        elif role == "user":
            anthropic_msgs.append({"role": "user", "content": msg.get("content", "")})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            text = msg.get("content")

            if not tool_calls:
                anthropic_msgs.append({"role": "assistant", "content": text or ""})
            else:
                blocks: List[dict] = []
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        input_obj = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        input_obj = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": input_obj,
                        }
                    )
                anthropic_msgs.append({"role": "assistant", "content": blocks})

        elif role == "tool":
            # Consecutive tool messages → single user message with tool_result blocks
            tool_blocks: List[dict] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                t = messages[i]
                tool_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": t.get("tool_call_id", ""),
                        "content": t.get("content", ""),
                    }
                )
                i += 1
            anthropic_msgs.append({"role": "user", "content": tool_blocks})
            continue  # skip the i += 1 below

        i += 1

    system = "\n\n".join(system_parts) if system_parts else None
    return system, anthropic_msgs


def _translate_tools(tools: Optional[list]) -> Optional[list]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return None

    out: List[dict] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object"}),
            }
        )
    return out or None
