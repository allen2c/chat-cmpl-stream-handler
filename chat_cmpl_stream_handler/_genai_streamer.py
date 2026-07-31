"""Native google-genai support: OpenAI chat completions in, genai calls out.

Unlike the other two adapters this one is a real translation, in both directions:

    messages          -> contents + system_instruction
    tools             -> FunctionDeclaration (raw JSON Schema, no dialect rewrite)
    response_format   -> response_json_schema + response_mime_type
    GenerateContentResponse -> ChatCompletionChunk

Gemini 3 requires a ``thought_signature`` to be echoed back on the part it came from, so
the round trip has to carry provider state the OpenAI shape has no room for. It rides in
``provider_specific_fields`` — the same key litellm uses, with the same two levels:

    message   provider_specific_fields.thought_signatures = [b64, ...]   (non-call parts)
    tool_call provider_specific_fields.thought_signature  =  b64         (that call's part)

Signatures are ``bytes`` on the wire and base64 ``str`` inside a chunk, because a chunk has
to survive a JSON round trip. The extension key is emitted exactly once, on a final chunk,
so the OpenAI SDK's delta accumulator never sees a repeated string key to concatenate.
"""

import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

#: Marks a tool-call id we invented because Gemini did not send one. Never goes back out.
SYNTHETIC_ID_PREFIX = "call_genai_"

PROVIDER_FIELDS = "provider_specific_fields"

_FINISH_REASONS: Dict[types.FinishReason, str] = {
    types.FinishReason.STOP: "stop",
    types.FinishReason.MAX_TOKENS: "length",
    types.FinishReason.SAFETY: "content_filter",
    types.FinishReason.RECITATION: "content_filter",
    types.FinishReason.BLOCKLIST: "content_filter",
    types.FinishReason.PROHIBITED_CONTENT: "content_filter",
    types.FinishReason.SPII: "content_filter",
    types.FinishReason.IMAGE_SAFETY: "content_filter",
    types.FinishReason.IMAGE_PROHIBITED_CONTENT: "content_filter",
}

#: OpenAI request options that map onto a differently named genai config field.
_RENAMED_OPTIONS = {
    "max_tokens": "max_output_tokens",
    "max_completion_tokens": "max_output_tokens",
    "stop": "stop_sequences",
    "n": "candidate_count",
    "top_logprobs": "logprobs",
}

#: OpenAI request options with no genai equivalent, and no harm in ignoring.
_IGNORED_OPTIONS = frozenset({"stream", "stream_options", "parallel_tool_calls", "user"})


class GenAIStreamer:
    """Streams from a native ``genai.Client``."""

    def __init__(self, client: genai.Client) -> None:
        self._client = client

    async def stream(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        contents, config = _to_genai_request(messages, kwargs)
        collector = _SignatureCollector()

        stream = await self._client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )

        usage: Optional[types.GenerateContentResponseUsageMetadata] = None
        finish_reason: Optional[str] = None

        async for response in stream:
            if response.usage_metadata is not None:
                usage = response.usage_metadata

            chunk = _chunk_from_response(response, model=model, collector=collector)
            if chunk is not None:
                yield chunk

            candidate = _first_candidate(response)
            if candidate is not None and candidate.finish_reason is not None:
                finish_reason = _FINISH_REASONS.get(candidate.finish_reason, "stop")

        if collector.saw_tool_call:
            finish_reason = "tool_calls"

        yield _final_chunk(model=model, collector=collector, usage=usage, finish_reason=finish_reason)


# --- request: OpenAI -> genai -----------------------------------------------------


def _to_genai_request(
    messages: Iterable[ChatCompletionMessageParam],
    options: Dict[str, Any],
) -> Tuple[List[types.Content], types.GenerateContentConfig]:
    contents, system_instruction = _to_contents(messages)
    return contents, _to_config(options, system_instruction)


def _to_contents(
    messages: Iterable[ChatCompletionMessageParam],
) -> Tuple[List[types.Content], Optional[str]]:
    contents: List[types.Content] = []
    system_chunks: List[str] = []
    tool_call_names: Dict[str, str] = {}

    for message in messages:
        message_dict: Dict[str, Any] = dict(message)
        role = message_dict.get("role")

        if role in ("system", "developer"):
            system_chunks.append(_text_of(message_dict.get("content")))
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=_text_of(message_dict.get("content")))]))
        elif role == "assistant":
            for tool_call in message_dict.get("tool_calls") or ():
                tool_call_names[tool_call["id"]] = tool_call["function"]["name"]
            contents.append(_assistant_content(message_dict))
        elif role == "tool":
            contents.append(_tool_content(message_dict, tool_call_names))
        else:
            raise ValueError(f"Cannot translate a {role!r} message to genai contents")

    return contents, "\n\n".join(chunk for chunk in system_chunks if chunk) or None


def _assistant_content(message: Dict[str, Any]) -> types.Content:
    signatures = list((message.get(PROVIDER_FIELDS) or {}).get("thought_signatures") or [])
    parts: List[types.Part] = []

    content = message.get("content")
    if content:
        parts.append(
            types.Part(
                text=_text_of(content),
                # Gemini takes one signature per part; the text part gets the first, the
                # same convention litellm follows on its way back in.
                thought_signature=_decode(signatures[0]) if signatures else None,
            )
        )

    for tool_call in message.get("tool_calls") or ():
        signature = (tool_call.get(PROVIDER_FIELDS) or {}).get("thought_signature")
        call_id = tool_call["id"]
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id=None if call_id.startswith(SYNTHETIC_ID_PREFIX) else call_id,
                    name=tool_call["function"]["name"],
                    args=json.loads(tool_call["function"].get("arguments") or "{}"),
                ),
                thought_signature=_decode(signature) if signature else None,
            )
        )

    return types.Content(role="model", parts=parts)


def _tool_content(message: Dict[str, Any], tool_call_names: Dict[str, str]) -> types.Content:
    call_id = message.get("tool_call_id") or ""
    name = tool_call_names.get(call_id)
    if name is None:
        raise ValueError(f"Tool message references unknown tool_call_id {call_id!r}")

    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=None if call_id.startswith(SYNTHETIC_ID_PREFIX) else call_id,
                    name=name,
                    response={"result": _text_of(message.get("content"))},
                )
            )
        ],
    )


def _to_config(options: Dict[str, Any], system_instruction: Optional[str]) -> types.GenerateContentConfig:
    """Map OpenAI request options onto a genai config.

    Anything not recognised is forwarded verbatim, so genai-native options work and typos
    are rejected loudly rather than silently dropped.
    """
    rest = {key: value for key, value in options.items() if key not in _IGNORED_OPTIONS}
    config: Dict[str, Any] = {}

    if system_instruction:
        config["system_instruction"] = system_instruction

    tools = rest.pop("tools", None)
    if tools:
        config["tools"] = [types.Tool(function_declarations=[_function_declaration(tool) for tool in tools])]

    tool_choice = rest.pop("tool_choice", None)
    if tool_choice is not None:
        config["tool_config"] = _tool_config(tool_choice)

    response_format = rest.pop("response_format", None)
    if response_format is not None:
        config.update(_response_format(response_format))

    for openai_name, genai_name in _RENAMED_OPTIONS.items():
        if openai_name in rest:
            config[genai_name] = rest.pop(openai_name)

    config.update(rest)
    return types.GenerateContentConfig(**config)


def _function_declaration(tool: Dict[str, Any]) -> types.FunctionDeclaration:
    function = tool["function"]
    return types.FunctionDeclaration(
        name=function["name"],
        description=function.get("description"),
        parameters_json_schema=function.get("parameters"),
    )


def _tool_config(tool_choice: Any) -> types.ToolConfig:
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=[name] if name else None,
            )
        )

    modes = {
        "auto": types.FunctionCallingConfigMode.AUTO,
        "none": types.FunctionCallingConfigMode.NONE,
        "required": types.FunctionCallingConfigMode.ANY,
    }
    mode = modes.get(str(tool_choice))
    if mode is None:
        raise ValueError(f"Unsupported tool_choice: {tool_choice!r}")
    return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))


def _response_format(response_format: Any) -> Dict[str, Any]:
    kind = dict(response_format).get("type") if isinstance(response_format, dict) else None

    if kind == "text" or response_format is None:
        return {}
    if kind == "json_object":
        return {"response_mime_type": "application/json"}
    if kind == "json_schema":
        schema = dict(response_format)["json_schema"].get("schema")
        return {"response_mime_type": "application/json", "response_json_schema": schema}

    raise ValueError(f"Unsupported response_format: {response_format!r}")


# --- response: genai -> OpenAI ----------------------------------------------------


class _SignatureCollector:
    """Collects provider state across a stream so it can be emitted once at the end."""

    def __init__(self) -> None:
        self.thought_signatures: List[str] = []
        self.tool_call_signatures: Dict[int, str] = {}
        self.tool_call_count = 0

    @property
    def saw_tool_call(self) -> bool:
        return self.tool_call_count > 0

    def next_tool_call_index(self) -> int:
        index = self.tool_call_count
        self.tool_call_count += 1
        return index


def _chunk_from_response(
    response: types.GenerateContentResponse,
    *,
    model: str,
    collector: _SignatureCollector,
) -> Optional[ChatCompletionChunk]:
    candidate = _first_candidate(response)
    if candidate is None or candidate.content is None:
        return None

    text_pieces: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for part in candidate.content.parts or ():
        signature = _encode(part.thought_signature) if part.thought_signature else None

        if part.function_call is not None:
            index = collector.next_tool_call_index()
            if signature:
                collector.tool_call_signatures[index] = signature
            tool_calls.append(
                {
                    "index": index,
                    "id": part.function_call.id or f"{SYNTHETIC_ID_PREFIX}{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name or "",
                        "arguments": json.dumps(part.function_call.args or {}),
                    },
                }
            )
            continue

        if signature:
            collector.thought_signatures.append(signature)

        # A thought part is Gemini's private reasoning. It has no OpenAI home, and echoing
        # it as assistant content would put it in front of the user.
        if part.text and not part.thought:
            text_pieces.append(part.text)

    if not text_pieces and not tool_calls:
        return None

    delta: Dict[str, Any] = {"role": "assistant"}
    if text_pieces:
        delta["content"] = "".join(text_pieces)
    if tool_calls:
        delta["tool_calls"] = tool_calls

    return _build_chunk(model=model, delta=delta)


def _final_chunk(
    *,
    model: str,
    collector: _SignatureCollector,
    usage: Optional[types.GenerateContentResponseUsageMetadata],
    finish_reason: Optional[str],
) -> ChatCompletionChunk:
    """One closing chunk carrying the finish reason, usage, and all provider state.

    Every extension key appears here and nowhere else. Repeated string keys get
    concatenated by the OpenAI accumulator, and the tool-call entries deliberately carry
    nothing but their index for the same reason.
    """
    delta: Dict[str, Any] = {}

    if collector.thought_signatures:
        delta[PROVIDER_FIELDS] = {"thought_signatures": collector.thought_signatures}

    if collector.tool_call_signatures:
        delta["tool_calls"] = [
            # `function` has to be present — the OpenAI SDK asserts on it — but it stays
            # empty so no repeated string key is there to be concatenated.
            {"index": index, "function": {}, PROVIDER_FIELDS: {"thought_signature": signature}}
            for index, signature in sorted(collector.tool_call_signatures.items())
        ]

    return _build_chunk(
        model=model,
        delta=delta,
        finish_reason=finish_reason or "stop",
        usage=_to_usage(usage),
    )


def _build_chunk(
    *,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> ChatCompletionChunk:
    raw: Dict[str, Any] = {
        "id": "genai",
        "created": 0,
        "model": model,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        raw["usage"] = usage
    return ChatCompletionChunk.model_validate(raw)


def _to_usage(usage: Optional[types.GenerateContentResponseUsageMetadata]) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None

    prompt_tokens = usage.prompt_token_count or 0
    completion_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.total_token_count or (prompt_tokens + completion_tokens),
    }


# --- small helpers ----------------------------------------------------------------


def _first_candidate(response: types.GenerateContentResponse) -> Optional[types.Candidate]:
    return response.candidates[0] if response.candidates else None


def _text_of(content: Any) -> str:
    """Flatten OpenAI message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    pieces: List[str] = []
    for part in content:
        part_dict = dict(part)
        if part_dict.get("type") != "text":
            raise ValueError(f"Cannot translate a {part_dict.get('type')!r} content part to genai")
        pieces.append(part_dict.get("text") or "")
    return "".join(pieces)


def _encode(signature: bytes) -> str:
    return base64.b64encode(signature).decode()


def _decode(signature: str) -> bytes:
    return base64.b64decode(signature)
