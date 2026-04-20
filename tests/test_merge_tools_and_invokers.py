"""Pure unit tests for ``merge_tools_and_invokers`` — no LLM, no I/O."""

from typing import Any

import pytest
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from chat_cmpl_stream_handler import FunctionTool, Tool, merge_tools_and_invokers


def _schema(name: str) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {}},
    }


async def _noop(_tc: ChatCompletionMessageToolCall, _ctx: Any) -> str:
    return "noop"


async def _other(_tc: ChatCompletionMessageToolCall, _ctx: Any) -> str:
    return "other"


def test_dict_path_unchanged():
    schemas, invokers = merge_tools_and_invokers(
        tool_invokers={"foo": _noop},
        stream_tools=[_schema("foo")],
    )
    assert [s["function"]["name"] for s in schemas] == ["foo"]
    assert invokers == {"foo": _noop}


def test_function_tool_derives_invoker():
    ft = FunctionTool(tool_param=_schema("bar"), invoker=_noop)
    schemas, invokers = merge_tools_and_invokers(tools=[ft])
    assert [s["function"]["name"] for s in schemas] == ["bar"]
    assert invokers["bar"] == ft.invoke


def test_duck_typed_tool_accepted():
    class Custom:
        tool_param = _schema("baz")

        async def invoke(self, _tc, _ctx):
            return "baz"

    custom = Custom()
    assert isinstance(custom, Tool)
    _, invokers = merge_tools_and_invokers(tools=[custom])
    assert invokers["baz"] == custom.invoke


def test_missing_invoker_raises():
    with pytest.raises(ValueError, match="No invoker for tool"):
        merge_tools_and_invokers(stream_tools=[_schema("orphan")])


def test_explicit_invoker_overrides_tool(caplog: pytest.LogCaptureFixture):
    ft = FunctionTool(tool_param=_schema("bar"), invoker=_noop)
    with caplog.at_level("WARNING", logger="chat_cmpl_stream_handler"):
        _, invokers = merge_tools_and_invokers(
            tools=[ft],
            tool_invokers={"bar": _other},
        )
    assert invokers["bar"] is _other
    assert any("overrides" in r.message for r in caplog.records)


def test_schema_dedup_last_write_wins():
    a = _schema("dup")
    b = _schema("dup")
    b["function"]["description"] = "second"
    schemas, _ = merge_tools_and_invokers(
        tool_invokers={"dup": _noop},
        stream_tools=[a, b],
    )
    assert len(schemas) == 1
    assert schemas[0]["function"].get("description") == "second"
