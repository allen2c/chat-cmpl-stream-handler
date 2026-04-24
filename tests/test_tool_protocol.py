"""Pure unit tests for the Tool protocol."""

from openai.types.chat import ChatCompletionToolParam

from chat_cmpl_stream_handler import FunctionTool, Tool, ToolResult

SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {"name": "x", "parameters": {}},
}


async def _noop(_tc, _ctx) -> str:
    return "noop"


async def _structured_noop(_tc, _ctx) -> ToolResult:
    return ToolResult(content="noop", metadata={"source": "test"})


def test_function_tool_is_a_tool():
    ft = FunctionTool(tool_param=SCHEMA, invoker=_noop)
    assert isinstance(ft, Tool)


def test_function_tool_accepts_tool_result_invoker():
    ft = FunctionTool(tool_param=SCHEMA, invoker=_structured_noop)
    assert isinstance(ft, Tool)


def test_tool_protocol_accepts_str_and_tool_result_return_types():
    class StringTool:
        tool_param = SCHEMA

        async def invoke(self, _tc, _ctx) -> str:
            return "ok"

    class StructuredTool:
        tool_param = SCHEMA

        async def invoke(self, _tc, _ctx) -> ToolResult:
            return ToolResult(content="ok")

    assert isinstance(StringTool(), Tool)
    assert isinstance(StructuredTool(), Tool)


def test_duck_typed_object_is_a_tool():
    class Custom:
        tool_param = SCHEMA

        async def invoke(self, _tc, _ctx) -> str:
            return "ok"

    assert isinstance(Custom(), Tool)


def test_missing_tool_param_is_not_a_tool():
    class NoSchema:
        async def invoke(self, _tc, _ctx) -> str:
            return "ok"

    assert not isinstance(NoSchema(), Tool)


def test_missing_invoke_is_not_a_tool():
    class NoInvoke:
        tool_param = SCHEMA

    assert not isinstance(NoInvoke(), Tool)


def test_runtime_checkable_does_not_validate_signatures():
    """Runtime protocol checks only verify attribute presence."""

    class SyncInvoke:
        tool_param = SCHEMA

        def invoke(self, _tc, _ctx) -> str:  # sync, wrong shape
            return "ok"

    assert isinstance(SyncInvoke(), Tool)
