"""Pure unit tests for the ``Tool`` protocol — no LLM, no I/O."""

from openai.types.chat import ChatCompletionToolParam

from chat_cmpl_stream_handler import FunctionTool, Tool

SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {"name": "x", "parameters": {}},
}


async def _noop(_tc, _ctx) -> str:
    return "noop"


def test_function_tool_is_a_tool():
    ft = FunctionTool(tool_param=SCHEMA, invoker=_noop)
    assert isinstance(ft, Tool)


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
    """``@runtime_checkable`` only checks attribute presence, not types or
    async-ness. Documenting this so future readers don't expect deeper checks.
    """

    class SyncInvoke:
        tool_param = SCHEMA

        def invoke(self, _tc, _ctx) -> str:  # sync, wrong shape
            return "ok"

    assert isinstance(SyncInvoke(), Tool)
