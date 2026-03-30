import pytest

from chat_cmpl_stream_handler.utils.mcp import list_mcp_tools

COINGECKO_URL: str = "https://mcp.api.coingecko.com"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_label, filter_tool",
    [
        (None, None),
        ("cg", None),
        (None, lambda t: "simple" in t["function"]["name"]),
    ],
    ids=["no-options", "with-label", "with-filter"],
)
async def test_list_mcp_tools(server_label, filter_tool) -> None:
    kwargs = {}
    if server_label is not None:
        kwargs["server_label"] = server_label
    if filter_tool is not None:
        kwargs["filter_tool"] = filter_tool

    tools = await list_mcp_tools(COINGECKO_URL, **kwargs)
    assert len(tools) > 0

    for tp in tools:
        func = tp["function"]

        assert tp["type"] == "function"
        assert func["strict"] is True

        params = func.get("parameters", {})
        if params.get("type") == "object" and params.get("properties"):
            assert params["additionalProperties"] is False
            assert "title" not in params
            for prop in params["properties"].values():
                assert "title" not in prop

        if server_label:
            assert func["name"].startswith(f"{server_label}__")

        if filter_tool:
            assert filter_tool(tp)
