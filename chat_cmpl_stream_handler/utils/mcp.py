import logging
from typing import Callable, Dict, List, Optional, Sequence, Text

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import Tool as McpTool
from openai.types.chat import ChatCompletionToolParam as ToolParam
from openai.types.shared_params import FunctionDefinition

logger: logging.Logger = logging.getLogger(__name__)

_COMMON_SSE_PATHS: Sequence[str] = (
    "",
    "/sse",
    "/mcp",
    "/mcp/sse",
    "/api/sse",
)


async def list_mcp_tools(
    server_url: str,
    *,
    server_label: Optional[Text] = None,
    filter_tool: Callable[[ToolParam], bool] = lambda _: True,
) -> List[ToolParam]:
    """Connect to an MCP SSE server, discover tools, and return them
    as OpenAI-compatible ChatCompletionToolParam objects.
    """
    mcp_tools: List[McpTool] = await _discover_and_connect(server_url)
    label_prefix: str = f"{server_label}__" if server_label else ""

    tool_params: List[ToolParam] = [
        _mcp_tool_to_tool_param(t, label_prefix=label_prefix) for t in mcp_tools
    ]

    return [tp for tp in tool_params if filter_tool(tp)]


def _sanitize_schema_for_strict(schema: Dict[str, object]) -> Dict[str, object]:
    """Recursively transform a JSON Schema to satisfy OpenAI strict mode:
    strip ``title``, inject ``additionalProperties: false``,
    and ensure all properties appear in ``required``.
    """
    result: Dict[str, object] = {}

    for key, value in schema.items():
        if key == "title":
            continue
        if isinstance(value, dict):
            result[key] = _sanitize_schema_for_strict(value)
        elif isinstance(value, list):
            result[key] = [
                _sanitize_schema_for_strict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    if result.get("type") == "object" and "properties" in result:
        result["additionalProperties"] = False
        prop_names: List[str] = list(result["properties"].keys())
        if prop_names:
            result["required"] = prop_names

    return result


def _mcp_tool_to_tool_param(
    tool: McpTool,
    *,
    label_prefix: str = "",
) -> ToolParam:
    """Convert an MCP Tool to an OpenAI ChatCompletionToolParam
    with strict-mode-compatible parameters.
    """
    name: str = f"{label_prefix}{tool.name}" if label_prefix else tool.name
    raw_schema: Dict[str, object] = tool.inputSchema if tool.inputSchema else {}
    parameters: Dict[str, object] = _sanitize_schema_for_strict(raw_schema)

    function_def: FunctionDefinition = {
        "name": name,
        "parameters": parameters,
        "strict": True,
    }
    if tool.description:
        function_def["description"] = tool.description

    return ToolParam(type="function", function=function_def)


async def _discover_and_connect(
    base_url: str,
) -> List[McpTool]:
    """Probe common SSE paths and return the tool list from the first
    successful MCP handshake.
    """
    base_url = base_url.rstrip("/")

    for path in _COMMON_SSE_PATHS:
        endpoint: str = f"{base_url}{path}"
        logger.debug("Probing MCP endpoint: %s", endpoint)

        try:
            async with sse_client(endpoint) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    logger.info(
                        "Connected to MCP at %s — found %d tools",
                        endpoint,
                        len(result.tools),
                    )
                    return list(result.tools)

        except* httpx.HTTPStatusError as eg:
            for exc in eg.exceptions:
                logger.debug("HTTP %s at %s", exc.response.status_code, endpoint)

        except* Exception as eg:
            for exc in eg.exceptions:
                logger.debug(
                    "%s at %s: %s",
                    type(exc).__name__,
                    endpoint,
                    str(exc).splitlines()[0] if str(exc) else "no details",
                )

    raise ConnectionError(
        f"Failed to establish MCP connection to {base_url} "
        f"(tried paths: {', '.join(p or '/' for p in _COMMON_SSE_PATHS)})"
    )
