import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Text,
    Tuple,
)

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
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

_TransportType = Text  # "sse" | "streamable_http"
ToolFilterFn = Callable[[ToolParam], bool]
ToolInvokerFn = Callable[[str, Any], Awaitable[str]]

_endpoint_cache: Dict[str, Tuple[str, _TransportType]] = {}


def clear_endpoint_cache() -> None:
    """Clear the endpoint discovery cache."""
    _endpoint_cache.clear()


async def call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: str,
    *,
    server_label: Optional[Text] = None,
    meta: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    session: Optional[ClientSession] = None,
) -> str:
    """Connect to an MCP server and invoke a single tool by name.

    If *session* is provided, it must already be initialized and is
    used directly without endpoint discovery or reconnection.

    If *server_label* is provided and *tool_name* starts with
    ``{server_label}__``, the prefix is stripped before calling the
    remote tool.

    When the remote tool signals an error (``isError=True``), the
    error content is still returned as a string so the LLM can
    observe it.
    """
    actual_name = tool_name
    if server_label:
        prefix = f"{server_label}__"
        if actual_name.startswith(prefix):
            actual_name = actual_name[len(prefix) :]

    parsed_args: Dict[str, object] = json.loads(arguments) if arguments else {}

    base_url = server_url.rstrip("/")
    async with _get_mcp_session(
        base_url,
        extra_headers=extra_headers,
        session=session,
    ) as active_session:
        result = await active_session.call_tool(actual_name, parsed_args, meta=meta)

        if result.isError:
            logger.warning("MCP tool %s returned an error", actual_name)

        parts: List[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(json.dumps(block.model_dump(), default=str))
        return "\n".join(parts)


async def list_mcp_tools(
    server_url: str,
    *,
    server_label: Optional[Text] = None,
    filter_tool: Callable[[ToolParam], bool] = lambda _: True,
    extra_headers: Optional[Dict[str, str]] = None,
    session: Optional[ClientSession] = None,
) -> List[ToolParam]:
    """Connect to an MCP server, discover tools, and return them
    as OpenAI-compatible ChatCompletionToolParam objects.

    If *session* is provided, it must already be initialized and is
    used directly without endpoint discovery or reconnection.
    """
    base_url = server_url.rstrip("/")
    async with _get_mcp_session(
        base_url,
        extra_headers=extra_headers,
        session=session,
    ) as active_session:
        result = await active_session.list_tools()
        mcp_tools: List[McpTool] = list(result.tools)

    label_prefix: str = f"{server_label}__" if server_label else ""

    tool_params: List[ToolParam] = [
        _mcp_tool_to_tool_param(t, label_prefix=label_prefix) for t in mcp_tools
    ]

    return [tp for tp in tool_params if filter_tool(tp)]


async def build_mcp_tools_and_invokers(
    servers: Sequence["MCPServerConfig"],
) -> Tuple[List[ToolParam], Dict[str, ToolInvokerFn]]:
    """Build ``(tools, tool_invokers)`` for ``stream_until_user_input``.

    When a server config includes an initialized *session*, both tool
    discovery and invocation reuse that session without reconnecting.
    """
    tools: List[ToolParam] = []
    tool_invokers: Dict[str, ToolInvokerFn] = {}

    for server in servers:
        server_tools = await list_mcp_tools(
            server.server_url,
            server_label=server.server_label,
            filter_tool=server.filter_tool,
            extra_headers=server.extra_headers,
            session=server.session,
        )

        for tool in server_tools:
            tool_name = tool["function"]["name"]
            if tool_name in tool_invokers:
                raise ValueError(f"Duplicate MCP tool name: {tool_name}")

            tool_invokers[tool_name] = _make_mcp_tool_invoker(
                server.server_url,
                tool_name,
                server_label=server.server_label,
                meta=server.meta,
                extra_headers=server.extra_headers,
                session=server.session,
            )

        tools.extend(server_tools)

    return tools, tool_invokers


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for exposing one MCP server as chat tools."""

    server_url: str
    server_label: Optional[Text] = None
    meta: Optional[Dict[str, Any]] = None
    extra_headers: Optional[Dict[str, str]] = None
    session: Optional[ClientSession] = None
    filter_tool: ToolFilterFn = lambda _: True


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


def _make_mcp_tool_invoker(
    server_url: str,
    tool_name: str,
    *,
    server_label: Optional[Text] = None,
    meta: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    session: Optional[ClientSession] = None,
) -> ToolInvokerFn:
    """Create a ``stream_until_user_input``-compatible invoker."""

    async def _invoke(arguments: str, context: Any) -> str:
        return await call_mcp_tool(
            server_url,
            tool_name,
            arguments,
            server_label=server_label,
            meta=_merge_mcp_meta(meta, context),
            extra_headers=extra_headers,
            session=session,
        )

    return _invoke


def _merge_mcp_meta(
    meta: Optional[Dict[str, Any]],
    context: Any,
) -> Optional[Dict[str, Any]]:
    """Merge invoker context into MCP meta without mutating caller data."""
    if context is None:
        return meta

    if meta is None:
        return {"context": context}

    meta_copy = json.loads(json.dumps(meta, default=str))

    if "context" in meta:
        logger.warning("MCP meta key 'context' was overwritten by invoker context")

    meta_copy["context"] = context
    return meta_copy


@asynccontextmanager
async def _get_mcp_session(
    base_url: str,
    *,
    extra_headers: Optional[Dict[str, str]] = None,
    session: Optional[ClientSession] = None,
) -> AsyncIterator[ClientSession]:
    """Yield an initialized MCP session.

    If *session* is provided, it must already be initialized and is
    yielded directly without endpoint discovery or reconnection.
    """
    base_url = base_url.rstrip("/")

    if session is not None:
        if session.get_server_capabilities() is None:
            raise ValueError("session must be initialized before reuse")
        yield session
        return

    for endpoint, transport in _connection_candidates(base_url):
        logger.debug("Trying MCP %s endpoint: %s", transport, endpoint)

        try:
            async with _mcp_transport(endpoint, transport, extra_headers) as (
                read,
                write,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    _endpoint_cache[base_url] = (endpoint, transport)
                    logger.info("Connected to MCP at %s via %s", endpoint, transport)
                    yield session
                    return

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
        f"(tried transports: streamable_http, sse)"
    )


def _connection_candidates(base_url: str) -> List[Tuple[str, _TransportType]]:
    """Return ``(endpoint, transport)`` pairs to try, cached first."""
    candidates: List[Tuple[str, _TransportType]] = [
        (base_url, "streamable_http"),
    ]
    for path in _COMMON_SSE_PATHS:
        candidates.append((f"{base_url}{path}", "sse"))

    cached = _endpoint_cache.get(base_url)
    if cached and cached in candidates:
        candidates.remove(cached)
        candidates.insert(0, cached)

    return candidates


@asynccontextmanager
async def _mcp_transport(
    endpoint: str,
    transport: _TransportType,
    extra_headers: Optional[Dict[str, str]] = None,
) -> AsyncIterator[Tuple[Any, Any]]:
    """Open the right MCP transport and yield ``(read, write)``."""
    if transport == "streamable_http":
        kwargs: Dict[str, Any] = {}
        if extra_headers:
            kwargs["http_client"] = httpx.AsyncClient(headers=extra_headers)
        async with streamable_http_client(endpoint, **kwargs) as (read, write, _):
            yield read, write
    else:
        async with sse_client(endpoint, headers=extra_headers) as (read, write):
            yield read, write
