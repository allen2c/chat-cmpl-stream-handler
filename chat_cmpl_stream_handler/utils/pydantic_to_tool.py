import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Type

from openai.types.chat import ChatCompletionToolParam as ToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

from chat_cmpl_stream_handler.utils.camel_to_snake import camel_to_snake

ToolInvokerFn = Callable[[str, Any], Awaitable[str]]
PydanticToolHandlerFn = Callable[[BaseModel, Any], Awaitable[str]]


def pydantic_to_tool(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolParam:
    """Convert a Pydantic model into an OpenAI strict-mode tool."""
    tool_name = name or camel_to_snake(model.__name__)
    tool_description = (
        description
        if description is not None
        else inspect.cleandoc(model.__doc__ or "")
    )
    parameters = _sanitize_schema_for_strict(model.model_json_schema())

    function_def: FunctionDefinition = {
        "name": tool_name,
        "parameters": parameters,
        "strict": True,
    }
    if tool_description:
        function_def["description"] = tool_description

    return ToolParam(type="function", function=function_def)


def build_pydantic_tools_and_invokers(
    configs: Sequence["PydanticToolConfig"],
) -> Tuple[List[ToolParam], Dict[str, ToolInvokerFn]]:
    """Build ``(tools, tool_invokers)`` for ``stream_until_user_input``."""
    tools: List[ToolParam] = []
    tool_invokers: Dict[str, ToolInvokerFn] = {}

    for config in configs:
        tool = pydantic_to_tool(
            config.model,
            name=config.name,
            description=config.description,
        )
        tool_name = tool["function"]["name"]

        if tool_name in tool_invokers:
            raise ValueError(f"Duplicate Pydantic tool name: {tool_name}")

        tools.append(tool)
        tool_invokers[tool_name] = _make_pydantic_tool_invoker(
            config.model,
            config.invoker,
        )

    return tools, tool_invokers


@dataclass(frozen=True)
class PydanticToolConfig:
    """Configuration for exposing one Pydantic-backed tool."""

    model: Type[BaseModel]
    invoker: PydanticToolHandlerFn
    name: Optional[str] = None
    description: Optional[str] = None


def _make_pydantic_tool_invoker(
    model: Type[BaseModel],
    invoker: PydanticToolHandlerFn,
) -> ToolInvokerFn:
    """Create a ``stream_until_user_input``-compatible invoker."""

    async def _invoke(arguments: str, context: Any) -> str:
        validated_input = model.model_validate_json(arguments or "{}")
        return await invoker(validated_input, context)

    return _invoke


def _sanitize_schema_for_strict(schema: Dict[str, object]) -> Dict[str, object]:
    """Recursively transform a JSON Schema to satisfy OpenAI strict mode."""
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
