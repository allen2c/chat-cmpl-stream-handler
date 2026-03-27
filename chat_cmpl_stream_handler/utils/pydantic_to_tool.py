from typing import Type

from openai.types.chat import ChatCompletionToolParam as ToolParam
from pydantic import BaseModel


def pydantic_to_tool(model: Type[BaseModel]) -> ToolParam:
    from chat_cmpl_stream_handler.utils.camel_to_snake import camel_to_snake

    tool_name = camel_to_snake(model.__name__)
    tool_description = model.__doc__ or ""

    parameters = model.model_json_schema()
    parameters.pop("title")
    parameters["additionalProperties"] = False

    return ToolParam(
        type="function",
        function={
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters,
        },
        strict=True,
    )
