from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel


def get_strict_json_schema(model: type[BaseModel]) -> ResponseFormatJSONSchema:
    schema = model.model_json_schema()

    # Recursively add additionalProperties: false to all objects
    def add_additional_properties_false(schema_obj):
        if isinstance(schema_obj, dict):
            if schema_obj.get("type") == "object":
                schema_obj["additionalProperties"] = False

            # Handle nested objects and $defs
            for value in schema_obj.values():
                if isinstance(value, dict):
                    add_additional_properties_false(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            add_additional_properties_false(item)

        return schema_obj

    schema = add_additional_properties_false(schema)

    return ResponseFormatJSONSchema(
        type="json_schema",
        json_schema=JSONSchema(
            name=model.__name__,
            description=model.__doc__ or "",
            schema=schema,
            strict=True,
        ),
    )
