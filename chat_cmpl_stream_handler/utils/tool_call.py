import json
from typing import Any, Dict

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)


def args_from_tool_call(tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
    """Parse tool call arguments JSON into a dictionary."""
    return (
        json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
    )
