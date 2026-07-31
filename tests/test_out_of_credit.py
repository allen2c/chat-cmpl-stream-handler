"""The out-of-credit predicate that keeps CI honest.

`is_out_of_credit` decides whether a provider failure is a billing state (skip) or
something worth seeing (fail). Getting it wrong in the generous direction turns real
failures green, so the strings below are copied verbatim from real provider responses —
including the Gemini spend cap that arrives as a 429 and reads exactly like a rate limit.
"""

import httpx
import pytest
from openai import APIStatusError

from tests.conftest import is_out_of_credit

GEMINI_SPEND_CAP = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'Your project has exceeded "
    "its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to "
    "manage your project spend cap.', 'status': 'RESOURCE_EXHAUSTED'}}]"
)

CANNOT_PAY = [
    pytest.param(GEMINI_SPEND_CAP, id="gemini-openai-path"),
    pytest.param(f"429 Too Many Requests. {{'message': '{GEMINI_SPEND_CAP}'}}", id="gemini-genai-path"),
    pytest.param(f"litellm.RateLimitError: Vertex_ai_betaException - b'{GEMINI_SPEND_CAP}'", id="gemini-litellm-path"),
    pytest.param(
        "Error code: 429 - {'error': {'message': 'You exceeded your current quota, "
        "please check your plan and billing details.', 'type': 'insufficient_quota'}}",
        id="openai-insufficient-quota",
    ),
]

WORTH_SEEING = [
    pytest.param(
        "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4.1-nano "
        "on requests per min (RPM): Limit 500, Used 500. Please try again in 120ms.'}}",
        id="a-real-rate-limit",
    ),
    pytest.param("Error code: 500 - internal server error", id="provider-outage"),
    pytest.param("Error code: 401 - invalid api key", id="bad-key"),
    pytest.param("the run did not complete", id="an-ordinary-assertion"),
]


@pytest.mark.parametrize("message", CANNOT_PAY)
def test_a_billing_failure_is_recognised(message: str):
    assert is_out_of_credit(RuntimeError(message))


@pytest.mark.parametrize("message", WORTH_SEEING)
def test_everything_else_still_fails(message: str):
    assert not is_out_of_credit(RuntimeError(message))


def test_a_402_needs_no_message_at_all():
    """Hugging Face returns 402 with wording we do not control. The status is enough."""
    request = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    response = httpx.Response(402, request=request, json={"error": "nope"})

    assert is_out_of_credit(APIStatusError("nope", response=response, body=None))
