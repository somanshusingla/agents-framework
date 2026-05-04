from __future__ import annotations

import json
from typing import AsyncIterator

from agent_framework.llm.types import LlmRequest, LlmResponse


class LlmClient:
    """Small deterministic default client used for tests and local smoke runs.

    Production users should inject a provider-backed implementation through the SDK.
    """

    async def complete(self, request: LlmRequest) -> LlmResponse:
        tool_messages = [message for message in request.messages if message.role == "tool"]
        if tool_messages:
            latest = tool_messages[-1]
            text = f"Tool result received from {latest.name}: {latest.content}"
            return LlmResponse(
                text=text,
                usage={
                    "input_tokens": _rough_token_count(request.input_text),
                    "output_tokens": _rough_token_count(text),
                },
            )

        text = f"Echo: {request.input_text}"
        tool_calls: list[dict] = []
        if request.input_text.startswith("tool:"):
            try:
                prefix, payload = request.input_text.split(" ", 1)
                name = prefix.split(":", 1)[1]
                args = json.loads(payload)
                tool_calls = [{"id": "call-1", "name": name, "args": args}]
                text = "Calling tool"
            except Exception:
                text = "Malformed tool call instruction"

        return LlmResponse(
            text=text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": _rough_token_count(request.input_text),
                "output_tokens": _rough_token_count(text),
            },
        )

    async def stream(self, request: LlmRequest) -> AsyncIterator[str]:
        for token in ("Echo:", *request.input_text.split()):
            yield token


def _rough_token_count(text: str) -> int:
    return max(1, len((text or "").split()))
