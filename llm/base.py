from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LlmRequest:
    prompt: str


@dataclass
class LlmResponse:
    text: str
    tool_calls: list[dict] | None = None


class LlmClient:
    async def complete(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(text=f"Echo: {request.prompt}", tool_calls=[])
