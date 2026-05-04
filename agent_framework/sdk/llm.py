from __future__ import annotations

from typing import Protocol, AsyncIterator

from agent_framework.llm.types import LlmRequest, LlmResponse


class LlmClientProtocol(Protocol):
    async def complete(self, request: LlmRequest) -> LlmResponse: ...
    async def stream(self, request: LlmRequest) -> AsyncIterator[str]: ...
