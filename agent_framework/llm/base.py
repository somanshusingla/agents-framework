from __future__ import annotations

from typing import AsyncIterator

from agent_framework.config import ModelSpec
from agent_framework.llm.factory import LlmClientFactory, default_llm_client_factory
from agent_framework.llm.types import LlmRequest, LlmResponse


class LlmClient:
    """Factory-backed default LLM client.

    The provider and model come from ``ModelSpec``. Hosts can inject any object
    matching ``LlmClientProtocol`` or register additional providers with the
    default factory.
    """

    def __init__(
        self,
        spec: ModelSpec | None = None,
        *,
        factory: LlmClientFactory | None = None,
    ) -> None:
        self.spec = spec or ModelSpec()
        self._factory = factory or default_llm_client_factory
        self._client = self._factory.create(self.spec)

    async def complete(self, request: LlmRequest) -> LlmResponse:
        return await self._client.complete(request)

    async def stream(self, request: LlmRequest) -> AsyncIterator[str]:
        async for chunk in self._client.stream(request):
            yield chunk
