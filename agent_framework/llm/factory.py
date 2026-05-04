from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from agent_framework.config import ModelSpec
from agent_framework.llm.types import LlmRequest, LlmResponse


class LlmClientProtocol(Protocol):
    async def complete(self, request: LlmRequest) -> LlmResponse: ...
    async def stream(self, request: LlmRequest): ...


LlmProviderFactory = Callable[[ModelSpec], LlmClientProtocol]


class LlmClientFactory:
    def __init__(self) -> None:
        self._providers: dict[str, LlmProviderFactory] = {}
        self._builtins_loaded = False

    def register(
        self,
        provider: str,
        factory: LlmProviderFactory,
        *,
        replace: bool = False,
    ) -> None:
        key = _normalize_provider(provider)
        if key in self._providers and not replace:
            raise ValueError(f"LLM provider already registered: {provider}")
        self._providers[key] = factory

    def provider(self, provider: str, *, replace: bool = False):
        def decorator(factory: LlmProviderFactory) -> LlmProviderFactory:
            self.register(provider, factory, replace=replace)
            return factory

        return decorator

    def create(self, spec: ModelSpec) -> LlmClientProtocol:
        self._ensure_builtin_providers()
        key = _normalize_provider(spec.provider)
        try:
            factory = self._providers[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers)) or "none"
            raise ValueError(
                f"Unknown LLM provider '{spec.provider}'. Registered providers: {available}"
            ) from exc
        return factory(spec)

    def providers(self) -> list[str]:
        self._ensure_builtin_providers()
        return sorted(self._providers)

    def _ensure_builtin_providers(self) -> None:
        if self._builtins_loaded:
            return
        self._builtins_loaded = True

        from agent_framework.llm.adapters.anthropic import AnthropicMessagesClient
        from agent_framework.llm.adapters.deepseek import DeepSeekChatClient
        from agent_framework.llm.adapters.openai import OpenAIChatClient

        self._providers.setdefault("openai", OpenAIChatClient)
        self._providers.setdefault("gpt", OpenAIChatClient)
        self._providers.setdefault("anthropic", AnthropicMessagesClient)
        self._providers.setdefault("claude", AnthropicMessagesClient)
        self._providers.setdefault("deepseek", DeepSeekChatClient)


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")


default_llm_client_factory = LlmClientFactory()


def create_llm_client(spec: ModelSpec, *, factory: LlmClientFactory | None = None) -> LlmClientProtocol:
    return (factory or default_llm_client_factory).create(spec)


def register_llm_provider(
    provider: str,
    factory: LlmProviderFactory,
    *,
    replace: bool = False,
) -> None:
    default_llm_client_factory.register(provider, factory, replace=replace)


def llm_provider(provider: str, *, replace: bool = False):
    return default_llm_client_factory.provider(provider, replace=replace)
