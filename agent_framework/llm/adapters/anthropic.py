from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agent_framework.config import ModelSpec
from agent_framework.llm.adapters.http import error_message, import_httpx, resolve_api_key
from agent_framework.llm.types import LlmMessage, LlmRequest, LlmResponse


class AnthropicMessagesClient:
    provider_name = "Anthropic"
    default_api_base = "https://api.anthropic.com/v1"
    default_api_key_env = "ANTHROPIC_API_KEY"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    async def complete(self, request: LlmRequest) -> LlmResponse:
        httpx = import_httpx()
        payload = self._payload(request, stream=False)
        async with httpx.AsyncClient(timeout=self.spec.timeout_seconds) as client:
            response = await client.post(self._url(), headers=self._headers(), json=payload)
        if response.status_code >= 400:
            raise error_message(self.provider_name, response.status_code, response.text)
        return self._parse_response(response.json())

    async def stream(self, request: LlmRequest) -> AsyncIterator[str]:
        httpx = import_httpx()
        payload = self._payload(request, stream=True)
        async with httpx.AsyncClient(timeout=self.spec.timeout_seconds) as client:
            async with client.stream("POST", self._url(), headers=self._headers(), json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise error_message(self.provider_name, response.status_code, body.decode("utf-8", errors="ignore"))
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data:
                        continue
                    event = json.loads(data)
                    if event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield str(delta["text"])

    def _payload(self, request: LlmRequest, *, stream: bool) -> dict[str, Any]:
        system, messages = _to_anthropic_messages(request.messages)
        payload: dict[str, Any] = {
            "model": self.spec.name,
            "max_tokens": self.spec.max_tokens,
            "messages": messages,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if self.spec.temperature is not None:
            payload["temperature"] = self.spec.temperature
        if request.tools:
            payload["tools"] = [_to_anthropic_tool(tool) for tool in request.tools]
        if request.tool_choice and request.tools:
            payload["tool_choice"] = {"type": "auto"}
        payload.update(self.spec.extra_body)
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-api-key": resolve_api_key(self.spec, self.default_api_key_env, self.provider_name),
            "anthropic-version": self.spec.extra_headers.get("anthropic-version", "2023-06-01"),
            "Content-Type": "application/json",
        }
        headers.update(self.spec.extra_headers)
        return headers

    def _url(self) -> str:
        return f"{(self.spec.api_base or self.default_api_base).rstrip('/')}/messages"

    def _parse_response(self, data: dict[str, Any]) -> LlmResponse:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "args": dict(block.get("input") or {}),
                    }
                )
        usage = data.get("usage") or {}
        return LlmResponse(
            text="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            usage={
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
            },
            raw=data,
        )


def _to_anthropic_messages(messages: list[LlmMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            converted.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue
        if message.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )
            continue

        flush_tool_results()
        if message.role == "assistant":
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id") or call.get("call_id") or ""),
                        "name": str(call.get("name") or call.get("tool_name") or ""),
                        "input": dict(call.get("args") or {}),
                    }
                )
            converted.append({"role": "assistant", "content": content or ""})
        else:
            converted.append({"role": "user", "content": message.content})

    flush_tool_results()
    return "\n\n".join(part for part in system_parts if part), converted


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or {}
    return {
        "name": function.get("name") or tool.get("name"),
        "description": function.get("description") or tool.get("description") or "",
        "input_schema": function.get("parameters") or tool.get("input_schema") or {"type": "object"},
    }
