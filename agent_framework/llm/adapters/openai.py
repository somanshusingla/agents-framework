from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agent_framework.config import ModelSpec
from agent_framework.llm.adapters.http import error_message, import_httpx, parse_json_object, resolve_api_key
from agent_framework.llm.types import LlmMessage, LlmRequest, LlmResponse


class OpenAICompatibleChatClient:
    provider_name = "OpenAI-compatible"
    default_api_base = ""
    default_api_key_env = ""
    completions_path = "/chat/completions"
    max_tokens_field = "max_tokens"

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
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    chunk = json.loads(data)
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content

    def _payload(self, request: LlmRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.name,
            "messages": [_to_openai_message(message) for message in request.messages],
            "stream": stream,
        }
        if self.spec.temperature is not None:
            payload["temperature"] = self.spec.temperature
        if self.spec.max_tokens:
            payload[self.max_tokens_field] = self.spec.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice and request.tools:
            payload["tool_choice"] = request.tool_choice
        payload.update(self.spec.extra_body)
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {resolve_api_key(self.spec, self.default_api_key_env, self.provider_name)}",
            "Content-Type": "application/json",
        }
        headers.update(self.spec.extra_headers)
        return headers

    def _url(self) -> str:
        return f"{(self.spec.api_base or self.default_api_base).rstrip('/')}{self.completions_path}"

    def _parse_response(self, data: dict[str, Any]) -> LlmResponse:
        choices = data.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        text = message.get("content") or ""
        tool_calls = [_parse_openai_tool_call(call) for call in message.get("tool_calls") or []]
        usage = data.get("usage") or {}
        return LlmResponse(
            text=text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            raw=data,
        )


class OpenAIChatClient(OpenAICompatibleChatClient):
    provider_name = "OpenAI"
    default_api_base = "https://api.openai.com/v1"
    default_api_key_env = "OPENAI_API_KEY"
    max_tokens_field = "max_completion_tokens"


def _to_openai_message(message: LlmMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }

    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [_to_openai_tool_call(call) for call in message.tool_calls]
    return payload


def _to_openai_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(call.get("id") or call.get("call_id") or ""),
        "type": "function",
        "function": {
            "name": str(call.get("name") or call.get("tool_name") or ""),
            "arguments": json.dumps(call.get("args") or {}, ensure_ascii=True),
        },
    }


def _parse_openai_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "id": str(call.get("id") or ""),
        "name": str(function.get("name") or ""),
        "args": parse_json_object(function.get("arguments")),
    }
