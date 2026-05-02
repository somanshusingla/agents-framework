from __future__ import annotations

import json
from typing import AsyncIterator
from llm.types import LlmRequest, LlmResponse


class LlmClient:
    async def complete(self, request: LlmRequest) -> LlmResponse:
        tool_calls = []
        text = f"Echo: {request.prompt}"
        if request.prompt.startswith("tool:"):
            # format: tool:<name> {json}
            try:
                prefix, payload = request.prompt.split(" ", 1)
                name = prefix.split(":", 1)[1]
                args = json.loads(payload)
                tool_calls = [{"tool_name": name, "args": args}]
                text = "Calling tool"
            except Exception:
                text = "Malformed tool call instruction"
        return LlmResponse(text=text, tool_calls=tool_calls, usage={"input_tokens": len(request.prompt.split()), "output_tokens": len(text.split())})

    async def stream(self, request: LlmRequest) -> AsyncIterator[str]:
        for tok in ("Echo:", *request.prompt.split()):
            yield tok
