from dataclasses import dataclass, field
from typing import Any


@dataclass
class LlmRequest:
    prompt: str
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LlmResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
