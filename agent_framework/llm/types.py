from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class LlmMessage:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    artifact: dict[str, Any] | None = None
    status: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
            "artifact": self.artifact,
            "status": self.status,
        }


@dataclass(slots=True)
class LlmRequest:
    prompt: str = ""
    messages: list[LlmMessage] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | None = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt and not self.messages:
            self.messages.append(LlmMessage(role="user", content=self.prompt))

    @property
    def input_text(self) -> str:
        if self.prompt:
            return self.prompt
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""


@dataclass(slots=True)
class LlmResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_message(self) -> LlmMessage:
        return LlmMessage(role="assistant", content=self.text, tool_calls=list(self.tool_calls))
