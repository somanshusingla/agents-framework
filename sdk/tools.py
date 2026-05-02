from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class BaseTool(ABC):
    name: str
    description: str
    approval_required: bool = False

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> dict[str, Any] | str:
        ...


@dataclass
class FunctionTool(BaseTool):
    name: str
    description: str
    fn: Callable[..., Any | Awaitable[Any]]
    approval_required: bool = False

    async def execute(self, args: dict[str, Any]) -> dict[str, Any] | str:
        result = self.fn(**args)
        if hasattr(result, "__await__"):
            result = await result
        return result


@dataclass
class AgentTool(BaseTool):
    name: str
    description: str
    invoker: Callable[[str], Awaitable[str]]
    approval_required: bool = False

    async def execute(self, args: dict[str, Any]) -> dict[str, Any] | str:
        text = args.get("input", "")
        out = await self.invoker(text)
        return {"output": out}
