from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, create_model

from agent_framework.sdk.compaction import ToolCompactionStrategy, TruncateCompactionStrategy


class EmptyArgs(BaseModel):
    pass


class TextInputArgs(BaseModel):
    input: str = ""


@dataclass(slots=True)
class ToolExecutionContext:
    thread_id: str
    run_id: str
    agent_name: str
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionResult:
    content: str
    status: str = "success"
    artifact: dict[str, Any] | None = None
    state_updates: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


class BaseTool(ABC):
    name: str
    description: str
    args_schema: type[BaseModel] = EmptyArgs
    approval_required: bool = False
    idempotent: bool = True
    compaction_strategy: ToolCompactionStrategy = TruncateCompactionStrategy()

    def tool_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

    async def invoke(
        self,
        context: ToolExecutionContext,
        args: dict[str, Any],
    ) -> ToolExecutionResult:
        try:
            validated = self.args_schema.model_validate(args)
            result = await self.execute(context, validated)
            if isinstance(result, ToolExecutionResult):
                return result
            return ToolExecutionResult(content=str(result))
        except Exception as exc:
            return ToolExecutionResult(
                content=f"Error executing {self.name}: {type(exc).__name__}: {exc}",
                status="error",
                artifact={"args": args},
            )

    @abstractmethod
    async def execute(
        self,
        context: ToolExecutionContext,
        args: BaseModel,
    ) -> ToolExecutionResult:
        ...


@dataclass
class FunctionTool(BaseTool):
    name: str
    description: str
    fn: Callable[..., Any | Awaitable[Any]]
    args_schema: type[BaseModel] = EmptyArgs
    approval_required: bool = False
    idempotent: bool = True
    compaction_strategy: ToolCompactionStrategy = field(default_factory=TruncateCompactionStrategy)

    def __post_init__(self) -> None:
        if self.args_schema is EmptyArgs:
            self.args_schema = _schema_from_callable(self.name, self.fn)

    async def execute(
        self,
        context: ToolExecutionContext,
        args: BaseModel,
    ) -> ToolExecutionResult:
        values = args.model_dump()
        result = self.fn(**values)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, ToolExecutionResult):
            return result
        return ToolExecutionResult(content=str(result), artifact={"output": result})


@dataclass
class AgentTool(BaseTool):
    name: str
    description: str
    invoker: Callable[[str], Awaitable[str]]
    args_schema: type[BaseModel] = TextInputArgs
    approval_required: bool = False
    idempotent: bool = True
    compaction_strategy: ToolCompactionStrategy = field(default_factory=TruncateCompactionStrategy)

    async def execute(
        self,
        context: ToolExecutionContext,
        args: BaseModel,
    ) -> ToolExecutionResult:
        values = args.model_dump()
        text = str(values.get("input", ""))
        return ToolExecutionResult(content=await self.invoker(text))


def _schema_from_callable(name: str, fn: Callable[..., Any]) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in inspect.signature(fn).parameters.items():
        if param_name == "context":
            continue
        if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
            continue
        annotation = Any if param.annotation is inspect.Signature.empty else param.annotation
        default = ... if param.default is inspect.Signature.empty else param.default
        fields[param_name] = (annotation, default)
    return create_model(f"{name.title().replace('_', '')}Args", **fields)
