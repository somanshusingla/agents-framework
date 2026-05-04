from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_framework.sdk.compaction import TruncateCompactionStrategy
from agent_framework.sdk.tools import BaseTool, ToolExecutionContext, ToolExecutionResult


class ReflectArgs(BaseModel):
    reflection_type: str = Field(description="Why reflection is being recorded")
    analysis: str
    need_replan: bool = False
    ready_for_final: bool = False


class ReflectTool(BaseTool):
    name = "reflect"
    description = "Record progress, failure analysis, synthesis, or final self-check before continuing."
    args_schema = ReflectArgs
    compaction_strategy = TruncateCompactionStrategy("Reflection", 1_500)

    async def execute(
        self,
        context: ToolExecutionContext,
        args: ReflectArgs,
    ) -> ToolExecutionResult:
        reflection = args.model_dump()
        reflections = list(context.state.get("reflections", [])) + [reflection]
        updates: dict[str, Any] = {"reflections": reflections}
        if args.need_replan:
            updates["must_replan"] = True
        return ToolExecutionResult(
            content=(
                f"Reflection recorded ({args.reflection_type}). "
                f"need_replan={args.need_replan}; ready_for_final={args.ready_for_final}\n"
                f"{args.analysis}"
            ),
            artifact={"reflection": reflection},
            state_updates=updates,
        )
