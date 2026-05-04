from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_framework.sdk.compaction import FinalReportCompactionStrategy, TruncateCompactionStrategy
from agent_framework.sdk.tools import BaseTool, ToolExecutionContext, ToolExecutionResult


class TaskItem(BaseModel):
    content: str
    status: Literal["pending", "in_progress", "completed", "blocked"] = "pending"


class PlanArgs(BaseModel):
    goal: str
    tasks: list[TaskItem]


class PlanTool(BaseTool):
    name = "plan"
    description = """Create or update the current task plan.

WHEN TO USE:
- Before acting on complex multi-step tasks.
- When the current plan is incomplete or reflection indicates replanning is needed.

WHEN NOT TO USE:
- For simple single-step tasks.
- After every tool call when the existing plan is still valid.

HOW TO USE:
- Regenerate the full task list with current statuses.
- Mark completed tasks as completed.
- Mark the next task to work on as in_progress.
- Keep future tasks as pending.
"""
    args_schema = PlanArgs
    compaction_strategy = TruncateCompactionStrategy("Task plan", 2_000)

    async def execute(
        self,
        context: ToolExecutionContext,
        args: PlanArgs,
    ) -> ToolExecutionResult:
        plan = args.model_dump()
        lines = [f"Goal: {args.goal}"]
        markers = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
            "blocked": "[!]",
        }
        lines.extend(f"{markers[task.status]} {task.content}" for task in args.tasks)
        return ToolExecutionResult(
            content="\n".join(lines),
            artifact={"plan": plan},
            state_updates={"plan": plan},
        )


class CreateTasksTool(PlanTool):
    """Backward-compatible alias for callers importing the old class name.

    The model-facing tool name is intentionally `plan`.
    """


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


class FinalAnswerArgs(BaseModel):
    output: str | dict[str, Any]


class FinalAnswerTool(BaseTool):
    name = "final_answer"
    description = "Submit the final answer for the configured agent workflow."
    args_schema = FinalAnswerArgs
    compaction_strategy = FinalReportCompactionStrategy()

    async def execute(
        self,
        context: ToolExecutionContext,
        args: FinalAnswerArgs,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            content="Final answer accepted.",
            artifact={"output": args.output},
            state_updates={"final_response": args.output},
        )
