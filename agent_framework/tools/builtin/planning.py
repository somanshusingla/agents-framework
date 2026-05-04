from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agent_framework.sdk.compaction import TruncateCompactionStrategy
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
