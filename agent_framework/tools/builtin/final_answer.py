from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent_framework.sdk.compaction import FinalReportCompactionStrategy
from agent_framework.sdk.tools import BaseTool, ToolExecutionContext, ToolExecutionResult


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
