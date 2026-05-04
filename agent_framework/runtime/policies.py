from __future__ import annotations

from typing import Any, Protocol

from agent_framework.config import RuntimeSpec
from agent_framework.sdk.tools import ToolExecutionContext, ToolExecutionResult


class ToolCallPolicy(Protocol):
    async def check(
        self,
        context: ToolExecutionContext,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolExecutionResult | None:
        ...


class ConfigurableToolCallPolicy:
    def __init__(self, runtime: RuntimeSpec) -> None:
        self._runtime = runtime

    async def check(
        self,
        context: ToolExecutionContext,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolExecutionResult | None:
        state = context.state
        history = list(state.get("tool_call_history", []))

        if self._runtime.required_first_tool and not history:
            if tool_name != self._runtime.required_first_tool:
                return ToolExecutionResult(
                    content=(
                        "Tool blocked: first model-controlled action must be "
                        f"{self._runtime.required_first_tool}."
                    ),
                    status="error",
                    artifact={"args": args, "blocked_by": "required_first_tool"},
                )

        if (
            self._runtime.final_tool
            and self._runtime.final_requires_reflection
            and tool_name == self._runtime.final_tool
            and not _has_ready_final_reflection(state)
        ):
            return ToolExecutionResult(
                content=(
                    "Tool blocked: record a final reflection with ready_for_final=true "
                    f"before calling {self._runtime.final_tool}."
                ),
                status="error",
                artifact={"args": args, "blocked_by": "final_reflection_required"},
            )

        return None


def _has_ready_final_reflection(state: dict[str, Any]) -> bool:
    for reflection in reversed(list(state.get("reflections", []))):
        if reflection.get("ready_for_final") is True:
            return True
    return False
