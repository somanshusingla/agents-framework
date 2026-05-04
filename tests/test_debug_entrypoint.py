import asyncio

from pydantic import BaseModel

from agent_framework.config import AgentSpec
from agent_framework.llm.types import LlmRequest, LlmResponse
from agent_framework.runtime.graph import build_react_graph
from agent_framework.sdk.tools import BaseTool, ToolExecutionContext, ToolExecutionResult
from agent_framework.tools.builtin import FinalAnswerTool, PlanTool, ReflectTool
from agent_framework.tools.registry import ToolRegistry


class DebugEchoArgs(BaseModel):
    value: str


class DebugEchoTool(BaseTool):
    name = "debug_echo"
    description = "Echo a value for debugger smoke testing."
    args_schema = DebugEchoArgs

    async def execute(
        self,
        context: ToolExecutionContext,
        args: DebugEchoArgs,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(content=f"debug:{args.value}")


class DebugScriptedLlm:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []
        self.responses = [
            LlmResponse(
                text="calling debug_echo",
                tool_calls=[
                    {"id": "debug-call-1", "name": "debug_echo", "args": {"value": "ok"}}
                ],
            ),
            LlmResponse(
                text="submitting final answer",
                tool_calls=[
                    {
                        "id": "debug-call-2",
                        "name": "final_answer",
                        "args": {"output": "debug smoke passed"},
                    }
                ],
            ),
        ]

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return self.responses.pop(0)


async def run_debug_smoke() -> dict:
    spec = AgentSpec(
        name="debug-agent",
        system_prompt="You are running a debugger smoke test.",
        tools={"enabled": ["plan", "reflect", "debug_echo", "final_answer"]},
        runtime={"max_steps": 4, "final_tool": "final_answer"},
    )
    registry = ToolRegistry([PlanTool(), ReflectTool(), DebugEchoTool(), FinalAnswerTool()])
    llm = DebugScriptedLlm()
    graph = build_react_graph(spec, registry, llm_client=llm)

    state = await graph.ainvoke({"thread_id": "debug-thread", "input_text": "run debug smoke"})

    assert state["status"] == "completed"
    assert state["output"] == "debug smoke passed"
    assert state["tool_call_history"] == ["debug_echo", "final_answer"]
    assert len(llm.requests) == 2
    return state


def test_debug_entrypoint_smoke():
    asyncio.run(run_debug_smoke())


if __name__ == "__main__":
    result = asyncio.run(run_debug_smoke())
    print(f"Debug smoke passed: {result['output']}")
