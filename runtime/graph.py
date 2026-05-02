from __future__ import annotations

from typing import Any, TypedDict
from langgraph.graph import END, StateGraph
from guardrails.policies import AllowAllGuardrail
from llm.base import LlmClient
from llm.types import LlmRequest
from tools.registry import ToolRegistry


class AgentState(TypedDict, total=False):
    input_text: str
    output: str
    blocked: bool
    pending_tool: dict[str, Any] | None


def build_graph(llm_client: LlmClient, guardrail: AllowAllGuardrail, registry: ToolRegistry):
    async def validate_input(state: AgentState):
        res = await guardrail.validate_input(state["input_text"])
        return {"blocked": not res.allowed}

    async def think(state: AgentState):
        req = LlmRequest(prompt=state["input_text"], tools=registry.specs())
        resp = await llm_client.complete(req)
        pending = resp.tool_calls[0] if resp.tool_calls else None
        return {"output": resp.text, "pending_tool": pending}

    async def execute_tool(state: AgentState):
        call = state.get("pending_tool")
        if not call:
            return {}
        tool = registry.get(call.get("tool_name", ""))
        if not tool:
            return {"output": f"{state.get('output','')}\nTool not found."}
        result = await tool.execute(call.get("args", {}))
        return {"output": f"{state.get('output','')}\nTool result: {result}", "pending_tool": None}

    async def validate_output(state: AgentState):
        _ = await guardrail.validate_output(state.get("output", ""))
        return {}

    def route_after_input(state: AgentState):
        return "end" if state.get("blocked") else "think"

    def route_after_think(state: AgentState):
        return "tool" if state.get("pending_tool") else "validate"

    graph = StateGraph(AgentState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("think", think)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("validate_output", validate_output)
    graph.set_entry_point("validate_input")
    graph.add_conditional_edges("validate_input", route_after_input, {"think": "think", "end": END})
    graph.add_conditional_edges("think", route_after_think, {"tool": "execute_tool", "validate": "validate_output"})
    graph.add_edge("execute_tool", "validate_output")
    graph.add_edge("validate_output", END)
    return graph.compile()
