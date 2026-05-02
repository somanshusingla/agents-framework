from __future__ import annotations

from typing import TypedDict
from langgraph.graph import END, StateGraph

from llm.base import LlmClient, LlmRequest
from guardrails.policies import AllowAllGuardrail


class AgentState(TypedDict, total=False):
    input_text: str
    output: str
    blocked: bool


def build_graph(llm_client: LlmClient, guardrail: AllowAllGuardrail):
    async def validate_input(state: AgentState):
        res = await guardrail.validate_input(state["input_text"])
        return {"blocked": not res.allowed}

    async def think(state: AgentState):
        resp = await llm_client.complete(LlmRequest(prompt=state["input_text"]))
        return {"output": resp.text}

    async def validate_output(state: AgentState):
        _ = await guardrail.validate_output(state.get("output", ""))
        return {}

    def route_after_input(state: AgentState):
        return "end" if state.get("blocked") else "think"

    graph = StateGraph(AgentState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("think", think)
    graph.add_node("validate_output", validate_output)
    graph.set_entry_point("validate_input")
    graph.add_conditional_edges("validate_input", route_after_input, {"think": "think", "end": END})
    graph.add_edge("think", "validate_output")
    graph.add_edge("validate_output", END)
    return graph.compile()
