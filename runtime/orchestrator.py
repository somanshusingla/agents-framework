from __future__ import annotations

import uuid
from runtime.graph import build_graph
from runtime.state import WorkflowResponse
from llm.base import LlmClient
from guardrails.policies import AllowAllGuardrail


class Orchestrator:
    def __init__(self) -> None:
        self.graph = build_graph(LlmClient(), AllowAllGuardrail())

    async def invoke(self, thread_id: str, text: str) -> WorkflowResponse:
        run_id = str(uuid.uuid4())
        result = await self.graph.ainvoke({"input_text": text})
        if result.get("blocked"):
            return WorkflowResponse(run_id=run_id, thread_id=thread_id, status="blocked", output=None)
        return WorkflowResponse(run_id=run_id, thread_id=thread_id, status="completed", output=result.get("output"))
