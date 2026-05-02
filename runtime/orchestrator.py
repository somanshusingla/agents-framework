from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from llm.types import LlmRequest
from runtime.graph import build_graph
from runtime.state import WorkflowResponse, ExecutionEvent, ToolCall
from llm.base import LlmClient
from guardrails.policies import AllowAllGuardrail
from tools.registry import ToolRegistry
from memory.session_manager import SessionManager
from observability.langsmith import LangSmithTracer
from sdk.tools import FunctionTool


class Orchestrator:
    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(FunctionTool(name="add", description="Add two numbers", fn=lambda a, b: {"sum": a + b}))
        self.graph = build_graph(LlmClient(), AllowAllGuardrail(), self.registry)
        self.sessions = SessionManager()
        self.tracer = LangSmithTracer()
        self.jobs: dict[str, dict] = {}
        self.pending_approvals: dict[str, list[ToolCall]] = {}

    async def invoke(self, thread_id: str, text: str) -> WorkflowResponse:
        run_id = str(uuid.uuid4())
        with self.tracer.span("workflow.invoke", {"thread_id": thread_id, "run_id": run_id}):
            result = await self.graph.ainvoke({"input_text": text})
        status = "blocked" if result.get("blocked") else "completed"
        response = WorkflowResponse(run_id=run_id, thread_id=thread_id, status=status, output=result.get("output"), usage=result.get("usage", {}))
        await self.sessions.append_event(
            ExecutionEvent(id=str(uuid.uuid4()), thread_id=thread_id, run_id=run_id, seq_no=1, ts=datetime.now(timezone.utc), event_type="final_output", payload=response.model_dump())
        )
        return response

    async def submit_job(self, thread_id: str, text: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": "queued", "result": None}

        async def _run():
            self.jobs[job_id]["status"] = "running"
            try:
                self.jobs[job_id]["result"] = (await self.invoke(thread_id, text)).model_dump()
                self.jobs[job_id]["status"] = "completed"
            except Exception as exc:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["result"] = {"error": str(exc)}

        asyncio.create_task(_run())
        return job_id

    async def stream(self, thread_id: str, text: str):
        run_id = str(uuid.uuid4())
        yield {"event": "run.started", "run_id": run_id, "thread_id": thread_id}
        client = LlmClient()
        async for tok in client.stream(LlmRequest(prompt=text)):
            yield {"event": "llm.token", "run_id": run_id, "thread_id": thread_id, "data": {"text": tok}}
        res = await self.invoke(thread_id, text)
        yield {"event": "run.completed", "run_id": run_id, "thread_id": thread_id, "data": res.model_dump()}
