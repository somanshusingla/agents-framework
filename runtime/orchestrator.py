from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable

from guardrails.policies import AllowAllGuardrail
from llm.base import LlmClient
from llm.types import LlmRequest
from memory.session_manager import SessionManager
from observability.langsmith import LangSmithTracer
from runtime.graph import build_graph
from runtime.state import ExecutionEvent, ToolCall, WorkflowResponse
from sdk.tools import FunctionTool
from tools.registry import ToolRegistry


class Orchestrator:
    def __init__(
        self,
        llm_client: object | None = None,
        guardrail_policy: object | None = None,
        session_manager: SessionManager | None = None,
        tracer: LangSmithTracer | None = None,
    ) -> None:
        self.registry = ToolRegistry()
        self.registry.register(FunctionTool(name="add", description="Add two numbers", fn=lambda a, b: {"sum": a + b}))
        self.llm_client = llm_client or LlmClient()
        self.guardrail_policy = guardrail_policy or AllowAllGuardrail()
        self.sessions = session_manager or SessionManager()
        self.tracer = tracer or LangSmithTracer()
        self.graph = build_graph(self.llm_client, self.guardrail_policy, self.registry)
        self.jobs: dict[str, dict] = {}
        self.pending_approvals: dict[str, list[ToolCall]] = {}
        self.workflows: dict[str, Callable[[str], Awaitable[str]]] = {}

    def register_workflow(self, name: str, handler: Callable[[str], Awaitable[str]]) -> None:
        self.workflows[name] = handler

    async def invoke(self, thread_id: str, text: str, workflow_name: str = "default") -> WorkflowResponse:
        run_id = str(uuid.uuid4())
        with self.tracer.span("workflow.invoke", {"thread_id": thread_id, "run_id": run_id, "workflow_name": workflow_name}):
            if workflow_name in self.workflows:
                output = await self.workflows[workflow_name](text)
                result = {"output": output, "usage": {}}
            else:
                result = await self.graph.ainvoke({"input_text": text})
        status = "blocked" if result.get("blocked") else "completed"
        response = WorkflowResponse(
            run_id=run_id,
            thread_id=thread_id,
            status=status,
            output=result.get("output"),
            usage=result.get("usage", {}),
        )
        await self.sessions.append_event(
            ExecutionEvent(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                run_id=run_id,
                seq_no=1,
                ts=datetime.now(timezone.utc),
                event_type="final_output",
                payload=response.model_dump(),
            )
        )
        return response

    async def submit_job(self, thread_id: str, text: str, workflow_name: str = "default") -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": "queued", "result": None}

        async def _run() -> None:
            self.jobs[job_id]["status"] = "running"
            try:
                self.jobs[job_id]["result"] = (await self.invoke(thread_id, text, workflow_name=workflow_name)).model_dump()
                self.jobs[job_id]["status"] = "completed"
            except Exception as exc:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["result"] = {"error": str(exc)}

        asyncio.create_task(_run())
        return job_id

    async def stream(self, thread_id: str, text: str, workflow_name: str = "default"):
        run_id = str(uuid.uuid4())
        yield {"event": "run.started", "run_id": run_id, "thread_id": thread_id}
        async for tok in self.llm_client.stream(LlmRequest(prompt=text)):
            yield {"event": "llm.token", "run_id": run_id, "thread_id": thread_id, "data": {"text": tok}}
        res = await self.invoke(thread_id, text, workflow_name=workflow_name)
        yield {"event": "run.completed", "run_id": run_id, "thread_id": thread_id, "data": res.model_dump()}
