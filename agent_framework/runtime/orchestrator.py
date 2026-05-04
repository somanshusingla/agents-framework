from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

from agent_framework.config import AgentSpec
from agent_framework.guardrails.policies import AllowAllGuardrail
from agent_framework.llm.factory import create_llm_client
from agent_framework.llm.types import LlmRequest
from agent_framework.memory.session_manager import SessionManagerProtocol, create_session_manager
from agent_framework.observability.langsmith import LangSmithTracer
from agent_framework.runtime.graph import ReactGraphBuilder
from agent_framework.runtime.state import WorkflowResponse
from agent_framework.tools.registry import ToolRegistry


class Orchestrator:
    def __init__(
        self,
        *,
        spec: AgentSpec | None = None,
        registry: ToolRegistry | None = None,
        llm_client: object | None = None,
        guardrail_policy: object | None = None,
        session_manager: SessionManagerProtocol | None = None,
        tracer: LangSmithTracer | None = None,
    ) -> None:
        self.spec = spec or AgentSpec()
        self.registry = registry or ToolRegistry()
        self.llm_client = llm_client or create_llm_client(self.spec.model)
        self.guardrail_policy = guardrail_policy or AllowAllGuardrail()
        self.sessions = session_manager or create_session_manager(self.spec.persistence)
        self.tracer = tracer or LangSmithTracer()
        self.graph = (
            ReactGraphBuilder(self.spec, self.registry)
            .with_llm_client(self.llm_client)
            .with_session_manager(self.sessions)
            .with_guardrails(self.guardrail_policy)
            .with_tracer(self.tracer)
            .build()
        )
        self.jobs: dict[str, dict[str, Any]] = {}
        self.workflows: dict[str, Callable[[str], Awaitable[str]]] = {}

    def register_workflow(self, name: str, handler: Callable[[str], Awaitable[str]]) -> None:
        self.workflows[name] = handler

    async def invoke(
        self,
        thread_id: str,
        text: str,
        workflow_name: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowResponse:
        run_id = str(uuid.uuid4())
        if workflow_name in self.workflows:
            output = await self.workflows[workflow_name](text)
            return WorkflowResponse(
                run_id=run_id,
                thread_id=thread_id,
                status="completed",
                output=output,
            )

        state = await self.graph.ainvoke(
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "input_text": text,
                "metadata": metadata or {},
            }
        )
        return WorkflowResponse(
            run_id=run_id,
            thread_id=thread_id,
            status=state.get("status", "completed"),
            output=state.get("output"),
            tool_calls=state.get("pending_tool_calls", []),
            usage=state.get("usage", {}),
            pending_approval=state.get("pending_approval", []),
        )

    async def submit_job(self, thread_id: str, text: str, workflow_name: str = "default") -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": "queued", "result": None}

        async def _run() -> None:
            self.jobs[job_id]["status"] = "running"
            try:
                self.jobs[job_id]["result"] = (
                    await self.invoke(thread_id, text, workflow_name=workflow_name)
                ).model_dump()
                self.jobs[job_id]["status"] = "completed"
            except Exception as exc:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["result"] = {"error": str(exc)}

        asyncio.create_task(_run())
        return job_id

    async def stream(self, thread_id: str, text: str, workflow_name: str = "default"):
        run_id = str(uuid.uuid4())
        yield {"event": "run.started", "run_id": run_id, "thread_id": thread_id}
        async for token in self.llm_client.stream(LlmRequest(prompt=text)):
            yield {
                "event": "llm.token",
                "run_id": run_id,
                "thread_id": thread_id,
                "data": {"text": token},
            }
        result = await self.invoke(thread_id, text, workflow_name=workflow_name)
        yield {
            "event": "run.completed",
            "run_id": run_id,
            "thread_id": thread_id,
            "data": result.model_dump(),
        }
