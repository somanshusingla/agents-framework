from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Awaitable

from guardrails.policies import AllowAllGuardrail
from llm.base import LlmClient
from memory.session_manager import SessionManager
from observability.langsmith import LangSmithTracer
from runtime.orchestrator import Orchestrator
from sdk.guardrails import GuardrailPolicyProtocol
from sdk.llm import LlmClientProtocol
from sdk.sessions import SessionManagerProtocol
from sdk.tools import BaseTool
from sdk.tracing import TracerProtocol


WorkflowHandler = Callable[[str], Awaitable[str]]


@dataclass
class AgentPlatformLibrary:
    """Library-first SDK for composing and embedding the framework."""

    llm_client: LlmClientProtocol = field(default_factory=LlmClient)
    guardrail_policy: GuardrailPolicyProtocol = field(default_factory=AllowAllGuardrail)
    session_manager: SessionManagerProtocol = field(default_factory=SessionManager)
    tracer: TracerProtocol = field(default_factory=LangSmithTracer)
    _tools: list[BaseTool] = field(default_factory=list)
    _workflows: dict[str, WorkflowHandler] = field(default_factory=dict)

    def register_tool(self, tool: BaseTool) -> None:
        self._tools.append(tool)

    def register_workflow(self, name: str, handler: WorkflowHandler) -> None:
        self._workflows[name] = handler

    def build(self) -> Orchestrator:
        orchestrator = Orchestrator(
            llm_client=self.llm_client,
            guardrail_policy=self.guardrail_policy,
            session_manager=self.session_manager,
            tracer=self.tracer,
        )
        for tool in self._tools:
            orchestrator.registry.register(tool)
        for name, handler in self._workflows.items():
            orchestrator.register_workflow(name, handler)
        return orchestrator
