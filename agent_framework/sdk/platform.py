from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent_framework.config import AgentSpec, load_agent_spec
from agent_framework.guardrails.policies import AllowAllGuardrail
from agent_framework.llm.base import LlmClient
from agent_framework.memory.session_manager import SessionManager
from agent_framework.observability.langsmith import LangSmithTracer
from agent_framework.runtime.orchestrator import Orchestrator
from agent_framework.sdk.guardrails import GuardrailPolicyProtocol
from agent_framework.sdk.llm import LlmClientProtocol
from agent_framework.sdk.sessions import SessionManagerProtocol
from agent_framework.sdk.tools import BaseTool
from agent_framework.sdk.tracing import TracerProtocol
from agent_framework.tools.registry import ToolRegistry

WorkflowHandler = Callable[[str], Awaitable[str]]


@dataclass
class AgentPlatformLibrary:
    """Composition root for embedding configured agents in a host application."""

    spec: AgentSpec = field(default_factory=AgentSpec)
    llm_client: LlmClientProtocol = field(default_factory=LlmClient)
    guardrail_policy: GuardrailPolicyProtocol = field(default_factory=AllowAllGuardrail)
    session_manager: SessionManagerProtocol = field(default_factory=SessionManager)
    tracer: TracerProtocol = field(default_factory=LangSmithTracer)
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    _workflows: dict[str, WorkflowHandler] = field(default_factory=dict)

    @classmethod
    def from_config(cls, path: str | Path, **overrides) -> "AgentPlatformLibrary":
        return cls(spec=load_agent_spec(path), **overrides)

    def register_tool(self, tool: BaseTool) -> None:
        self.registry.register(tool)

    def register_tools(self, tools: list[BaseTool]) -> None:
        self.registry.register_many(tools)

    def register_workflow(self, name: str, handler: WorkflowHandler) -> None:
        self._workflows[name] = handler

    def build(self) -> Orchestrator:
        orchestrator = Orchestrator(
            spec=self.spec,
            registry=self.registry,
            llm_client=self.llm_client,
            guardrail_policy=self.guardrail_policy,
            session_manager=self.session_manager,
            tracer=self.tracer,
        )
        for name, handler in self._workflows.items():
            orchestrator.register_workflow(name, handler)
        return orchestrator
