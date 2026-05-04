import asyncio

from pydantic import BaseModel

from agent_framework.config import AgentSpec, ModelSpec, load_agent_spec
from agent_framework.llm.factory import LlmClientFactory
from agent_framework.llm.types import LlmMessage, LlmRequest, LlmResponse
from agent_framework.memory.session_manager import InMemorySessionManager as FrameworkInMemorySessionManager
from agent_framework.runtime.graph import ReactGraphBuilder, build_react_graph
from agent_framework.sdk import AgentPlatformLibrary, ToolExecutionContext, ToolExecutionResult
from agent_framework.sdk.tools import BaseTool
from agent_framework.tools.builtin import FinalAnswerTool, PlanTool, ReflectTool
from agent_framework.tools.registry import ToolRegistry


class InMemorySessionManager:
    def __init__(self) -> None:
        self.events = []

    async def load_events(self, thread_id: str):
        return [event for event in self.events if event.thread_id == thread_id]

    async def append_event(self, event, workflow_name: str = "default", agent_name: str = "default") -> None:
        self.events.append(event)


class LookupArgs(BaseModel):
    query: str


class LookupTool(BaseTool):
    name = "lookup"
    description = "Lookup test data"
    args_schema = LookupArgs

    async def execute(self, context: ToolExecutionContext, args: LookupArgs) -> ToolExecutionResult:
        return ToolExecutionResult(
            content=f"lookup:{args.query}",
            artifact={"query": args.query},
            state_updates={"key_findings": [f"found {args.query}"]},
        )


class ScriptedLlm:
    def __init__(self, responses: list[LlmResponse]) -> None:
        self.responses = responses
        self.requests: list[LlmRequest] = []

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request: LlmRequest):
        yield "ok"


def test_load_agent_spec_from_yaml(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are configured.", encoding="utf-8")
    config = tmp_path / "agent.yaml"
    config.write_text(
        """
agent:
  name: configured-agent
  workflow: react
  system_prompt_path: system.md
tools:
  enabled: [lookup]
context:
  token_threshold: 100
runtime:
  max_steps: 4
""",
        encoding="utf-8",
    )

    spec = load_agent_spec(config)

    assert spec.name == "configured-agent"
    assert spec.system_prompt == "You are configured."
    assert spec.tools.enabled == ["lookup"]
    assert spec.runtime.max_steps == 4


def test_reusable_graph_executes_configured_agent_end_to_end():
    async def run():
        spec = AgentSpec(
            name="agent-a",
            system_prompt="Use tools.",
            tools={"enabled": ["lookup", "final_answer"]},
            runtime={"max_steps": 4, "final_tool": "final_answer"},
        )
        registry = ToolRegistry([LookupTool(), FinalAnswerTool()])
        llm = ScriptedLlm(
            [
                LlmResponse(
                    text="calling lookup",
                    tool_calls=[{"id": "1", "name": "lookup", "args": {"query": "alpha"}}],
                ),
                LlmResponse(
                    text="final",
                    tool_calls=[
                        {"id": "2", "name": "final_answer", "args": {"output": "done alpha"}}
                    ],
                ),
            ]
        )
        graph = build_react_graph(spec, registry, llm_client=llm, session_manager=InMemorySessionManager())

        state = await graph.ainvoke({"thread_id": "t", "input_text": "run"})

        assert state["status"] == "completed"
        assert state["output"] == "done alpha"
        assert state["tool_call_history"] == ["lookup", "final_answer"]

    asyncio.run(run())


def test_same_graph_template_supports_different_agent_configs():
    async def run():
        registry = ToolRegistry([LookupTool(), FinalAnswerTool()])
        spec_a = AgentSpec(
            name="agent-a",
            system_prompt="Prompt A",
            tools={"enabled": ["final_answer"]},
            runtime={"final_tool": "final_answer"},
        )
        spec_b = AgentSpec(
            name="agent-b",
            system_prompt="Prompt B",
            tools={"enabled": ["final_answer"]},
            runtime={"final_tool": "final_answer"},
        )
        llm_a = ScriptedLlm(
            [LlmResponse(text="a", tool_calls=[{"id": "a", "name": "final_answer", "args": {"output": "A"}}])]
        )
        llm_b = ScriptedLlm(
            [LlmResponse(text="b", tool_calls=[{"id": "b", "name": "final_answer", "args": {"output": "B"}}])]
        )

        graph_a = build_react_graph(spec_a, registry, llm_client=llm_a)
        graph_b = build_react_graph(spec_b, registry, llm_client=llm_b)

        assert (await graph_a.ainvoke({"thread_id": "a", "input_text": "go"}))["output"] == "A"
        assert (await graph_b.ainvoke({"thread_id": "b", "input_text": "go"}))["output"] == "B"
        assert llm_a.requests[0].messages[0].content == "Prompt A"
        assert llm_b.requests[0].messages[0].content == "Prompt B"

    asyncio.run(run())


def test_context_optimizer_compacts_large_tool_outputs_before_llm_call():
    async def run():
        spec = AgentSpec(
            name="compact-agent",
            system_prompt="Use compact context.",
            tools={"enabled": ["lookup", "final_answer"], "compaction_overrides": {"lookup": "file_read"}},
            context={"token_threshold": 10_000, "tool_result_threshold": 20},
            runtime={"final_tool": "final_answer"},
        )
        registry = ToolRegistry([LookupTool(), FinalAnswerTool()])
        llm = ScriptedLlm(
            [LlmResponse(text="done", tool_calls=[{"id": "2", "name": "final_answer", "args": {"output": "ok"}}])]
        )
        graph = build_react_graph(spec, registry, llm_client=llm)
        state = await graph.ainvoke(
            {
                "thread_id": "t",
                "input_text": "continue",
                "messages": [
                    LlmMessage(role="user", content="continue"),
                    LlmMessage(
                        role="assistant",
                        content="calling lookup",
                        tool_calls=[{"id": "1", "name": "lookup", "args": {"query": "alpha"}}],
                    ),
                    LlmMessage(
                        role="tool",
                        name="lookup",
                        tool_call_id="1",
                        content="line\n" * 100,
                        artifact={"path": "large.txt"},
                    ),
                ],
            }
        )

        assert state["output"] == "ok"
        tool_messages = [message for message in llm.requests[0].messages if message.role == "tool"]
        assert "File content compacted" in tool_messages[0].content
        assert "1:final_answer" in state["raw_artifacts"]

    asyncio.run(run())


def test_policy_hooks_enforce_required_first_and_final_reflection():
    async def run():
        spec = AgentSpec(
            name="policy-agent",
            tools={"enabled": ["plan", "reflect", "final_answer"]},
            runtime={
                "max_steps": 5,
                "required_first_tool": "plan",
                "final_tool": "final_answer",
                "final_requires_reflection": True,
            },
        )
        registry = ToolRegistry([PlanTool(), ReflectTool(), FinalAnswerTool()])
        llm = ScriptedLlm(
            [
                LlmResponse(
                    text="bad final",
                    tool_calls=[{"id": "1", "name": "final_answer", "args": {"output": "too soon"}}],
                ),
                LlmResponse(
                    text="plan",
                    tool_calls=[
                        {
                            "id": "2",
                            "name": "plan",
                            "args": {"goal": "test", "tasks": [{"content": "plan", "status": "completed"}]},
                        }
                    ],
                ),
                LlmResponse(
                    text="reflect",
                    tool_calls=[
                        {
                            "id": "3",
                            "name": "reflect",
                            "args": {
                                "reflection_type": "self_check",
                                "analysis": "ready",
                                "ready_for_final": True,
                            },
                        }
                    ],
                ),
                LlmResponse(
                    text="final",
                    tool_calls=[{"id": "4", "name": "final_answer", "args": {"output": "done"}}],
                ),
            ]
        )
        graph = build_react_graph(spec, registry, llm_client=llm)
        state = await graph.ainvoke({"thread_id": "t", "input_text": "run"})

        assert state["output"] == "done"
        blocked_messages = [message.content for message in state["messages"] if getattr(message, "role", "") == "tool"]
        assert any("first model-controlled action" in content for content in blocked_messages)

    asyncio.run(run())


def test_llm_factory_registers_custom_provider():
    class CustomClient:
        def __init__(self, spec: ModelSpec) -> None:
            self.spec = spec

        async def complete(self, request: LlmRequest) -> LlmResponse:
            return LlmResponse(text=f"{self.spec.provider}:{self.spec.name}:{request.input_text}")

        async def stream(self, request: LlmRequest):
            yield "custom"

    async def run():
        factory = LlmClientFactory()
        factory.register("custom", CustomClient)
        factory.register("openai", CustomClient, replace=True)
        client = factory.create(ModelSpec(provider="custom", name="model-a"))
        openai_client = factory.create(ModelSpec(provider="openai", name="model-b"))

        response = await client.complete(LlmRequest(prompt="hello"))
        openai_response = await openai_client.complete(LlmRequest(prompt="hi"))

        assert response.text == "custom:model-a:hello"
        assert openai_response.text == "openai:model-b:hi"

    asyncio.run(run())


def test_graph_builder_builds_react_graph():
    async def run():
        spec = AgentSpec(
            name="builder-agent",
            tools={"enabled": ["final_answer"]},
            runtime={"final_tool": "final_answer"},
        )
        registry = ToolRegistry([FinalAnswerTool()])
        llm = ScriptedLlm(
            [LlmResponse(text="final", tool_calls=[{"id": "1", "name": "final_answer", "args": {"output": "ok"}}])]
        )

        graph = ReactGraphBuilder(spec, registry).with_llm_client(llm).build()
        state = await graph.ainvoke({"thread_id": "t", "input_text": "run"})

        assert state["output"] == "ok"

    asyncio.run(run())


def test_llm_context_summarizer_runs_when_context_stays_over_threshold():
    async def run():
        spec = AgentSpec(
            name="summary-agent",
            tools={"enabled": ["final_answer"]},
            context={
                "token_threshold": 1,
                "tool_result_threshold": 10_000,
                "summarization": {"keep_recent": 1, "strategy": "llm"},
            },
            runtime={"final_tool": "final_answer"},
        )
        registry = ToolRegistry([FinalAnswerTool()])
        llm = ScriptedLlm(
            [
                LlmResponse(text="summary from llm"),
                LlmResponse(
                    text="final",
                    tool_calls=[{"id": "1", "name": "final_answer", "args": {"output": "done"}}],
                ),
            ]
        )
        graph = build_react_graph(spec, registry, llm_client=llm)

        state = await graph.ainvoke(
            {
                "thread_id": "t",
                "input_text": "continue",
                "messages": [
                    LlmMessage(role="user", content="start"),
                    LlmMessage(role="assistant", content="step one"),
                    LlmMessage(role="user", content="step two"),
                    LlmMessage(role="assistant", content="latest"),
                ],
            }
        )

        assert state["output"] == "done"
        assert llm.requests[0].metadata["purpose"] == "context_summarization"
        assert any(
            message.role == "system" and "summary from llm" in message.content
            for message in llm.requests[1].messages
        )

    asyncio.run(run())


def test_platform_library_registers_custom_workflow():
    async def run():
        platform = AgentPlatformLibrary(session_manager=InMemorySessionManager())

        async def custom_workflow(text: str) -> str:
            return f"custom::{text}"

        platform.register_workflow("custom", custom_workflow)
        orchestrator = platform.build()
        response = await orchestrator.invoke("t-1", "hello", workflow_name="custom")

        assert response.output == "custom::hello"

    asyncio.run(run())


def test_platform_bootstrap_binds_session_backend_from_config():
    platform = AgentPlatformLibrary(spec=AgentSpec(persistence={"backend": "memory"})).bootstrap()

    assert isinstance(platform.session_manager, FrameworkInMemorySessionManager)
