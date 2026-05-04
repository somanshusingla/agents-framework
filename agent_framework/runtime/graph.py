from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from agent_framework.config import AgentSpec
from agent_framework.guardrails.policies import AllowAllGuardrail
from agent_framework.llm.base import LlmClient
from agent_framework.llm.factory import create_llm_client
from agent_framework.llm.types import LlmMessage, LlmRequest
from agent_framework.observability.langsmith import LangSmithTracer
from agent_framework.runtime.context_optimizer import (
    ContextOptimizer,
    DeterministicContextSummarizer,
    LlmContextSummarizer,
    count_messages,
)
from agent_framework.runtime.policies import ConfigurableToolCallPolicy, ToolCallPolicy
from agent_framework.runtime.state import AgentState, ExecutionEvent, ToolCall
from agent_framework.sdk.tools import ToolExecutionContext, ToolExecutionResult
from agent_framework.tools.registry import ToolRegistry

BootstrapHook = Callable[[AgentState], dict[str, Any] | Awaitable[dict[str, Any]]]


class ReactGraphBuilder:
    def __init__(self, spec: AgentSpec, registry: ToolRegistry) -> None:
        self.spec = spec
        self.registry = registry
        self.llm_client: Any | None = None
        self.session_manager: Any | None = None
        self.guardrails: Any | None = None
        self.tracer: Any | None = None
        self.context_optimizer: ContextOptimizer | None = None
        self.bootstrap_hook: BootstrapHook | None = None
        self.tool_call_policy: ToolCallPolicy | None = None

    def with_llm_client(self, llm_client: Any | None) -> "ReactGraphBuilder":
        self.llm_client = llm_client
        return self

    def with_session_manager(self, session_manager: Any | None) -> "ReactGraphBuilder":
        self.session_manager = session_manager
        return self

    def with_guardrails(self, guardrails: Any | None) -> "ReactGraphBuilder":
        self.guardrails = guardrails
        return self

    def with_tracer(self, tracer: Any | None) -> "ReactGraphBuilder":
        self.tracer = tracer
        return self

    def with_context_optimizer(self, context_optimizer: ContextOptimizer | None) -> "ReactGraphBuilder":
        self.context_optimizer = context_optimizer
        return self

    def with_bootstrap_hook(self, bootstrap_hook: BootstrapHook | None) -> "ReactGraphBuilder":
        self.bootstrap_hook = bootstrap_hook
        return self

    def with_tool_call_policy(self, tool_call_policy: ToolCallPolicy | None) -> "ReactGraphBuilder":
        self.tool_call_policy = tool_call_policy
        return self

    def build(self) -> Any:
        return _compile_react_graph(
            self.spec,
            self.registry,
            llm_client=self.llm_client,
            session_manager=self.session_manager,
            guardrails=self.guardrails,
            tracer=self.tracer,
            context_optimizer=self.context_optimizer,
            bootstrap_hook=self.bootstrap_hook,
            tool_call_policy=self.tool_call_policy,
        )


def build_react_graph(
    spec: AgentSpec,
    registry: ToolRegistry,
    llm_client: Any | None = None,
    session_manager: Any | None = None,
    guardrails: Any | None = None,
    tracer: Any | None = None,
    *,
    context_optimizer: ContextOptimizer | None = None,
    bootstrap_hook: BootstrapHook | None = None,
    tool_call_policy: ToolCallPolicy | None = None,
) -> Any:
    return (
        ReactGraphBuilder(spec, registry)
        .with_llm_client(llm_client)
        .with_session_manager(session_manager)
        .with_guardrails(guardrails)
        .with_tracer(tracer)
        .with_context_optimizer(context_optimizer)
        .with_bootstrap_hook(bootstrap_hook)
        .with_tool_call_policy(tool_call_policy)
        .build()
    )


def _compile_react_graph(
    spec: AgentSpec,
    registry: ToolRegistry,
    llm_client: Any | None = None,
    session_manager: Any | None = None,
    guardrails: Any | None = None,
    tracer: Any | None = None,
    *,
    context_optimizer: ContextOptimizer | None = None,
    bootstrap_hook: BootstrapHook | None = None,
    tool_call_policy: ToolCallPolicy | None = None,
) -> Any:
    """Build the reusable ReAct graph for one configured agent."""

    llm = llm_client or LlmClient(spec.model)
    trace = tracer or LangSmithTracer()
    guardrail_policy = guardrails or AllowAllGuardrail()
    optimizer = context_optimizer or ContextOptimizer(
        spec.context,
        registry.compaction_strategies(
            spec.tools.enabled or None,
            overrides=spec.tools.compaction_overrides,
        ),
        summarizer=_build_context_summarizer(spec, llm),
    )
    policy = tool_call_policy or ConfigurableToolCallPolicy(spec.runtime)
    enabled_tools = spec.tools.enabled or None

    async def initialize_run(state: AgentState) -> dict[str, Any]:
        run_id = state.get("run_id") or str(uuid.uuid4())
        thread_id = state.get("thread_id") or str(uuid.uuid4())
        input_text = str(state.get("input_text") or "").strip()
        messages = [_coerce_message(message) for message in list(state.get("messages") or [])]
        if input_text and not messages:
            messages = [LlmMessage(role="user", content=input_text)]
        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "workflow_name": spec.workflow,
            "agent_name": spec.name,
            "input_text": input_text,
            "metadata": dict(state.get("metadata") or {}),
            "messages": messages,
            "raw_artifacts": dict(state.get("raw_artifacts") or {}),
            "tool_call_history": list(state.get("tool_call_history") or []),
            "key_findings": list(state.get("key_findings") or []),
            "reflections": list(state.get("reflections") or []),
            "agent_step_count": int(state.get("agent_step_count") or 0),
            "usage": dict(state.get("usage") or {}),
            "domain_state": dict(state.get("domain_state") or {}),
            "status": "running",
            "error": None,
        }

    async def bootstrap(state: AgentState) -> dict[str, Any]:
        if bootstrap_hook is None or state.get("error"):
            return {}
        return await _maybe_await(bootstrap_hook(state))

    async def validate_input(state: AgentState) -> dict[str, Any]:
        result = await guardrail_policy.validate_input(state.get("input_text", ""))
        if not result.allowed:
            return {"status": "blocked", "error": result.reason or "Input blocked by guardrail"}
        if result.transformed_text:
            messages = list(state.get("messages") or [])
            if messages and messages[0].role == "user":
                messages[0] = LlmMessage(role="user", content=result.transformed_text)
            return {"input_text": result.transformed_text, "messages": messages}
        return {}

    async def context_guard(state: AgentState) -> dict[str, Any]:
        projection = await optimizer.project_async(list(state.get("messages") or []), state=state)
        return {
            "working_messages": projection.messages,
            "token_count": count_messages(projection.messages),
            "compaction_info": projection.info,
        }

    async def model_call(state: AgentState) -> dict[str, Any]:
        if state.get("error"):
            return {}
        step_count = int(state.get("agent_step_count") or 0)
        if step_count >= spec.runtime.max_steps:
            return {
                "status": "failed",
                "error": f"Max agent steps reached: {spec.runtime.max_steps}",
            }

        messages = _model_messages(spec, state)
        request = LlmRequest(
            messages=messages,
            tools=registry.tool_definitions(enabled_tools),
            tool_choice="auto" if registry.active_tools(enabled_tools) else None,
            metadata={
                "agent_name": spec.name,
                "thread_id": state.get("thread_id"),
                "run_id": state.get("run_id"),
            },
        )
        with trace.span(
            "llm.complete",
            {
                "run_type": "llm",
                "agent_name": spec.name,
                "provider": spec.model.provider,
                "model": spec.model.name,
                "inputs": {
                    "message_count": len(messages),
                    "tool_count": len(request.tools),
                    "thread_id": state.get("thread_id"),
                    "run_id": state.get("run_id"),
                },
            },
        ) as span:
            response = await llm.complete(request)
            _span_outputs(
                span,
                {
                    "usage": response.usage,
                    "tool_call_count": len(response.tool_calls),
                    "text_preview": response.text[:500],
                },
            )
        assistant_message = response.as_message()
        pending = [_normalize_tool_call(call) for call in response.tool_calls]
        usage = dict(state.get("usage") or {})
        for key, value in (response.usage or {}).items():
            usage[key] = int(usage.get(key, 0)) + int(value)
        return {
            "messages": [*list(state.get("messages") or []), assistant_message],
            "pending_tool_calls": pending,
            "usage": usage,
            "agent_step_count": step_count + 1,
        }

    async def check_tool_approval(state: AgentState) -> dict[str, Any]:
        pending = list(state.get("pending_tool_calls") or [])
        approval_needed: list[ToolCall] = []
        for call in pending:
            tool = registry.get(call["name"])
            if tool and tool.approval_required:
                approval_needed.append(
                    ToolCall(
                        call_id=call["id"],
                        tool_name=call["name"],
                        args=call.get("args", {}),
                        approval_required=True,
                    )
                )
        if approval_needed:
            return {"pending_approval": approval_needed, "status": "waiting_approval"}
        return {}

    async def execute_tools(state: AgentState) -> dict[str, Any]:
        if state.get("status") == "waiting_approval":
            return {}

        working_state = dict(state)
        raw_artifacts = dict(state.get("raw_artifacts") or {})
        history = list(state.get("tool_call_history") or [])
        messages = list(state.get("messages") or [])
        combined_updates: dict[str, Any] = {}
        tool_messages: list[LlmMessage] = []

        for call in list(state.get("pending_tool_calls") or []):
            name = str(call.get("name") or "")
            args = dict(call.get("args") or {})
            call_id = str(call.get("id") or f"call-{len(tool_messages) + 1}")
            context = ToolExecutionContext(
                thread_id=str(state.get("thread_id")),
                run_id=str(state.get("run_id")),
                agent_name=spec.name,
                state=working_state,
                metadata=dict(state.get("metadata") or {}),
            )

            with trace.span(
                f"tool.{name}",
                {
                    "run_type": "tool",
                    "agent_name": spec.name,
                    "tool_name": name,
                    "thread_id": state.get("thread_id"),
                    "run_id": state.get("run_id"),
                    "inputs": {"args": args},
                },
            ) as span:
                blocked = await policy.check(context, name, args)
                tool = registry.get(name)
                if blocked is not None:
                    result = blocked
                elif tool is None:
                    result = ToolExecutionResult(content=f"Tool {name} is not registered", status="error")
                else:
                    result = await tool.invoke(context, args)
                _span_outputs(
                    span,
                    {
                        "status": result.status,
                        "content_preview": result.content[:500],
                        "artifact_keys": sorted((result.artifact or {}).keys()),
                    },
                )

            artifact = result.artifact or {}
            artifact.setdefault("args", args)
            if tool and tool.compaction_strategy.preserve_artifact(result.content, artifact):
                raw_key = f"{len(raw_artifacts) + 1}:{name}"
                raw_artifacts[raw_key] = {"content": result.content, "artifact": artifact}

            if result.state_updates:
                working_state.update(result.state_updates)
                combined_updates.update(result.state_updates)

            if result.status == "success":
                history.append(name)

            tool_messages.append(
                LlmMessage(
                    role="tool",
                    content=result.content,
                    name=name,
                    tool_call_id=call_id,
                    artifact=artifact,
                    status=result.status,
                )
            )

            if spec.runtime.final_tool and name == spec.runtime.final_tool and result.status == "success":
                combined_updates["final_response"] = _final_output_from_result(result)

        combined_updates.update(
            {
                "messages": [*messages, *tool_messages],
                "pending_tool_calls": [],
                "raw_artifacts": raw_artifacts,
                "tool_call_history": history,
            }
        )
        return combined_updates

    async def finalize(state: AgentState) -> dict[str, Any]:
        if state.get("output") is not None:
            return {}
        if state.get("final_response") is not None:
            return {"output": state["final_response"], "status": "completed"}
        if state.get("error"):
            return {"output": state["error"], "status": state.get("status") or "failed"}
        for message in reversed(list(state.get("messages") or [])):
            if message.role == "assistant" and not message.tool_calls:
                return {"output": message.content, "status": "completed"}
        return {"output": None, "status": state.get("status") or "completed"}

    async def validate_output(state: AgentState) -> dict[str, Any]:
        if state.get("output") is None:
            return {}
        result = await guardrail_policy.validate_output(str(state.get("output")))
        if result.action == "sanitize" and result.transformed_text is not None:
            return {"output": result.transformed_text}
        if not result.allowed:
            return {"status": "blocked", "output": result.reason or "Output blocked by guardrail"}
        return {}

    async def persist_session(state: AgentState) -> dict[str, Any]:
        if session_manager is None:
            return {}
        existing_events = await session_manager.load_events(str(state["thread_id"]))
        event = ExecutionEvent(
            id=str(uuid.uuid4()),
            thread_id=str(state["thread_id"]),
            run_id=str(state["run_id"]),
            seq_no=len(existing_events) + 1,
            ts=datetime.now(timezone.utc),
            event_type="final_output",
            payload={
                "agent_name": spec.name,
                "status": state.get("status"),
                "output": state.get("output"),
                "usage": state.get("usage", {}),
            },
        )
        await session_manager.append_event(event, workflow_name=spec.workflow, agent_name=spec.name)
        return {}

    def route_after_input(state: AgentState) -> str:
        return "finalize" if state.get("error") else "context_guard"

    def route_after_model(state: AgentState) -> str:
        if state.get("error") or state.get("final_response") is not None:
            return "finalize"
        return "approval" if state.get("pending_tool_calls") else "finalize"

    def route_after_approval(state: AgentState) -> str:
        return "finalize" if state.get("status") == "waiting_approval" else "tools"

    def route_after_tools(state: AgentState) -> str:
        if state.get("error") or state.get("final_response") is not None:
            return "finalize"
        return "context_guard"

    graph = StateGraph(AgentState)
    graph.add_node("initialize_run", _traced_node(trace, spec, "initialize_run", initialize_run))
    graph.add_node("bootstrap", _traced_node(trace, spec, "bootstrap", bootstrap))
    graph.add_node("validate_input", _traced_node(trace, spec, "validate_input", validate_input))
    graph.add_node("context_guard", _traced_node(trace, spec, "context_guard", context_guard))
    graph.add_node("model_call", _traced_node(trace, spec, "model_call", model_call))
    graph.add_node("check_tool_approval", _traced_node(trace, spec, "check_tool_approval", check_tool_approval))
    graph.add_node("execute_tools", _traced_node(trace, spec, "execute_tools", execute_tools))
    graph.add_node("finalize", _traced_node(trace, spec, "finalize", finalize))
    graph.add_node("validate_output", _traced_node(trace, spec, "validate_output", validate_output))
    graph.add_node("persist_session", _traced_node(trace, spec, "persist_session", persist_session))

    graph.add_edge(START, "initialize_run")
    graph.add_edge("initialize_run", "bootstrap")
    graph.add_edge("bootstrap", "validate_input")
    graph.add_conditional_edges(
        "validate_input",
        route_after_input,
        {"context_guard": "context_guard", "finalize": "finalize"},
    )
    graph.add_edge("context_guard", "model_call")
    graph.add_conditional_edges(
        "model_call",
        route_after_model,
        {"approval": "check_tool_approval", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "check_tool_approval",
        route_after_approval,
        {"tools": "execute_tools", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"context_guard": "context_guard", "finalize": "finalize"},
    )
    graph.add_edge("finalize", "validate_output")
    graph.add_edge("validate_output", "persist_session")
    graph.add_edge("persist_session", END)
    return graph.compile()


def _build_context_summarizer(spec: AgentSpec, llm: Any) -> Any:
    strategy = spec.context.summarization.strategy
    if not spec.context.summarization.enabled or strategy in {"none", "deterministic"}:
        return DeterministicContextSummarizer()

    summary_provider = spec.context.summarization.provider
    summary_model = spec.context.summarization.model
    if summary_provider or summary_model:
        summary_spec = spec.model.model_copy(
            update={
                "provider": summary_provider or spec.model.provider,
                "name": summary_model or spec.model.name,
                "max_tokens": spec.context.summarization.max_summary_tokens,
            }
        )
        return LlmContextSummarizer(create_llm_client(summary_spec))

    return LlmContextSummarizer(llm)


def _traced_node(
    tracer: Any,
    spec: AgentSpec,
    name: str,
    func: Callable[[AgentState], Awaitable[dict[str, Any]]],
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def wrapped(state: AgentState) -> dict[str, Any]:
        with tracer.span(
            f"node.{name}",
            {
                "run_type": "chain",
                "agent_name": spec.name,
                "workflow": spec.workflow,
                "thread_id": state.get("thread_id"),
                "run_id": state.get("run_id"),
                "inputs": _state_trace_inputs(state),
            },
        ) as span:
            result = await func(state)
            _span_outputs(span, _update_trace_outputs(result))
            return result

    return wrapped


def _state_trace_inputs(state: AgentState) -> dict[str, Any]:
    return {
        "status": state.get("status"),
        "message_count": len(list(state.get("messages") or [])),
        "pending_tool_call_count": len(list(state.get("pending_tool_calls") or [])),
        "agent_step_count": state.get("agent_step_count"),
    }


def _update_trace_outputs(update: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": update.get("status"),
        "keys": sorted(update.keys()),
        "message_count": len(list(update.get("messages") or [])) if "messages" in update else None,
        "pending_tool_call_count": len(list(update.get("pending_tool_calls") or []))
        if "pending_tool_calls" in update
        else None,
        "usage": update.get("usage"),
    }


def _span_outputs(span: Any, outputs: dict[str, Any]) -> None:
    cleaned = {key: value for key, value in outputs.items() if value is not None}
    if hasattr(span, "add_outputs"):
        span.add_outputs(cleaned)


def _model_messages(spec: AgentSpec, state: AgentState) -> list[LlmMessage]:
    messages = [LlmMessage(role="system", content=spec.system_prompt)] if spec.system_prompt else []
    messages.extend(list(state.get("working_messages") or state.get("messages") or []))
    return messages


def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    name = call.get("name") or call.get("tool_name") or ""
    call_id = call.get("id") or call.get("call_id") or f"call-{uuid.uuid4()}"
    return {"id": str(call_id), "name": str(name), "args": dict(call.get("args") or {})}


def _coerce_message(message: Any) -> LlmMessage:
    if isinstance(message, LlmMessage):
        return message
    if isinstance(message, dict):
        return LlmMessage(
            role=message.get("role", "user"),
            content=str(message.get("content") or ""),
            name=message.get("name"),
            tool_call_id=message.get("tool_call_id"),
            tool_calls=list(message.get("tool_calls") or []),
            artifact=message.get("artifact"),
            status=message.get("status"),
        )
    return LlmMessage(role="user", content=str(message))


def _final_output_from_result(result: ToolExecutionResult) -> str | dict[str, Any]:
    artifact = result.artifact or {}
    if "output" in artifact:
        return artifact["output"]
    if "report" in artifact:
        return artifact["report"]
    return result.content


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
