from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from agent_framework.llm.types import LlmMessage


class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    approval_required: bool = False


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    ok: bool
    output: dict | str | None = None
    error: str | None = None


class GuardrailResult(BaseModel):
    phase: Literal["input", "output"]
    allowed: bool
    action: Literal["allow", "block", "sanitize"] = "allow"
    reason: str | None = None
    transformed_text: str | None = None


class ExecutionEvent(BaseModel):
    id: str
    thread_id: str
    run_id: str
    seq_no: int
    ts: datetime
    event_type: str
    payload: dict[str, Any]


class WorkflowResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    output: str | dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    pending_approval: list[ToolCall] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    run_id: str
    thread_id: str
    workflow_name: str
    agent_name: str
    input_text: str
    metadata: dict[str, Any]
    messages: list[LlmMessage]
    working_messages: list[LlmMessage]
    pending_tool_calls: list[dict[str, Any]]
    pending_approval: list[ToolCall]
    raw_artifacts: dict[str, Any]
    tool_call_history: list[str]
    key_findings: list[str]
    plan: dict[str, Any] | None
    reflections: list[dict[str, Any]]
    token_count: int
    compaction_info: dict[str, Any] | None
    agent_step_count: int
    final_response: str | dict[str, Any] | None
    output: str | dict[str, Any] | None
    usage: dict[str, int]
    status: str
    error: str | None
    domain_state: dict[str, Any]
