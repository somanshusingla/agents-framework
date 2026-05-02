from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    args: dict[str, Any]
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
    output: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
