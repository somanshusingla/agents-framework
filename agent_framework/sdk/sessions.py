from __future__ import annotations

from typing import Protocol

from agent_framework.runtime.state import ExecutionEvent


class SessionManagerProtocol(Protocol):
    async def load_events(self, thread_id: str) -> list[ExecutionEvent]: ...
    async def append_event(self, event: ExecutionEvent, workflow_name: str = "default", agent_name: str = "default") -> None: ...
