import asyncio

from runtime.state import ExecutionEvent
from sdk import AgentPlatformLibrary, FunctionTool


class InMemorySessionManager:
    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []

    async def load_events(self, thread_id: str) -> list[ExecutionEvent]:
        return [event for event in self.events if event.thread_id == thread_id]

    async def append_event(self, event: ExecutionEvent, workflow_name: str = "default", agent_name: str = "default") -> None:
        self.events.append(event)


def test_library_registers_tools_into_orchestrator():
    platform = AgentPlatformLibrary(session_manager=InMemorySessionManager())
    platform.register_tool(FunctionTool(name="mul", description="Multiply", fn=lambda a, b: {"product": a * b}))

    orchestrator = platform.build()

    assert orchestrator.registry.get("mul") is not None


def test_library_registers_custom_workflow():
    async def run():
        platform = AgentPlatformLibrary(session_manager=InMemorySessionManager())

        async def custom_workflow(text: str) -> str:
            return f"custom::{text}"

        platform.register_workflow("custom", custom_workflow)
        orchestrator = platform.build()
        response = await orchestrator.invoke("t-1", "hello", workflow_name="custom")

        assert response.output == "custom::hello"

    asyncio.run(run())
