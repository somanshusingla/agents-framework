# SDK Guide: Building Agent Applications with `book-grounded-langgraph-agent`

This project can now be consumed as a Python library. You compose the runtime by registering tools/workflows and optionally injecting your own implementations for LLM, guardrails, sessions, and tracing.

## SDK Components

- `sdk.tools`: Tool contracts and tool wrappers (`BaseTool`, `FunctionTool`, `AgentTool`).
- `sdk.llm`: LLM protocol (`LlmClientProtocol`).
- `sdk.guardrails`: Guardrail policy protocol (`GuardrailPolicyProtocol`).
- `sdk.sessions`: Session persistence protocol (`SessionManagerProtocol`).
- `sdk.tracing`: Tracing protocol (`TracerProtocol`).
- `sdk.platform`: Library composition root (`AgentPlatformLibrary`).

## Quick Start

```python
from sdk import AgentPlatformLibrary, FunctionTool

platform = AgentPlatformLibrary()
platform.register_tool(
    FunctionTool(
        name="weather_lookup",
        description="Look up weather by city",
        fn=lambda city: {"city": city, "forecast": "sunny"},
    )
)
orchestrator = platform.build()

# await orchestrator.invoke(thread_id="t-1", text="tool:weather_lookup {\"city\":\"Austin\"}")
```

## Registering Custom Workflows

```python
async def summarize_workflow(text: str) -> str:
    return f"Summary: {text[:100]}"

platform.register_workflow("summarize", summarize_workflow)
orchestrator = platform.build()
# await orchestrator.invoke("thread-1", "long input", workflow_name="summarize")
```

## Injecting Custom Component Implementations

### Custom LLM
Implement `complete` and `stream` methods compatible with `LlmClientProtocol`.

### Custom Guardrails
Implement `validate_input` and `validate_output` returning `GuardrailResult`.

### Custom Session Store
Implement `load_events` and `append_event` methods compatible with `SessionManagerProtocol`.

### Custom Tracing
Implement `span(name, attrs)` context manager compatible with `TracerProtocol`.

## API Adapter vs Library Use

- Use `api/app.py` if you want ready-made REST/SSE endpoints.
- Use `sdk/platform.py` directly when embedding in your own app/service.

## Production Integration Notes

1. Register business tools in platform startup, not per-request.
2. Use deterministic tool names and version them if behavior changes.
3. Inject enterprise guardrails and tracing for compliance.
4. Persist session events to durable storage for replay/debugging.
5. Use custom workflows for domain-specific orchestration.
