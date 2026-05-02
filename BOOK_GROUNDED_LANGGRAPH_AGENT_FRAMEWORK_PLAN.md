# Book-Grounded LangGraph Agent Framework Plan

## Summary

Use `C:\Users\Somanshu\Downloads\Build_an_AI_Agent_(From_Scratch)_v4_MEAP.pdf` plus the earlier AI platform chapters to design the framework. Execution is gated:

1. Build `docs/hld.md` first.
2. After HLD approval, build `docs/lld.md`.
3. After LLD approval, implement the framework.

The repo is currently empty. LangGraph and FastAPI are not installed locally yet, so implementation will define dependencies in `pyproject.toml` rather than relying on the current environment.

## Architecture Direction

- Use LangGraph as the runtime core.
- Map the book's `ExecutionContext` to LangGraph graph state plus an immutable event log.
- Preserve the book's separation between storage and presentation:
  - graph state/session store keeps full raw history
  - `LlmRequestBuilder` decides what subset reaches the model
- Implement the book's ReAct loop as a LangGraph state machine:
  - initialize context
  - load session
  - validate input
  - build model request
  - think/model call
  - route final answer vs tool calls
  - approve sensitive tools when needed
  - execute tools
  - optimize context
  - validate output
  - persist session
  - emit observability
- Use LangGraph checkpointers for durable execution, pause/resume, and human-in-the-loop.
- Use `thread_id` as the durable execution/session identity.
- Use streaming modes for progress events and token/custom output.
- Keep side effects isolated in graph nodes/tools with idempotency keys.

## HLD Deliverable

Create `docs/hld.md` with Mermaid diagrams.

HLD must include:

- System goals, non-goals, v1 boundaries.
- Agent component model from the book:
  - `ExecutionContext`
  - `BaseTool`, `FunctionTool`, MCP tool, `AgentTool`
  - `LlmRequest`, `LlmResponse`, `LlmClient`
  - ReAct agent loop
  - context optimization
  - session and memory
  - human-in-the-loop
  - planning and reflection
  - workspace/code execution as a secured extension
  - multi-agent workflow, agent-as-tool, and transfer patterns
  - observability and evaluation extension points
- Platform mapping from the earlier chapters:
  - SDK/runtime
  - workflow API gateway
  - model abstraction
  - session service
  - tool service
  - guardrails
  - observability
- Mermaid diagrams:
  - platform context
  - component architecture
  - LangGraph ReAct state graph
  - sync request sequence
  - streaming request sequence
  - human approval pause/resume flow
  - Docker Compose deployment shape

## LLD Deliverable

Create `docs/lld.md` after HLD approval.

LLD must define:

- Python package structure under `agent_framework/`.
- SDK interfaces:
  - `AgentPlatform`
  - `Agent`
  - `@workflow`
  - `BaseTool`
  - `FunctionTool`
  - `AgentTool`
  - `LlmClient`
  - `SessionManager`
  - `GuardrailPolicy`
- LangGraph state schemas:
  - `AgentState`
  - `ExecutionEvent`
  - `ToolCall`
  - `ToolResult`
  - `GuardrailResult`
  - `TraceEvent`
  - `WorkflowJob`
- Graph nodes and conditional edges.
- API contracts:
  - sync workflow invocation
  - async job polling
  - SSE streaming
  - health and metrics endpoints
- SQLite persistence:
  - sessions
  - messages/events
  - checkpoints
  - jobs
  - traces
  - tool calls
  - guardrail events
- Mermaid diagrams:
  - package/module diagram
  - detailed graph node diagram
  - persistence ER diagram
  - tool execution sequence
  - approval/resume sequence
  - multi-agent orchestration diagram

## Implementation Scope

V1 implements:

- LangGraph-backed single-agent runtime.
- ReAct loop.
- Session memory.
- Context optimization with sliding window and compaction; summarization as an interface with one basic implementation.
- Tool registry with function tools and approval metadata.
- Guardrails before input and after output.
- Planning and reflection as built-in tools.
- FastAPI gateway for sync, async, and streaming workflows.
- SQLite persistence and LangGraph checkpointer.
- Structured traces, logs, and metrics.
- Docker Compose local runtime.
- Example agents.

V1 defers:

- Full RAG/data indexing service.
- Full evaluation service with LLM-as-judge.
- Networked A2A protocol.
- Arbitrary code execution sandbox, except as a documented extension point.

## Test And Acceptance

- HLD accepted when architecture diagrams and book-to-framework mapping are complete.
- LLD accepted when implementation decisions are fully specified.
- Implementation accepted when:
  - examples run locally
  - LangGraph graph execution works
  - sync, async, and SSE APIs work
  - sessions persist across turns
  - tool calls execute and record results
  - approval-required tools pause and resume
  - guardrails block/sanitize correctly
  - traces show node timings and model/tool usage
  - Docker Compose starts the runtime

## References

- Local book: `C:\Users\Somanshu\Downloads\Build_an_AI_Agent_(From_Scratch)_v4_MEAP.pdf`
- LangGraph persistence: [docs.langchain.com/oss/python/langgraph/persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- LangGraph durable execution: [docs.langchain.com/oss/python/langgraph/durable-execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- LangGraph streaming: [docs.langchain.com/oss/python/langgraph/streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- LangGraph design approach: [docs.langchain.com/oss/python/langgraph/thinking-in-langgraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
