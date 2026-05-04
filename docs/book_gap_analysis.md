# Book-Grounded Framework Gap Analysis

This framework now has the reusable ReAct graph, config-driven agent specs, class-based
tools, planning, reflection, guardrails, session persistence, and context compaction.
The following book-aligned layers remain incomplete or intentionally basic.

## Missing Or Shallow Layers

- **Provider LLM adapters**: the current default LLM is a deterministic local test client.
  Production adapters for OpenAI-compatible, Anthropic, and streaming tool-call deltas are
  still needed.
- **Human-in-the-loop approvals**: tools can declare `approval_required`, and the graph can
  return `waiting_approval`, but durable pause/resume APIs and approval decisions are not
  implemented end to end.
- **Durable LangGraph checkpointing**: session events persist, but LangGraph checkpointer
  integration and resumable graph state are still missing.
- **Long-term memory**: the framework has short-term state/session events, but no memory
  extraction, storage, retrieval tool, or cross-session memory policy.
- **Retrieval/RAG layer**: there is no knowledge-base abstraction, retriever tool, document
  chunking, or vector-store integration.
- **MCP tool adapter**: the SDK has class/function/agent tools, but no adapter for
  external MCP tool definitions and clients.
- **Workspace/code execution isolation**: the book's sandbox/workspace layer is not present.
  It should remain a separate secured extension rather than unrestricted local execution.
- **Multi-agent orchestration**: `AgentTool` exists, but transfer/handoff workflows,
  specialist agent routing, and isolated context per sub-agent are not implemented.
- **Evaluation layer**: no replay harness, benchmark runner, or LLM-as-judge evaluation
  hooks exist yet.
- **Observability depth**: tracing is a stub. Node/tool/model spans, token metrics,
  tool-frequency metrics, and guardrail metrics need real implementations.
- **Config hook loading**: config models include hook paths, but dynamic import and validation
  for bootstrap/tool-call policy hooks still need implementation.
- **Semantic summarization**: context summarization is deterministic and cheap today. A real
  LLM-backed summarizer should be added as an optional strategy after deterministic compaction.

## Planning And Reflection Status

- `plan` is the model-facing planning tool. It records the current task list in state and
  context.
- `reflect` is the model-facing reflection tool. It records progress reviews, error analysis,
  result synthesis, and final self-checks.
- `reflect.need_replan=true` updates state so configured policies/prompts can route the model
  back to `plan`.
