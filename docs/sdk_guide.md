# SDK Guide: Building Config-Driven ReAct Agents

The framework is now exposed as the `agent_framework` package. It provides one reusable
ReAct graph template that can be configured per agent with YAML, Python tool classes,
and optional policy/bootstrap hooks.

## Agent Config

```yaml
agent:
  name: support-agent
  workflow: react
  system_prompt: "You are a concise support agent."

tools:
  enabled:
    - plan
    - reflect
    - lookup_account
    - final_answer
  compaction_overrides:
    lookup_account: truncate

runtime:
  max_steps: 8
  required_first_tool: plan
  final_tool: final_answer

context:
  strategy: hierarchical
  token_threshold: 50000
  keep_recent: 10
  tool_result_threshold: 1000
```

Load it with:

```python
from agent_framework import AgentPlatformLibrary

platform = AgentPlatformLibrary.from_config("agent.yaml")
```

## Tool Classes

Tools are regular Python classes. Config chooses which registered tools are enabled.

```python
from pydantic import BaseModel
from agent_framework import BaseTool, ToolExecutionContext, ToolExecutionResult

class LookupArgs(BaseModel):
    account_id: str

class LookupAccountTool(BaseTool):
    name = "lookup_account"
    description = "Look up account details by account id."
    args_schema = LookupArgs

    async def execute(self, context: ToolExecutionContext, args: LookupArgs):
        return ToolExecutionResult(
            content=f"Account {args.account_id} is active.",
            artifact={"account_id": args.account_id},
        )
```

Register and run:

```python
platform.register_tool(LookupAccountTool())
orchestrator = platform.build()
response = await orchestrator.invoke("thread-1", "Check account A123")
```

## Reusable Graph

All configured agents use the same `build_react_graph(...)` template unless a host
application explicitly registers a custom workflow. Domain behavior belongs in:

- system prompts
- tool implementations
- YAML runtime/context settings
- optional bootstrap hooks
- optional tool-call policy hooks

The graph itself stays generic: initialize, validate, compact context, call the LLM,
execute tools, repeat, finalize, validate output, and persist events.

## Context Compaction

The default optimizer keeps raw history in state/session and sends a compact projection
to the LLM. It:

1. does nothing below the configured threshold,
2. compacts deterministic tool calls/results first,
3. preserves the original task and recent messages,
4. summarizes older intermediate work only when still over threshold.

Each tool can provide a default compaction strategy, and YAML can override strategies
by tool name.
