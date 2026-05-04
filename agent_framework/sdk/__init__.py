from agent_framework.sdk.compaction import (
    CodeSearchCompactionStrategy,
    DeploymentEventsCompactionStrategy,
    FileReadCompactionStrategy,
    FinalReportCompactionStrategy,
    RawJsonLogsCompactionStrategy,
    ReferenceCompactionStrategy,
    ToolCompactionStrategy,
    TruncateCompactionStrategy,
    create_compaction_strategy,
)
from agent_framework.sdk.guardrails import GuardrailPolicyProtocol
from agent_framework.sdk.llm import LlmClientProtocol
from agent_framework.sdk.sessions import SessionManagerProtocol
from agent_framework.sdk.tools import (
    AgentTool,
    BaseTool,
    FunctionTool,
    ToolExecutionContext,
    ToolExecutionResult,
)
from agent_framework.sdk.tracing import TracerProtocol

__all__ = [
    "AgentPlatformLibrary",
    "AgentTool",
    "BaseTool",
    "CodeSearchCompactionStrategy",
    "DeploymentEventsCompactionStrategy",
    "FileReadCompactionStrategy",
    "FinalReportCompactionStrategy",
    "FunctionTool",
    "GuardrailPolicyProtocol",
    "LlmClientProtocol",
    "RawJsonLogsCompactionStrategy",
    "ReferenceCompactionStrategy",
    "SessionManagerProtocol",
    "ToolCompactionStrategy",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "TracerProtocol",
    "TruncateCompactionStrategy",
    "create_compaction_strategy",
]


def __getattr__(name: str):
    if name == "AgentPlatformLibrary":
        from agent_framework.sdk.platform import AgentPlatformLibrary

        return AgentPlatformLibrary
    raise AttributeError(name)
