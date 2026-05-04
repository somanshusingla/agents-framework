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
from agent_framework.sdk.llm import (
    LlmClientFactory,
    LlmClientProtocol,
    create_llm_client,
    llm_provider,
    register_llm_provider,
)
from agent_framework.memory.session_manager import (
    InMemorySessionManager,
    NoOpSessionManager,
    SessionManager,
    create_session_manager,
    register_session_backend,
    session_backend,
)
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
    "InMemorySessionManager",
    "LlmClientFactory",
    "LlmClientProtocol",
    "NoOpSessionManager",
    "RawJsonLogsCompactionStrategy",
    "ReferenceCompactionStrategy",
    "SessionManager",
    "SessionManagerProtocol",
    "ToolCompactionStrategy",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "TracerProtocol",
    "TruncateCompactionStrategy",
    "create_compaction_strategy",
    "create_llm_client",
    "create_session_manager",
    "llm_provider",
    "register_llm_provider",
    "register_session_backend",
    "session_backend",
]


def __getattr__(name: str):
    if name == "AgentPlatformLibrary":
        from agent_framework.sdk.platform import AgentPlatformLibrary

        return AgentPlatformLibrary
    raise AttributeError(name)
