from __future__ import annotations

from agent_framework.sdk.compaction import ToolCompactionStrategy, create_compaction_strategy
from agent_framework.sdk.tools import BaseTool


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        if tools:
            self.register_many(tools)

    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[BaseTool], *, replace: bool = False) -> None:
        for tool in tools:
            self.register(tool, replace=replace)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def active_tools(self, enabled: list[str] | None = None) -> list[BaseTool]:
        if enabled is None:
            return list(self._tools.values())
        missing = sorted(set(enabled) - set(self._tools))
        if missing:
            raise ValueError(f"Unknown enabled tool(s): {', '.join(missing)}")
        return [self._tools[name] for name in enabled]

    def tool_definitions(self, enabled: list[str] | None = None) -> list[dict]:
        return [tool.tool_definition() for tool in self.active_tools(enabled)]

    def compaction_strategies(
        self,
        enabled: list[str] | None = None,
        overrides: dict[str, str] | None = None,
    ) -> dict[str, ToolCompactionStrategy]:
        strategies = {tool.name: tool.compaction_strategy for tool in self.active_tools(enabled)}
        for tool_name, strategy_name in (overrides or {}).items():
            if tool_name not in strategies:
                raise ValueError(f"Cannot override compaction for unknown tool: {tool_name}")
            strategies[tool_name] = create_compaction_strategy(strategy_name)
        return strategies

    # Backward-compatible alias for the early prototype.
    def specs(self) -> list[dict]:
        return self.tool_definitions()
