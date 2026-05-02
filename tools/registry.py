from sdk.tools import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "approval_required": t.approval_required} for t in self._tools.values()]
