import asyncio

import pytest
from pydantic import BaseModel

from agent_framework.sdk import FunctionTool, ToolExecutionContext, ToolExecutionResult
from agent_framework.sdk.compaction import FileReadCompactionStrategy
from agent_framework.sdk.tools import BaseTool
from agent_framework.tools.registry import ToolRegistry


class EchoArgs(BaseModel):
    text: str


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text"
    args_schema = EchoArgs
    compaction_strategy = FileReadCompactionStrategy()

    async def execute(self, context: ToolExecutionContext, args: EchoArgs) -> ToolExecutionResult:
        return ToolExecutionResult(content=args.text, artifact={"text": args.text})


def test_base_tool_validates_and_executes():
    async def run():
        tool = EchoTool()
        result = await tool.invoke(
            ToolExecutionContext(thread_id="t", run_id="r", agent_name="a"),
            {"text": "hello"},
        )
        assert result.content == "hello"
        assert result.ok

    asyncio.run(run())


def test_base_tool_wraps_validation_errors():
    async def run():
        tool = EchoTool()
        result = await tool.invoke(
            ToolExecutionContext(thread_id="t", run_id="r", agent_name="a"),
            {},
        )
        assert result.status == "error"
        assert "Error executing echo" in result.content

    asyncio.run(run())


def test_function_tool_derives_schema_and_runs_sync_callable():
    async def run():
        tool = FunctionTool(name="add", description="Add", fn=lambda a, b: a + b)
        result = await tool.invoke(
            ToolExecutionContext(thread_id="t", run_id="r", agent_name="a"),
            {"a": 1, "b": 2},
        )
        assert result.content == "3"
        assert result.artifact == {"output": 3}

    asyncio.run(run())


def test_registry_lookup_definitions_duplicates_and_compaction_overrides():
    registry = ToolRegistry([EchoTool()])
    assert registry.get("echo") is not None
    assert registry.names() == ["echo"]
    assert registry.tool_definitions(["echo"])[0]["function"]["name"] == "echo"
    assert isinstance(registry.compaction_strategies(["echo"])["echo"], FileReadCompactionStrategy)
    assert registry.compaction_strategies(["echo"], {"echo": "truncate"})["echo"]

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())

    with pytest.raises(ValueError, match="Unknown enabled"):
        registry.tool_definitions(["missing"])
