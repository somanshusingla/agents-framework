import asyncio

from llm.base import LlmClient
from llm.types import LlmRequest
from sdk.tools import FunctionTool


def test_llm_plain_echo():
    async def run():
        c = LlmClient()
        r = await c.complete(LlmRequest(prompt="hello world"))
        assert r.text.startswith("Echo:")
    asyncio.run(run())


def test_function_tool_exec():
    async def run():
        t = FunctionTool(name="x", description="x", fn=lambda a, b: a + b)
        out = await t.execute({"a": 1, "b": 2})
        assert out == 3
    asyncio.run(run())
