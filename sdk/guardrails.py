from __future__ import annotations

from typing import Protocol

from runtime.state import GuardrailResult


class GuardrailPolicyProtocol(Protocol):
    async def validate_input(self, text: str) -> GuardrailResult: ...
    async def validate_output(self, text: str) -> GuardrailResult: ...
