from agent_framework.runtime.state import GuardrailResult


class AllowAllGuardrail:
    async def validate_input(self, text: str) -> GuardrailResult:
        return GuardrailResult(phase="input", allowed=True)

    async def validate_output(self, text: str) -> GuardrailResult:
        return GuardrailResult(phase="output", allowed=True)
