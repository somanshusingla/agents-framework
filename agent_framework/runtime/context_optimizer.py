from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework.config import ContextSpec
from agent_framework.llm.types import LlmMessage
from agent_framework.sdk.compaction import ToolCompactionStrategy, create_compaction_strategy


@dataclass(slots=True)
class ContextProjection:
    messages: list[LlmMessage]
    info: dict[str, Any] | None


class ContextOptimizer:
    """Build a compact model-facing projection without mutating raw session state."""

    def __init__(
        self,
        context_spec: ContextSpec | None = None,
        strategies: dict[str, ToolCompactionStrategy] | None = None,
    ) -> None:
        self._spec = context_spec or ContextSpec()
        self._strategies = strategies or {}
        self._default_strategy = create_compaction_strategy(self._spec.default_tool_strategy)

    def project(
        self,
        messages: list[LlmMessage],
        *,
        state: dict[str, Any] | None = None,
        force: bool = False,
    ) -> ContextProjection:
        if self._spec.strategy == "none":
            return ContextProjection(list(messages), None)

        before = count_messages(messages)
        oversized_tool_result = any(
            message.role == "tool" and len(message.content or "") > self._spec.tool_result_threshold
            for message in messages
        )
        if not force and before < self._spec.token_threshold and not oversized_tool_result:
            return ContextProjection(list(messages), None)

        compacted, compacted_count = self._compact_tool_messages(messages)
        after_compaction = count_messages(compacted)
        summary_added = False

        if (
            self._spec.summarization.enabled
            and after_compaction >= self._spec.token_threshold
            and len(compacted) > self._spec.summarization.keep_recent + 2
        ):
            compacted = self._summarize_old_messages(compacted, state or {})
            summary_added = True

        after = count_messages(compacted)
        return ContextProjection(
            compacted,
            {
                "before_tokens": before,
                "after_tokens": after,
                "before_msgs": len(messages),
                "after_msgs": len(compacted),
                "compacted_tool_messages": compacted_count,
                "summary_added": summary_added,
            },
        )

    def _compact_tool_messages(self, messages: list[LlmMessage]) -> tuple[list[LlmMessage], int]:
        tool_args_by_id: dict[str, dict[str, Any]] = {}
        projected: list[LlmMessage] = []
        recent_start = max(len(messages) - self._spec.keep_recent, 0)
        compacted_count = 0
        global_compaction = count_messages(messages) >= self._spec.token_threshold

        for index, message in enumerate(messages):
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    call_id = str(tool_call.get("id") or tool_call.get("call_id") or "")
                    if call_id:
                        tool_args_by_id[call_id] = dict(tool_call.get("args") or {})
                projected.append(message)
                continue

            if message.role != "tool":
                projected.append(message)
                continue

            content_too_large = len(message.content or "") > self._spec.tool_result_threshold
            preserve_recent_small = index >= recent_start and not content_too_large
            if not content_too_large and (not global_compaction or preserve_recent_small):
                projected.append(message)
                continue

            strategy = self._strategies.get(message.name or "", self._default_strategy)
            args = tool_args_by_id.get(str(message.tool_call_id), {})
            artifact = message.artifact or {}
            compacted_count += 1
            projected.append(
                LlmMessage(
                    role="tool",
                    content=strategy.compact_tool_result(message.content, args, artifact),
                    name=message.name,
                    tool_call_id=message.tool_call_id,
                    artifact=artifact,
                    status=message.status,
                )
            )

        return projected, compacted_count

    def _summarize_old_messages(
        self,
        messages: list[LlmMessage],
        state: dict[str, Any],
    ) -> list[LlmMessage]:
        first_user_idx = next(
            (index for index, message in enumerate(messages) if message.role == "user"),
            0,
        )
        keep_recent = self._spec.summarization.keep_recent
        summary_end = max(len(messages) - keep_recent, first_user_idx + 1)
        summary_start = int(state.get("last_summary_index", first_user_idx))
        if summary_end <= summary_start + 1:
            return messages

        preserved_start = messages[: summary_start + 1]
        preserved_end = messages[summary_end:]
        to_summarize = messages[summary_start + 1 : summary_end]
        summary = _format_summary(to_summarize)
        state["last_summary_index"] = len(preserved_start)
        return [
            *preserved_start,
            LlmMessage(role="system", content=f"[Previous work summary]\n{summary}"),
            *preserved_end,
        ]


def count_messages(messages: list[LlmMessage]) -> int:
    return sum(count_message(message) for message in messages)


def count_message(message: LlmMessage) -> int:
    tokens = _rough_token_count(message.content)
    if message.name:
        tokens += _rough_token_count(message.name)
    if message.tool_calls:
        tokens += _rough_token_count(str(message.tool_calls))
    return tokens + 4


def _rough_token_count(text: str | None) -> int:
    return max(1, len((text or "").split()))


def _format_summary(messages: list[LlmMessage]) -> str:
    lines = ["Intermediate work was summarized to keep the model context compact."]
    for message in messages[:30]:
        if message.role == "tool":
            preview = (message.content or "").replace("\n", " ")[:180]
            lines.append(f"- Tool {message.name or 'unknown'} returned: {preview}")
        elif message.tool_calls:
            names = ", ".join(str(call.get("name") or call.get("tool_name")) for call in message.tool_calls)
            lines.append(f"- Assistant requested tool(s): {names}")
        else:
            preview = (message.content or "").replace("\n", " ")[:180]
            lines.append(f"- {message.role}: {preview}")
    if len(messages) > 30:
        lines.append(f"- {len(messages) - 30} additional message(s) omitted from summary.")
    return "\n".join(lines)
