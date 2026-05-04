from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class ToolCompactionStrategy(Protocol):
    def compact_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        ...

    def compact_tool_result(
        self,
        content: str,
        args: dict[str, Any],
        artifact: dict[str, Any],
    ) -> str:
        ...

    def preserve_artifact(self, content: str, artifact: dict[str, Any]) -> bool:
        ...


@dataclass(slots=True)
class TruncateCompactionStrategy:
    label: str = "tool"
    max_chars: int = 1_200

    def compact_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        return dict(tool_call)

    def compact_tool_result(
        self,
        content: str,
        args: dict[str, Any],
        artifact: dict[str, Any],
    ) -> str:
        text = (content or "").strip()
        if len(text) <= self.max_chars:
            return text
        return f"{self.label} result compacted. Args={_safe_json(args)}\n{text[: self.max_chars]}..."

    def preserve_artifact(self, content: str, artifact: dict[str, Any]) -> bool:
        return bool(content or artifact)


@dataclass(slots=True)
class ReferenceCompactionStrategy(TruncateCompactionStrategy):
    reference_template: str = "{label} result processed. Re-run the tool with narrower arguments if needed."

    def compact_tool_result(
        self,
        content: str,
        args: dict[str, Any],
        artifact: dict[str, Any],
    ) -> str:
        return self.reference_template.format(label=self.label, args=_safe_json(args))


class RawJsonLogsCompactionStrategy(TruncateCompactionStrategy):
    def __init__(self, bucket_size: int = 100, selected_per_bucket: int = 10) -> None:
        super().__init__("Raw JSON logs", 8_000)
        self._bucket_size = bucket_size
        self._selected_per_bucket = selected_per_bucket

    def compact_tool_result(
        self,
        content: str,
        args: dict[str, Any],
        artifact: dict[str, Any],
    ) -> str:
        entries = _extract_entries(content, artifact)
        if entries is None:
            return super().compact_tool_result(content, args, artifact)

        compacted = _stratified_sample_entries(
            entries,
            bucket_size=self._bucket_size,
            selected_per_bucket=self._selected_per_bucket,
        )
        payload = {
            "metadata": {
                "compaction": "stratified_raw_json",
                "original_entry_count": len(entries),
                "compacted_entry_count": len(compacted),
                "bucket_size": self._bucket_size,
                "selected_per_bucket": self._selected_per_bucket,
            },
            "entries": compacted,
        }
        return json.dumps(payload, ensure_ascii=True, indent=2, default=str)


class CodeSearchCompactionStrategy(TruncateCompactionStrategy):
    def __init__(self) -> None:
        super().__init__("Code search", 2_000)

    def compact_tool_result(
        self,
        content: str,
        args: dict[str, Any],
        artifact: dict[str, Any],
    ) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        header = lines[:1]
        matches = [line[:220] for line in lines[1:41]]
        if len(lines) > 42:
            matches.append(f"... compacted {len(lines) - 41} additional matches")
        return "\n".join(header + matches) if lines else content[:300]


class FileReadCompactionStrategy(TruncateCompactionStrategy):
    def __init__(self) -> None:
        super().__init__("File read", 2_600)

    def compact_tool_result(
        self,
        content: str,
        args: dict[str, Any],
        artifact: dict[str, Any],
    ) -> str:
        path = args.get("file_path") or args.get("path") or "unknown"
        lines = [line for line in content.splitlines() if line.strip()]
        kept = lines[:80]
        return f"File content compacted for {path}. Re-read narrower ranges if needed.\n" + "\n".join(
            line[:240] for line in kept
        )


class DeploymentEventsCompactionStrategy(TruncateCompactionStrategy):
    def __init__(self) -> None:
        super().__init__("Deployment events", 2_400)

    def compact_tool_result(
        self,
        content: str,
        args: dict[str, Any],
        artifact: dict[str, Any],
    ) -> str:
        lines = [line for line in content.splitlines() if line.strip()]
        kept = lines[:25]
        if len(lines) > 25:
            kept.append(f"... compacted {len(lines) - 25} additional lines")
        return "\n".join(kept)


class FinalReportCompactionStrategy(TruncateCompactionStrategy):
    def __init__(self) -> None:
        super().__init__("Final report", 8_000)


def create_compaction_strategy(name: str | None) -> ToolCompactionStrategy:
    strategy = (name or "truncate").strip().lower().replace("-", "_")
    if strategy in {"truncate", "default"}:
        return TruncateCompactionStrategy()
    if strategy == "reference":
        return ReferenceCompactionStrategy()
    if strategy == "raw_json_logs":
        return RawJsonLogsCompactionStrategy()
    if strategy == "code_search":
        return CodeSearchCompactionStrategy()
    if strategy == "file_read":
        return FileReadCompactionStrategy()
    if strategy == "deployment_events":
        return DeploymentEventsCompactionStrategy()
    if strategy == "final_report":
        return FinalReportCompactionStrategy()
    raise ValueError(f"Unknown compaction strategy: {name}")


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)[:500]
    except TypeError:
        return str(value)[:500]


def _extract_entries(content: str, artifact: dict[str, Any]) -> list[Any] | None:
    entries = artifact.get("entries")
    if isinstance(entries, list):
        return entries
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("entries"), list):
        return parsed["entries"]
    return None


def _stratified_sample_entries(
    entries: list[Any],
    *,
    bucket_size: int,
    selected_per_bucket: int,
) -> list[Any]:
    if bucket_size <= 0 or selected_per_bucket <= 0:
        return []
    selected: list[Any] = []
    for start in range(0, len(entries), bucket_size):
        bucket = entries[start : start + bucket_size]
        for index in _evenly_spaced_indices(len(bucket), min(selected_per_bucket, len(bucket))):
            selected.append(bucket[index])
    return selected


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]
