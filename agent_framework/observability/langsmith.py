from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator


@dataclass
class TraceSpan:
    name: str
    run: Any = None
    outputs: dict[str, Any] = field(default_factory=dict)

    def add_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs.update(outputs)

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        if self.run is not None and hasattr(self.run, "extra"):
            extra = getattr(self.run, "extra") or {}
            extra.setdefault("metadata", {}).update(metadata)
            self.run.extra = extra


class LangSmithTracer:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        project_name: str | None = None,
    ) -> None:
        self.enabled = _env_enabled() if enabled is None else enabled
        self.project_name = project_name or os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
        self._trace = None
        if self.enabled:
            try:
                from langsmith import trace

                self._trace = trace
            except ImportError:
                self.enabled = False

    @contextmanager
    def span(self, name: str, attrs: dict | None = None) -> Iterator[TraceSpan]:
        attrs = dict(attrs or {})
        run_type = attrs.pop("run_type", "chain")
        inputs = attrs.pop("inputs", None)
        metadata = attrs
        start = perf_counter()

        if not self.enabled or self._trace is None:
            span = TraceSpan(name=name)
            try:
                yield span
            finally:
                span.add_metadata({"duration_ms": round((perf_counter() - start) * 1000, 3)})
            return

        with self._trace(
            name,
            run_type=run_type,
            inputs=inputs or {},
            metadata=metadata,
            project_name=self.project_name,
        ) as run:
            span = TraceSpan(name=name, run=run)
            try:
                yield span
            finally:
                span.add_metadata({"duration_ms": round((perf_counter() - start) * 1000, 3)})
                if span.outputs and hasattr(run, "end"):
                    run.end(outputs=span.outputs)


def _env_enabled() -> bool:
    flag = (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or os.getenv("LANGSMITH_API_KEY")
    )
    return bool(flag and str(flag).lower() not in {"0", "false", "no", "off"})
