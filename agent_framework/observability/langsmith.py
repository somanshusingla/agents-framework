from __future__ import annotations

import os
from contextlib import contextmanager
from time import perf_counter


class LangSmithTracer:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("LANGSMITH_API_KEY"))

    @contextmanager
    def span(self, name: str, attrs: dict | None = None):
        start = perf_counter()
        try:
            yield
        finally:
            _ = perf_counter() - start
            _ = name, attrs
