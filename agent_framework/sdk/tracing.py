from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any
from typing import Protocol


class TracerProtocol(Protocol):
    def span(self, name: str, attrs: dict | None = None) -> AbstractContextManager[Any]: ...
