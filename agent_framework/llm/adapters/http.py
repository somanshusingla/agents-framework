from __future__ import annotations

import json
import os
from typing import Any

from agent_framework.config import ModelSpec


def resolve_api_key(spec: ModelSpec, default_env: str, provider: str) -> str:
    key = spec.api_key or os.getenv(spec.api_key_env or default_env)
    if not key:
        raise RuntimeError(
            f"Missing API key for {provider}. Set {spec.api_key_env or default_env} "
            "or provide model.api_key in config."
        )
    return key


def import_httpx():
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Provider-backed LLM clients require httpx. Install project dependencies "
            "with `pip install -e .` or add `httpx` to the environment."
        ) from exc
    return httpx


def parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def error_message(provider: str, status_code: int, body: str) -> RuntimeError:
    return RuntimeError(f"{provider} request failed with HTTP {status_code}: {body[:2_000]}")
