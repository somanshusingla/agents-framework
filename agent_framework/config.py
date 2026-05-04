from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelSpec(BaseModel):
    provider: str = "openai"
    name: str = "gpt-5.4-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ToolsSpec(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    compaction_overrides: dict[str, str] = Field(default_factory=dict)


class SummarizationSpec(BaseModel):
    enabled: bool = True
    keep_recent: int = 5
    strategy: Literal["llm", "deterministic", "none"] = "llm"
    provider: str | None = None
    model: str | None = None
    max_summary_tokens: int = 1024


class ContextSpec(BaseModel):
    strategy: Literal["hierarchical", "none"] = "hierarchical"
    token_threshold: int = 50_000
    keep_recent: int = 10
    tool_result_threshold: int = 1_000
    default_tool_strategy: str = "truncate"
    summarization: SummarizationSpec = Field(default_factory=SummarizationSpec)


class RuntimeSpec(BaseModel):
    max_steps: int = 10
    required_first_tool: str | None = None
    final_tool: str | None = None
    final_requires_reflection: bool = False


class HooksSpec(BaseModel):
    bootstrap: str | None = None
    tool_call_policy: str | None = None


class PersistenceSpec(BaseModel):
    backend: str = "sqlite"
    db_url: str | None = None


class AgentSpec(BaseModel):
    name: str = "default"
    description: str = ""
    workflow: Literal["react"] = "react"
    system_prompt: str = ""
    system_prompt_path: Path | None = None
    model: ModelSpec = Field(default_factory=ModelSpec)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    context: ContextSpec = Field(default_factory=ContextSpec)
    hooks: HooksSpec = Field(default_factory=HooksSpec)
    persistence: PersistenceSpec = Field(default_factory=PersistenceSpec)

    @model_validator(mode="after")
    def validate_workflow(self) -> "AgentSpec":
        if self.workflow != "react":
            raise ValueError(f"Unsupported workflow template: {self.workflow}")
        return self


def load_agent_spec(path: str | Path) -> AgentSpec:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Agent config must be a mapping: {config_path}")

    data = _normalize_agent_config(raw)
    spec = AgentSpec.model_validate(data)
    if spec.system_prompt_path:
        prompt_path = spec.system_prompt_path
        if not prompt_path.is_absolute():
            prompt_path = config_path.parent / prompt_path
        spec.system_prompt = prompt_path.read_text(encoding="utf-8")
        spec.system_prompt_path = prompt_path
    return spec


def _normalize_agent_config(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    nested_agent = data.pop("agent", None)
    if isinstance(nested_agent, dict):
        merged = dict(nested_agent)
        for key, value in data.items():
            if key not in merged:
                merged[key] = value
        data = merged
    return data
