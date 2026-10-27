"""Model registry (configs/models.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .paths import REPO_ROOT

DEFAULT_REGISTRY = REPO_ROOT / "configs" / "models.yaml"


class RequestDefaults(BaseModel):
    """Per-request generation parameters. Layered: model -> template config -> CLI."""

    model_config = ConfigDict(extra="forbid")

    reasoning_effort: str | None = "low"
    temperature: float = 0.0
    top_p: float | None = None
    seed: int = 0
    max_tokens: int = 8192
    timeout_s: float = 900.0
    model_retries: int = 1
    retry_temperature: float = 0.7
    transient_retries: int = 6
    extra_body: dict[str, Any] | None = None

    def merged(self, overrides: dict[str, Any] | None) -> RequestDefaults:
        if not overrides:
            return self
        data = self.model_dump()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return RequestDefaults.model_validate(data)


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    hf_repo: str
    local_dir: str
    served_name: str
    hf_exclude: list[str] = Field(default_factory=list)
    startup_timeout_s: float = 1200.0
    concurrency: int = 128
    vllm_args: dict[str, Any] = Field(default_factory=dict)
    request: RequestDefaults = Field(default_factory=RequestDefaults)

    def model_path(self, models_dir: Path) -> Path:
        return models_dir / self.local_dir


def load_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(Path(path or DEFAULT_REGISTRY).read_text()) or {}
    return {name: ModelSpec.model_validate({"name": name, **spec}) for name, spec in raw.items()}


def get_model(name: str, path: str | Path | None = None) -> ModelSpec:
    registry = load_registry(path)
    if name not in registry:
        raise KeyError(f"Unknown model {name!r}. Known models: {', '.join(sorted(registry))}")
    return registry[name]
