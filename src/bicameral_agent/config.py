"""Centralized hyperparameter configuration system.

Provides nested frozen Pydantic v2 models that load from TOML, support
environment variable overrides (``BICAMERAL_`` prefix, ``__`` nesting
separator), and produce existing config types via adapter methods.
"""

from __future__ import annotations

import os
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bicameral_agent.model_client import (
    VALID_THINKING_LEVELS,
    default_model,
    provider_names,
    validate_provider_model,
)
from bicameral_agent.heuristic_controller import (
    DEFAULT_AUDITOR_HIGH_STOP_THRESHOLD,
    DEFAULT_AUDITOR_STOP_THRESHOLD,
    DEFAULT_QUEUE_DEPTH_GUARD,
    DEFAULT_REFRESHER_INTERVAL,
    DEFAULT_SCANNER_INTERVAL,
    DEFAULT_STAGGER_TOLERANCE_MS,
)
from bicameral_agent.queue import InterruptConfig, Priority


_DEFAULT_TOML = files("bicameral_agent.data").joinpath("default_config.toml")


class ModelConfig(BaseModel):
    """LLM model configuration.

    ``name`` left unset defaults to the provider's default model from the
    ``model_client`` registry, and is cross-checked against ``provider`` so
    e.g. provider='ollama' with a Gemini tag is rejected at config time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = "gemini"
    name: str = ""  # empty -> the provider's default model (see below)
    thinking_level: str = "medium"
    temperature: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_provider_default_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("name"):
            provider = data.get("provider", "gemini")
            if provider in provider_names():
                data = {**data, "name": default_model(provider)}
        return data

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        allowed = set(provider_names())
        if v not in allowed:
            msg = f"provider must be one of {allowed}, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("thinking_level")
    @classmethod
    def _validate_thinking_level(cls, v: str) -> str:
        if v not in VALID_THINKING_LEVELS:
            msg = f"thinking_level must be one of {sorted(VALID_THINKING_LEVELS)}, got {v!r}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _cross_validate_name(self) -> ModelConfig:
        validate_provider_model(self.provider, self.name)
        return self

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 2.0):
            msg = f"temperature must be between 0.0 and 2.0, got {v}"
            raise ValueError(msg)
        return v


class QueueConfig(BaseModel):
    """Context queue thresholds and injection semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count_threshold: int = 5
    priority_threshold: int = 3
    token_threshold: int = 1000
    expiry_turns: int | None = None
    persistent_injection: bool = True

    @field_validator("expiry_turns")
    @classmethod
    def _validate_expiry_turns(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            msg = f"expiry_turns must be >= 1, got {v}"
            raise ValueError(msg)
        return v


class ToolBudgetConfig(BaseModel):
    """Token budget for a single tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_calls: int = 10
    max_input_tokens: int = 50_000
    max_output_tokens: int = 20_000


SEARCH_PROVIDER_NAMES = ("mock", "brave")
"""Search backends selectable for the research gap scanner (issue #100)."""


class ToolsConfig(BaseModel):
    """Tool budget and search-backend configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_budget: ToolBudgetConfig = Field(default_factory=ToolBudgetConfig)
    budgets: dict[str, ToolBudgetConfig] = Field(default_factory=dict)
    search_provider: str = "mock"
    """Gap-scanner search backend: "mock" (offline snippets) or "brave"
    (Brave Web Search API; requires ``BRAVE_API_KEY``). Overridden by the
    scripts' ``--search-provider`` flag (CLI > config > mock)."""

    @field_validator("search_provider")
    @classmethod
    def _validate_search_provider(cls, v: str) -> str:
        if v not in SEARCH_PROVIDER_NAMES:
            msg = f"search_provider must be one of {SEARCH_PROVIDER_NAMES}, got {v!r}"
            raise ValueError(msg)
        return v


class HeuristicConfig(BaseModel):
    """Heuristic controller tuning parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scanner_interval: int = Field(default=DEFAULT_SCANNER_INTERVAL, ge=1)
    refresher_interval: int = Field(default=DEFAULT_REFRESHER_INTERVAL, ge=1)
    auditor_stop_threshold: int = DEFAULT_AUDITOR_STOP_THRESHOLD
    auditor_high_stop_threshold: int = DEFAULT_AUDITOR_HIGH_STOP_THRESHOLD
    queue_depth_guard: int = DEFAULT_QUEUE_DEPTH_GUARD
    stagger_tolerance_ms: float = DEFAULT_STAGGER_TOLERANCE_MS


class TrainingConfig(BaseModel):
    """RL training hyperparameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_simulations: int = 100
    gamma: float = 0.99
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "quality": 1.0,
            "efficiency": 0.5,
            "token_waste": -0.3,
        }
    )

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, v: float) -> float:
        if v <= 0:
            msg = f"learning_rate must be > 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("gamma")
    @classmethod
    def _validate_gamma(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            msg = f"gamma must be between 0.0 and 1.0, got {v}"
            raise ValueError(msg)
        return v


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search hyperparameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    c_puct: float = 1.4
    dirichlet_alpha: float = 0.3
    num_simulations: int = 100
    temperature: float = 1.0


class CostConfig(BaseModel):
    """Cost tracking and budget configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_budget: float | None = None
    episode_budget: float | None = None


class EvaluationConfig(BaseModel):
    """Evaluation run configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_tasks: int = 10
    random_seed: int = 42


class RunConfig(BaseModel):
    """Episode-run execution settings shared by the runner scripts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parallel_episodes: int = Field(default=1, ge=1)
    """Episodes run concurrently (1 = sequential).

    Should match the provider's concurrent-request allowance; overridden
    by the scripts' ``--parallel-episodes`` flag (CLI > config > 1).
    """


class HyperConfig(BaseModel):
    """Root hyperparameter configuration.

    Loads from TOML, supports env var overrides, and provides adapter
    methods to produce existing config types (``EpisodeConfig``,
    ``InterruptConfig``, ``TokenBudget``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    measurement_model: ModelConfig | None = None
    """Model pinned for the measurement roles (LLM judge and simulated user).

    A single section rather than per-role ones: judge and sim-user together
    form the measurement apparatus, which issue #53 requires held fixed as
    one unit while ``[model]`` (the answerer) varies. ``None`` means the
    measurement roles use the answerer's client (back-compat). Only
    ``provider``/``name`` are consumed; the measurement roles manage their
    own generation settings.
    """
    queue: QueueConfig = Field(default_factory=QueueConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    heuristic: HeuristicConfig = Field(default_factory=HeuristicConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> HyperConfig:
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_defaults(cls) -> HyperConfig:
        """Load the bundled default configuration."""
        raw = _DEFAULT_TOML.read_text(encoding="utf-8")
        data = tomllib.loads(raw)
        return cls.model_validate(data)

    def with_env_overrides(self) -> HyperConfig:
        """Return a new instance with ``BICAMERAL_`` env var overrides applied.

        Environment variables use the ``BICAMERAL_`` prefix and ``__`` as the
        nesting separator. For example, ``BICAMERAL_MODEL__NAME=gemini-2``
        sets ``model.name``.
        """
        overrides: dict[str, Any] = {}
        prefix = "BICAMERAL_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].lower().split("__")
            _set_nested(overrides, parts, _coerce_value(value))

        if not overrides:
            return self

        current = self.model_dump()
        # A provider override without an explicit name drops the old
        # provider's model name so the new provider's default applies.
        model_ov = overrides.get("model")
        if (
            isinstance(model_ov, dict)
            and "name" not in model_ov
            and model_ov.get("provider") not in (None, self.model.provider)
        ):
            current["model"].pop("name", None)
        _deep_merge(current, overrides)
        return HyperConfig.model_validate(current)

    def to_episode_config(self, **overrides: Any):
        """Produce an ``EpisodeConfig`` from these hyperparameters.

        Accepts keyword overrides for any ``EpisodeConfig`` field.
        """
        from bicameral_agent.episode_runner import EpisodeConfig

        defaults = {
            "thinking_level": self.model.thinking_level,
            "temperature": self.model.temperature,
            "interrupt_config": self.to_interrupt_config(),
            "tool_token_budget": self.to_token_budget(),
            "persistent_injection": self.queue.persistent_injection,
            "queue_expiry_turns": self.queue.expiry_turns,
        }
        defaults.update(overrides)
        return EpisodeConfig(**defaults)

    def to_heuristic_controller(self):
        """Produce a ``HeuristicController`` tuned by heuristic settings."""
        from bicameral_agent.heuristic_controller import HeuristicController

        return HeuristicController(
            scanner_interval=self.heuristic.scanner_interval,
            refresher_interval=self.heuristic.refresher_interval,
            auditor_stop_threshold=self.heuristic.auditor_stop_threshold,
            auditor_high_stop_threshold=self.heuristic.auditor_high_stop_threshold,
            queue_depth_guard=self.heuristic.queue_depth_guard,
            stagger_tolerance_ms=self.heuristic.stagger_tolerance_ms,
        )

    def to_interrupt_config(self) -> InterruptConfig:
        """Produce an ``InterruptConfig`` from queue settings."""
        return InterruptConfig(
            count_threshold=self.queue.count_threshold,
            priority_threshold=Priority(self.queue.priority_threshold),
            token_threshold=self.queue.token_threshold,
        )

    def to_token_budget(self, tool_id: str | None = None):
        """Produce a ``TokenBudget``, using per-tool config if available."""
        from bicameral_agent.tool_primitive import TokenBudget

        budget_cfg = self.tools.budgets.get(tool_id) if tool_id else None
        if budget_cfg is None:
            budget_cfg = self.tools.default_budget
        return TokenBudget(
            max_calls=budget_cfg.max_calls,
            max_input_tokens=budget_cfg.max_input_tokens,
            max_output_tokens=budget_cfg.max_output_tokens,
        )

    def to_cost_tracker(self):
        """Produce a ``CostTracker`` from cost settings."""
        from bicameral_agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.set_budget(self.cost.session_budget)
        tracker.set_episode_budget(self.cost.episode_budget)
        return tracker

    def to_model_client(self, *, on_completion=None):
        """Produce a model client for the configured provider and model."""
        from bicameral_agent.model_client import build_client

        return build_client(
            self.model.provider,
            self.model.name,
            on_completion=on_completion,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for episode metadata storage."""
        return self.model_dump()


def _set_nested(d: dict, parts: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of key parts."""
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _coerce_value(raw: str) -> str | int | float | bool:
    """Attempt to coerce an env var string to a typed value."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
