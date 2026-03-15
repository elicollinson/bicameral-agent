"""Tests for the hyperparameter configuration system."""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from bicameral_agent.config import (
    HyperConfig,
    ModelConfig,
    QueueConfig,
    ToolBudgetConfig,
    ToolsConfig,
    TrainingConfig,
)
from bicameral_agent.episode_runner import EpisodeConfig
from bicameral_agent.queue import InterruptConfig, Priority
from bicameral_agent.tool_primitive import TokenBudget


class TestDefaults:
    """Default config loads without errors and matches hardcoded values."""

    def test_from_defaults_loads(self):
        cfg = HyperConfig.from_defaults()
        assert isinstance(cfg, HyperConfig)

    def test_default_constructor(self):
        cfg = HyperConfig()
        assert cfg.model.name == "gemini-3.1-flash-lite-preview"
        assert cfg.model.thinking_level == "medium"
        assert cfg.model.temperature is None

    def test_queue_defaults(self):
        cfg = HyperConfig()
        assert cfg.queue.count_threshold == 5
        assert cfg.queue.priority_threshold == 3
        assert cfg.queue.token_threshold == 1000
        assert cfg.queue.expiry_turns == 10
        assert cfg.queue.max_depth == 3

    def test_tools_defaults(self):
        cfg = HyperConfig()
        assert cfg.tools.default_budget.max_calls == 10
        assert cfg.tools.default_budget.max_input_tokens == 50_000
        assert cfg.tools.default_budget.max_output_tokens == 20_000
        assert cfg.tools.budgets == {}
        assert cfg.tools.priority_map == {}

    def test_heuristic_defaults(self):
        cfg = HyperConfig()
        assert cfg.heuristic.scanner_interval == 5
        assert cfg.heuristic.refresher_interval == 8
        assert cfg.heuristic.auditor_stop_threshold == 1
        assert cfg.heuristic.auditor_high_stop_threshold == 2
        assert cfg.heuristic.queue_depth_guard == 3
        assert cfg.heuristic.stagger_tolerance_ms == 1000.0

    def test_training_defaults(self):
        cfg = HyperConfig()
        assert cfg.training.learning_rate == pytest.approx(1e-3)
        assert cfg.training.batch_size == 32
        assert cfg.training.gamma == pytest.approx(0.99)

    def test_mcts_defaults(self):
        cfg = HyperConfig()
        assert cfg.mcts.c_puct == pytest.approx(1.4)
        assert cfg.mcts.dirichlet_alpha == pytest.approx(0.3)
        assert cfg.mcts.num_simulations == 100
        assert cfg.mcts.temperature == pytest.approx(1.0)

    def test_evaluation_defaults(self):
        cfg = HyperConfig()
        assert cfg.evaluation.num_tasks == 10
        assert cfg.evaluation.random_seed == 42

    def test_from_defaults_matches_constructor(self):
        from_defaults = HyperConfig.from_defaults()
        from_constructor = HyperConfig()
        assert from_defaults.model_dump() == from_constructor.model_dump()


class TestToml:
    """TOML loading and round-trip."""

    def test_from_toml(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(textwrap.dedent("""\
            [model]
            name = "gemini-2"
            thinking_level = "high"
            temperature = 0.7

            [queue]
            count_threshold = 10

            [heuristic]
            scanner_interval = 3
        """))
        cfg = HyperConfig.from_toml(toml_file)
        assert cfg.model.name == "gemini-2"
        assert cfg.model.thinking_level == "high"
        assert cfg.model.temperature == pytest.approx(0.7)
        assert cfg.queue.count_threshold == 10
        assert cfg.heuristic.scanner_interval == 3
        # Unspecified sections keep defaults
        assert cfg.training.learning_rate == pytest.approx(1e-3)

    def test_round_trip_dict(self):
        cfg = HyperConfig()
        d = cfg.to_dict()
        reconstructed = HyperConfig.model_validate(d)
        assert reconstructed.model_dump() == cfg.model_dump()

    def test_from_toml_with_tool_budgets(self, tmp_path):
        toml_file = tmp_path / "budgets.toml"
        toml_file.write_text(textwrap.dedent("""\
            [tools.budgets.research_gap_scanner]
            max_calls = 5
            max_input_tokens = 25000
            max_output_tokens = 10000
        """))
        cfg = HyperConfig.from_toml(toml_file)
        assert "research_gap_scanner" in cfg.tools.budgets
        b = cfg.tools.budgets["research_gap_scanner"]
        assert b.max_calls == 5
        assert b.max_input_tokens == 25000


class TestEnvOverrides:
    """Environment variable override via BICAMERAL_ prefix."""

    def test_model_name_override(self, monkeypatch):
        monkeypatch.setenv("BICAMERAL_MODEL__NAME", "gemini-2")
        cfg = HyperConfig().with_env_overrides()
        assert cfg.model.name == "gemini-2"

    def test_numeric_override(self, monkeypatch):
        monkeypatch.setenv("BICAMERAL_QUEUE__COUNT_THRESHOLD", "20")
        cfg = HyperConfig().with_env_overrides()
        assert cfg.queue.count_threshold == 20

    def test_float_override(self, monkeypatch):
        monkeypatch.setenv("BICAMERAL_TRAINING__LEARNING_RATE", "0.01")
        cfg = HyperConfig().with_env_overrides()
        assert cfg.training.learning_rate == pytest.approx(0.01)

    def test_no_overrides_returns_equal(self):
        cfg = HyperConfig()
        overridden = cfg.with_env_overrides()
        assert overridden.model_dump() == cfg.model_dump()

    def test_override_preserves_other_fields(self, monkeypatch):
        monkeypatch.setenv("BICAMERAL_MODEL__NAME", "gemini-2")
        cfg = HyperConfig().with_env_overrides()
        assert cfg.model.thinking_level == "medium"
        assert cfg.queue.count_threshold == 5


class TestValidation:
    """Pydantic validation catches invalid values."""

    def test_invalid_thinking_level(self):
        with pytest.raises(ValidationError, match="thinking_level"):
            ModelConfig(thinking_level="ultra")

    def test_temperature_too_high(self):
        with pytest.raises(ValidationError, match="temperature"):
            ModelConfig(temperature=3.0)

    def test_temperature_negative(self):
        with pytest.raises(ValidationError, match="temperature"):
            ModelConfig(temperature=-0.1)

    def test_negative_learning_rate(self):
        with pytest.raises(ValidationError, match="learning_rate"):
            TrainingConfig(learning_rate=-0.001)

    def test_zero_learning_rate(self):
        with pytest.raises(ValidationError, match="learning_rate"):
            TrainingConfig(learning_rate=0.0)

    def test_gamma_too_high(self):
        with pytest.raises(ValidationError, match="gamma"):
            TrainingConfig(gamma=1.5)

    def test_gamma_negative(self):
        with pytest.raises(ValidationError, match="gamma"):
            TrainingConfig(gamma=-0.1)


class TestFrozen:
    """All config models are immutable."""

    def test_hyper_config_frozen(self):
        cfg = HyperConfig()
        with pytest.raises(ValidationError):
            cfg.model = ModelConfig(name="other")

    def test_model_config_frozen(self):
        cfg = ModelConfig()
        with pytest.raises(ValidationError):
            cfg.name = "other"

    def test_queue_config_frozen(self):
        cfg = QueueConfig()
        with pytest.raises(ValidationError):
            cfg.count_threshold = 99

    def test_training_config_frozen(self):
        cfg = TrainingConfig()
        with pytest.raises(ValidationError):
            cfg.learning_rate = 0.1


class TestAdapters:
    """Adapter methods produce correct existing types."""

    def test_to_episode_config(self):
        cfg = HyperConfig()
        ec = cfg.to_episode_config()
        assert isinstance(ec, EpisodeConfig)
        assert ec.thinking_level == "medium"
        assert ec.tool_token_budget.max_calls == 10
        assert ec.tool_token_budget.max_input_tokens == 50_000

    def test_to_episode_config_with_overrides(self):
        cfg = HyperConfig()
        ec = cfg.to_episode_config(max_turns=50)
        assert ec.max_turns == 50

    def test_to_interrupt_config(self):
        cfg = HyperConfig()
        ic = cfg.to_interrupt_config()
        assert isinstance(ic, InterruptConfig)
        assert ic.count_threshold == 5
        assert ic.priority_threshold == Priority.CRITICAL
        assert ic.token_threshold == 1000

    def test_to_token_budget_default(self):
        cfg = HyperConfig()
        tb = cfg.to_token_budget()
        assert isinstance(tb, TokenBudget)
        assert tb.max_calls == 10
        assert tb.max_input_tokens == 50_000
        assert tb.max_output_tokens == 20_000

    def test_to_token_budget_per_tool(self):
        cfg = HyperConfig(
            tools=ToolsConfig(
                budgets={
                    "scanner": ToolBudgetConfig(
                        max_calls=5,
                        max_input_tokens=25_000,
                        max_output_tokens=10_000,
                    )
                }
            )
        )
        tb = cfg.to_token_budget("scanner")
        assert tb.max_calls == 5
        assert tb.max_input_tokens == 25_000

    def test_to_token_budget_unknown_tool_falls_back(self):
        cfg = HyperConfig()
        tb = cfg.to_token_budget("nonexistent_tool")
        assert tb.max_calls == 10  # falls back to default

    def test_to_dict(self):
        cfg = HyperConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "queue" in d
        assert d["model"]["name"] == "gemini-3.1-flash-lite-preview"


class TestEpisodeMetadata:
    """HyperConfig stored in episode metadata via EpisodeRunner."""

    def test_metadata_stored(self, make_episode):
        cfg = HyperConfig()
        ep = make_episode(metadata={"hyperparameters": cfg.to_dict()})
        assert "hyperparameters" in ep.metadata
        hp = ep.metadata["hyperparameters"]
        assert hp["model"]["name"] == "gemini-3.1-flash-lite-preview"
        assert hp["training"]["learning_rate"] == pytest.approx(1e-3)
