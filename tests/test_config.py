"""Tests for the hyperparameter configuration system."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from bicameral_agent.config import (
    HeuristicConfig,
    HyperConfig,
    ModelConfig,
    QueueConfig,
    ToolBudgetConfig,
    ToolsConfig,
    TrainingConfig,
)
from bicameral_agent.episode_runner import EpisodeConfig
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.heuristic_controller import Action, ExecutingTool, FullState
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
        assert cfg.queue.expiry_turns is None
        assert cfg.queue.persistent_injection is True

    def test_tools_defaults(self):
        cfg = HyperConfig()
        assert cfg.tools.default_budget.max_calls == 10
        assert cfg.tools.default_budget.max_input_tokens == 50_000
        assert cfg.tools.default_budget.max_output_tokens == 20_000
        assert cfg.tools.budgets == {}

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

    def test_measurement_model_defaults_to_none(self):
        """Unset [measurement_model] means measurement roles use [model]."""
        assert HyperConfig().measurement_model is None
        assert HyperConfig.from_defaults().measurement_model is None


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

    def test_from_toml_unknown_key_raises(self, tmp_path):
        toml_file = tmp_path / "typo.toml"
        toml_file.write_text(textwrap.dedent("""\
            [training]
            learning_rat = 0.01
        """))
        with pytest.raises(ValidationError, match="learning_rat"):
            HyperConfig.from_toml(toml_file)

    def test_from_toml_measurement_model(self, tmp_path):
        """[measurement_model] parses, fills the provider's default name."""
        toml_file = tmp_path / "measurement.toml"
        toml_file.write_text(textwrap.dedent("""\
            [model]
            provider = "ollama"

            [measurement_model]
            provider = "gemini"
        """))
        cfg = HyperConfig.from_toml(toml_file)
        assert cfg.model.provider == "ollama"
        assert cfg.measurement_model is not None
        assert cfg.measurement_model.provider == "gemini"
        assert cfg.measurement_model.name == "gemini-3.1-flash-lite-preview"

    def test_from_toml_measurement_model_mismatch_raises(self, tmp_path):
        """A cross-provider model tag is rejected at config time."""
        toml_file = tmp_path / "mismatch.toml"
        toml_file.write_text(textwrap.dedent("""\
            [measurement_model]
            provider = "ollama"
            name = "gemini-3.1-flash-lite-preview"
        """))
        with pytest.raises(ValidationError):
            HyperConfig.from_toml(toml_file)

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

    def test_typo_field_raises(self, monkeypatch):
        """A misspelled override errors instead of being silently dropped."""
        monkeypatch.setenv("BICAMERAL_TRAINING__LEARNING_RAT", "0.01")
        with pytest.raises(ValidationError, match="learning_rat"):
            HyperConfig().with_env_overrides()

    def test_typo_section_raises(self, monkeypatch):
        monkeypatch.setenv("BICAMERAL_TRANING__LEARNING_RATE", "0.01")
        with pytest.raises(ValidationError, match="traning"):
            HyperConfig().with_env_overrides()


class TestValidation:
    """Pydantic validation catches invalid values."""

    def test_invalid_thinking_level(self):
        with pytest.raises(ValidationError, match="thinking_level"):
            ModelConfig(thinking_level="ultra")

    def test_thinking_level_none_rejected(self):
        """'none' is not part of the client vocabulary; use 'minimal'."""
        with pytest.raises(ValidationError, match="thinking_level"):
            ModelConfig(thinking_level="none")

    def test_thinking_level_minimal_accepted(self):
        assert ModelConfig(thinking_level="minimal").thinking_level == "minimal"

    def test_expiry_turns_below_one_rejected(self):
        with pytest.raises(ValidationError, match="expiry_turns"):
            QueueConfig(expiry_turns=0)

    def test_scanner_interval_zero_rejected(self):
        with pytest.raises(ValidationError, match="scanner_interval"):
            HeuristicConfig(scanner_interval=0)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="learning_rat"):
            TrainingConfig(learning_rat=0.01)

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
        assert ec.temperature is None
        assert ec.queue_expiry_turns is None
        assert ec.tool_token_budget.max_calls == 10
        assert ec.tool_token_budget.max_input_tokens == 50_000
        assert ec.persistent_injection is True

    def test_to_episode_config_carries_temperature_and_expiry(self):
        cfg = HyperConfig(
            model=ModelConfig(temperature=0.3),
            queue=QueueConfig(expiry_turns=2),
        )
        ec = cfg.to_episode_config()
        assert ec.temperature == pytest.approx(0.3)
        assert ec.queue_expiry_turns == 2

    def test_to_heuristic_controller(self):
        from bicameral_agent.heuristic_controller import HeuristicController

        cfg = HyperConfig()
        ctrl = cfg.to_heuristic_controller()
        assert isinstance(ctrl, HeuristicController)

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


_CONFIG_THINKING_LEVELS = ["minimal", "low", "medium", "high"]


class TestThinkingLevelVocabulary:
    """Every config-allowed thinking level is accepted end-to-end by both clients."""

    @pytest.mark.parametrize("level", _CONFIG_THINKING_LEVELS)
    def test_config_accepts_level(self, level):
        assert ModelConfig(thinking_level=level).thinking_level == level

    @pytest.mark.parametrize("level", _CONFIG_THINKING_LEVELS)
    def test_gemini_accepts_config_level(self, level):
        from bicameral_agent.gemini import GeminiClient

        cfg = ModelConfig(thinking_level=level)
        with patch("bicameral_agent.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
        sentinel = object()
        with patch.object(
            client, "_execute_with_retry", return_value=sentinel
        ) as mock_exec:
            result = client.generate(
                [{"role": "user", "content": "hi"}],
                thinking_level=cfg.thinking_level,
            )
        assert result is sentinel
        mock_exec.assert_called_once()

    @pytest.mark.parametrize("level", _CONFIG_THINKING_LEVELS)
    def test_ollama_accepts_config_level(self, level):
        from bicameral_agent.ollama_cloud import OllamaCloudClient

        cfg = ModelConfig(thinking_level=level)
        client = OllamaCloudClient(api_key="test-key")
        raw = {
            "message": {"content": "ok"},
            "prompt_eval_count": 1,
            "eval_count": 1,
            "done_reason": "stop",
        }
        with patch.object(client, "_post", return_value=raw) as mock_post:
            result = client.generate(
                [{"role": "user", "content": "hi"}],
                thinking_level=cfg.thinking_level,
            )
        assert result.content == "ok"
        mock_post.assert_called_once()


def _make_state(**overrides) -> FullState:
    defaults = dict(
        turn_number=1,
        stop_count=0,
        followup_type=FollowUpType.ELABORATION,
        queue_depth=0,
        executing_tools=(),
        predicted_latencies={},
    )
    defaults.update(overrides)
    return FullState(**defaults)


class TestHeuristicWiring:
    """Heuristic config values change actual controller decisions."""

    def test_scanner_interval_fires_on_turn_2(self):
        cfg = HyperConfig(heuristic=HeuristicConfig(scanner_interval=2))
        ctrl = cfg.to_heuristic_controller()
        assert ctrl.decide(_make_state(turn_number=2)) == Action.SCANNER

    def test_default_scanner_interval_does_not_fire_on_turn_2(self):
        ctrl = HyperConfig().to_heuristic_controller()
        assert ctrl.decide(_make_state(turn_number=2)) == Action.DO_NOTHING

    def test_refresher_interval_fires_on_turn_3(self):
        cfg = HyperConfig(heuristic=HeuristicConfig(refresher_interval=3))
        ctrl = cfg.to_heuristic_controller()
        assert ctrl.decide(_make_state(turn_number=3)) == Action.REFRESHER

    def test_auditor_stop_threshold_raised_suppresses_auditor(self):
        cfg = HyperConfig(
            heuristic=HeuristicConfig(
                auditor_stop_threshold=3, auditor_high_stop_threshold=4
            )
        )
        ctrl = cfg.to_heuristic_controller()
        # stop_count=2 triggers the auditor at defaults but not here.
        assert ctrl.decide(_make_state(turn_number=3, stop_count=2)) == Action.DO_NOTHING
        default_ctrl = HyperConfig().to_heuristic_controller()
        assert default_ctrl.decide(_make_state(turn_number=3, stop_count=2)) == Action.AUDITOR

    def test_queue_depth_guard_lowered_suppresses_tool(self):
        cfg = HyperConfig(heuristic=HeuristicConfig(queue_depth_guard=1))
        ctrl = cfg.to_heuristic_controller()
        # Turn 1 scanner candidate is suppressed by a queue depth of 1.
        assert ctrl.decide(_make_state(turn_number=1, queue_depth=1)) == Action.DO_NOTHING
        default_ctrl = HyperConfig().to_heuristic_controller()
        assert default_ctrl.decide(_make_state(turn_number=1, queue_depth=1)) == Action.SCANNER

    def test_stagger_tolerance_widened_suppresses_tool(self):
        state = _make_state(
            turn_number=1,
            executing_tools=(
                ExecutingTool(tool_id="other", predicted_remaining_ms=3000.0),
            ),
        )
        cfg = HyperConfig(heuristic=HeuristicConfig(stagger_tolerance_ms=5000.0))
        assert cfg.to_heuristic_controller().decide(state) == Action.DO_NOTHING
        assert HyperConfig().to_heuristic_controller().decide(state) == Action.SCANNER


class TestEpisodeWiring:
    """Model/queue config values reach the API call and queue deposits."""

    def _run_episode(self, hyper: HyperConfig, controller_action=None):
        from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
        from bicameral_agent.episode_runner import Controller, EpisodeRunner
        from bicameral_agent.gemini import GeminiClient, GeminiResponse
        from bicameral_agent.simulated_user import ActionType, UserAction

        client = MagicMock(spec=GeminiClient)
        client.generate.return_value = GeminiResponse(
            content="answer",
            input_tokens=10,
            output_tokens=20,
            duration_ms=100.0,
            finish_reason="STOP",
        )

        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        if controller_action is not None:
            ctrl.decide.side_effect = controller_action
        else:
            ctrl.decide.return_value = Action.DO_NOTHING

        task = ResearchQATask(
            task_id="test-001",
            difficulty=TaskDifficulty.TYPICAL,
            split=TaskSplit.EVAL,
            question="What is photosynthesis?",
            gold_answer="Light energy becomes chemical energy.",
            known_gaps=None,
            known_assumptions=None,
            scoring_rubric="5: Complete. 3: Partial. 1: Wrong.",
        )

        user_actions = iter([
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="Tell me more",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            ),
            UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            ),
        ])

        runner = EpisodeRunner(client, hyper_config=hyper)
        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.side_effect = lambda *a, **k: next(user_actions)
            MockSimUser.return_value = mock_sim
            episode = runner.run_episode(task, ctrl)
        return episode, client

    def test_temperature_and_thinking_level_reach_generate(self):
        hyper = HyperConfig(
            model=ModelConfig(temperature=0.3, thinking_level="low")
        )
        _, client = self._run_episode(hyper)
        kwargs = client.generate.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.3)
        assert kwargs["thinking_level"] == "low"

    def test_default_temperature_is_none_at_generate(self):
        _, client = self._run_episode(HyperConfig())
        assert client.generate.call_args.kwargs["temperature"] is None

    def test_queue_expiry_turns_expires_deposit(self):
        from bicameral_agent.queue import QueueItem
        from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

        hyper = HyperConfig(queue=QueueConfig(expiry_turns=1))
        deposit = QueueItem(
            content="a gap was found",
            priority=Priority.LOW,
            source_tool_id="research_gap_scanner",
            token_count=5,
        )
        result = ToolResult(
            queue_deposit=deposit,
            metadata=ToolMetadata(
                tool_id="research_gap_scanner",
                action_taken="scanned",
                confidence=0.8,
                items_found=1,
                estimated_relevance=0.7,
                tokens_consumed=50,
            ),
        )
        with patch("bicameral_agent.episode_runner.ResearchGapScanner") as MockScanner:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = result
            MockScanner.return_value = mock_tool
            episode, _ = self._run_episode(
                hyper,
                controller_action=[Action.SCANNER, Action.DO_NOTHING],
            )
        # Deposit on turn 1 with expiry_turns=1 expires at the start of turn 2.
        assert episode.metadata["expired_queue_items"] == 1
