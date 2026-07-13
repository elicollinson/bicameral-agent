"""Tests for the shared CLI flag registration / client resolution helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from bicameral_agent.brave_search import BraveSearchProvider
from bicameral_agent.config import HyperConfig
from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.episode_runner import Controller, EpisodeConfig, EpisodeRunner
from bicameral_agent.gemini import GeminiResponse
from bicameral_agent.heuristic_controller import Action
from bicameral_agent.runner_setup import (
    add_model_args,
    effective_hyper_config,
    resolve_parallel_episodes,
    resolve_runner_clients,
    resolve_search_provider,
)
from bicameral_agent.simulated_user import ActionType, UserAction


@dataclass
class _FakeClient:
    model: str


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    return parser.parse_args(argv)


@pytest.fixture
def hyper() -> HyperConfig:
    return HyperConfig.from_defaults()


class TestAddModelArgs:
    def test_registers_shared_flags_with_none_defaults(self):
        args = _parse([])
        assert args.config is None
        assert args.provider is None
        assert args.model is None
        assert args.judge_provider is None
        assert args.judge_model is None
        assert args.episode_budget is None
        assert args.search_provider is None

    def test_rejects_unknown_search_provider(self):
        with pytest.raises(SystemExit):
            _parse(["--search-provider", "not-a-backend"])

    def test_parses_values(self):
        args = _parse(
            [
                "--provider", "gemini",
                "--model", "tag-a",
                "--judge-provider", "gemini",
                "--judge-model", "tag-b",
                "--episode-budget", "0.5",
            ]
        )
        assert (args.provider, args.model) == ("gemini", "tag-a")
        assert (args.judge_provider, args.judge_model) == ("gemini", "tag-b")
        assert args.episode_budget == 0.5

    def test_rejects_unknown_provider(self):
        with pytest.raises(SystemExit):
            _parse(["--provider", "not-a-provider"])


class TestResolveRunnerClients:
    def test_cli_overrides_config(self, hyper):
        args = _parse(["--provider", "gemini", "--model", "tag-a"])
        with patch(
            "bicameral_agent.runner_setup.build_client",
            side_effect=lambda provider, model: _FakeClient(model=model or "default"),
        ) as mock_build:
            resolve_runner_clients(args, hyper)
        assert mock_build.call_args_list[0].args == ("gemini", "tag-a")

    def test_config_used_when_no_cli_flags(self, hyper):
        args = _parse([])
        with patch(
            "bicameral_agent.runner_setup.build_client",
            side_effect=lambda provider, model: _FakeClient(model=model or "default"),
        ) as mock_build:
            resolve_runner_clients(args, hyper)
        assert mock_build.call_args_list[0].args == (
            hyper.model.provider,
            hyper.model.name,
        )

    def test_judge_client_reused_when_roles_match(self, hyper):
        args = _parse(
            [
                "--provider", "gemini", "--model", "tag-a",
                "--judge-provider", "gemini", "--judge-model", "tag-a",
            ]
        )
        with patch(
            "bicameral_agent.runner_setup.build_client",
            side_effect=lambda provider, model: _FakeClient(model=model),
        ) as mock_build:
            client, judge_client, provenance = resolve_runner_clients(args, hyper)
        assert judge_client is client
        assert mock_build.call_count == 1
        assert provenance["answerer"] == {"provider": "gemini", "model": "tag-a"}
        assert provenance["measurement"] == {"provider": "gemini", "model": "tag-a"}

    def test_distinct_judge_gets_own_client(self, hyper):
        args = _parse(
            [
                "--provider", "gemini", "--model", "tag-a",
                "--judge-provider", "gemini", "--judge-model", "tag-b",
            ]
        )
        with patch(
            "bicameral_agent.runner_setup.build_client",
            side_effect=lambda provider, model: _FakeClient(model=model),
        ) as mock_build:
            client, judge_client, provenance = resolve_runner_clients(args, hyper)
        assert judge_client is not client
        assert mock_build.call_count == 2
        assert provenance["measurement"] == {"provider": "gemini", "model": "tag-b"}


class TestResolveSearchProvider:
    """CLI flag > config [tools].search_provider > mock (None)."""

    def test_defaults_to_none_for_mock(self, hyper):
        args = _parse([])
        assert resolve_search_provider(args, hyper) is None

    def test_config_brave_builds_provider(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        args = _parse([])
        hyper = HyperConfig.model_validate({"tools": {"search_provider": "brave"}})
        assert isinstance(resolve_search_provider(args, hyper), BraveSearchProvider)

    def test_cli_brave_overrides_config_mock(self, monkeypatch, hyper):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        args = _parse(["--search-provider", "brave"])
        assert isinstance(resolve_search_provider(args, hyper), BraveSearchProvider)

    def test_cli_mock_overrides_config_brave(self):
        args = _parse(["--search-provider", "mock"])
        hyper = HyperConfig.model_validate({"tools": {"search_provider": "brave"}})
        assert resolve_search_provider(args, hyper) is None

    def test_brave_without_key_fails_fast(self, monkeypatch, hyper):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        args = _parse(["--search-provider", "brave"])
        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            resolve_search_provider(args, hyper)


class TestResolveParallelEpisodes:
    """CLI flag > config [run].parallel_episodes > default 1."""

    def test_defaults_to_one(self, hyper):
        args = argparse.Namespace(parallel_episodes=None)
        assert resolve_parallel_episodes(args, hyper) == 1

    def test_config_used_when_flag_unset(self):
        args = argparse.Namespace(parallel_episodes=None)
        hyper = HyperConfig.model_validate({"run": {"parallel_episodes": 5}})
        assert resolve_parallel_episodes(args, hyper) == 5

    def test_cli_overrides_config(self):
        args = argparse.Namespace(parallel_episodes=3)
        hyper = HyperConfig.model_validate({"run": {"parallel_episodes": 5}})
        assert resolve_parallel_episodes(args, hyper) == 3

    def test_explicit_one_overrides_config(self):
        args = argparse.Namespace(parallel_episodes=1)
        hyper = HyperConfig.model_validate({"run": {"parallel_episodes": 5}})
        assert resolve_parallel_episodes(args, hyper) == 1


def _fake_build_client(provider: str, model: str | None) -> MagicMock:
    """A client double matching what the scripts get from ``build_client``."""
    client = MagicMock()
    client.model = model or "fake-default"
    client.generate.return_value = GeminiResponse(
        content="Test response",
        input_tokens=10,
        output_tokens=20,
        duration_ms=100.0,
        finish_reason="STOP",
    )
    return client


def _resolve_effective(
    argv: list[str], hyper: HyperConfig, parallel_episodes: int | None = None
):
    """Run the scripts' resolution flow and return (client, effective config)."""
    args = _parse(argv)
    args.parallel_episodes = parallel_episodes
    with patch(
        "bicameral_agent.runner_setup.build_client", side_effect=_fake_build_client
    ):
        client, _judge, provenance = resolve_runner_clients(args, hyper)
    return client, effective_hyper_config(args, hyper, provenance)


class TestEffectiveHyperConfig:
    """The runner's hyper_config must carry resolved values, not defaults (#103)."""

    def test_cli_overrides_recorded(self, hyper):
        _, effective = _resolve_effective(
            [
                "--provider", "ollama", "--model", "tag-a",
                "--judge-provider", "gemini", "--judge-model", "tag-b",
                "--search-provider", "brave",
            ],
            hyper,
            parallel_episodes=10,
        )
        assert (effective.model.provider, effective.model.name) == ("ollama", "tag-a")
        assert effective.measurement_model is not None
        assert (
            effective.measurement_model.provider,
            effective.measurement_model.name,
        ) == ("gemini", "tag-b")
        assert effective.run.parallel_episodes == 10
        assert effective.tools.search_provider == "brave"

    def test_no_flags_keeps_config_values(self, hyper):
        _, effective = _resolve_effective([], hyper)
        assert effective.model == hyper.model
        assert effective.run.parallel_episodes == hyper.run.parallel_episodes
        assert effective.tools.search_provider == hyper.tools.search_provider
        # The resolved measurement pin is stamped even when the config
        # leaves [measurement_model] unset (defaults to gemini, issue #53).
        assert effective.measurement_model is not None
        assert effective.measurement_model.provider == "gemini"

    def test_original_config_untouched(self, hyper):
        _resolve_effective(["--provider", "ollama"], hyper, parallel_episodes=10)
        assert hyper.model.provider == "gemini"
        assert hyper.run.parallel_episodes == 1

    def test_episode_metadata_records_effective_values(self, hyper):
        """A mocked-client run with CLI-style overrides stamps the resolved
        provider/parallel_episodes/search_provider into episode metadata."""
        client, effective = _resolve_effective(
            ["--provider", "ollama", "--search-provider", "brave"],
            hyper,
            parallel_episodes=10,
        )

        runner = EpisodeRunner(
            client, config=EpisodeConfig(max_turns=3), hyper_config=effective
        )
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING
        task = ResearchQATask(
            task_id="test-103",
            difficulty=TaskDifficulty.TYPICAL,
            split=TaskSplit.EVAL,
            question="What is photosynthesis?",
            gold_answer="Plants convert light energy into chemical energy.",
            known_gaps=None,
            known_assumptions=None,
            scoring_rubric="5: Complete explanation. 3: Partial. 1: Wrong.",
        )
        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            sim = MagicMock()
            sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = sim
            episode = runner.run_episode(task, ctrl)

        recorded = episode.metadata["hyperparameters"]
        assert recorded["model"]["provider"] == "ollama"
        assert recorded["run"]["parallel_episodes"] == 10
        assert recorded["tools"]["search_provider"] == "brave"
        assert recorded["measurement_model"]["provider"] == "gemini"
