"""Tests for the shared CLI flag registration / client resolution helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from bicameral_agent.config import HyperConfig
from bicameral_agent.runner_setup import add_model_args, resolve_runner_clients


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
