"""Shared CLI flags and client resolution for episode-running scripts.

``scripts/run_baseline_benchmark.py`` and ``scripts/train_mcts.py`` both
need the same model/provider flags and the same answerer/measurement
client resolution (provider precedence, the issue #53 measurement-pinning
rule, and the judge-client-reuse optimization). This module is the single
home for that logic so the two scripts cannot drift (PR #79 review).
"""

from __future__ import annotations

import argparse
import logging

from bicameral_agent.config import HyperConfig
from bicameral_agent.model_client import (
    ModelClient,
    build_client,
    default_model,
    provider_names,
)

logger = logging.getLogger(__name__)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Register the model/provider flags shared by episode-running CLIs.

    Adds ``--config``, ``--provider``, ``--model``, ``--judge-provider``,
    ``--judge-model`` and ``--episode-budget``; consume them with
    :func:`resolve_runner_clients`.
    """
    parser.add_argument("--config", default=None,
                        help="Hyperparameter TOML file (defaults to the bundled "
                             "config; BICAMERAL_ env overrides always apply, "
                             "CLI flags win over both).")
    parser.add_argument("--provider", choices=list(provider_names()), default=None,
                        help="Model backend to run the answerer against "
                             "(overrides the config file).")
    parser.add_argument("--model", default=None,
                        help="Model id/tag (overrides the config file).")
    parser.add_argument("--judge-provider", choices=list(provider_names()),
                        default=None,
                        help="Model backend for the measurement roles (LLM "
                             "judge and simulated user), held fixed while "
                             "--provider varies. Defaults to the config's "
                             "[measurement_model], else gemini.")
    parser.add_argument("--judge-model", default=None,
                        help="Measurement model id/tag (overrides the config "
                             "file; unset uses the judge provider's default).")
    parser.add_argument("--episode-budget", type=float, default=None,
                        help="Optional per-episode cost ceiling in USD.")


def resolve_parallel_episodes(args: argparse.Namespace, hyper: HyperConfig) -> int:
    """Resolve episode concurrency: CLI flag > config ``[run]`` > default 1.

    The ``--parallel-episodes`` flag is registered per-script (the help
    text differs) with ``default=None`` so an unset flag is distinguishable
    from an explicit ``1`` and falls through to the config's
    ``run.parallel_episodes`` (validated >= 1 at config load).
    """
    if args.parallel_episodes is not None:
        return args.parallel_episodes
    return hyper.run.parallel_episodes


def resolve_runner_clients(
    args: argparse.Namespace, hyper: HyperConfig
) -> tuple[ModelClient, ModelClient, dict[str, dict[str, str]]]:
    """Resolve the answerer and measurement clients from CLI args + config.

    Precedence for both roles: CLI > config > provider default; a
    configured model name only applies to its configured provider. The
    measurement roles (LLM judge + simulated user) are pinned
    independently of the answerer so cross-model comparisons stay on one
    judging scale (issue #53); when they resolve to the answerer's exact
    provider/model, the answerer's client instance is reused.

    Returns ``(client, judge_client, provenance)`` where ``provenance`` is
    ``{"answerer": {"provider", "model"}, "measurement": {...}}`` for
    reports and logging.
    """
    provider = args.provider or hyper.model.provider
    model = args.model or (hyper.model.name if provider == hyper.model.provider else None)
    client = build_client(provider, model)

    measurement = hyper.measurement_model
    judge_provider = args.judge_provider or (
        measurement.provider if measurement is not None else "gemini"
    )
    judge_model = args.judge_model or (
        measurement.name
        if measurement is not None and judge_provider == measurement.provider
        else None
    )
    resolved_judge_model = judge_model or default_model(judge_provider)
    if judge_provider == provider and resolved_judge_model == client.model:
        judge_client = client
    else:
        judge_client = build_client(judge_provider, judge_model)
    logger.info(
        "Answerer: %s/%s; measurement (judge + sim-user): %s/%s",
        provider, client.model, judge_provider, judge_client.model,
    )
    provenance = {
        "answerer": {"provider": provider, "model": client.model},
        "measurement": {"provider": judge_provider, "model": judge_client.model},
    }
    return client, judge_client, provenance
