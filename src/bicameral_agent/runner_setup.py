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

from bicameral_agent.brave_search import BraveSearchProvider
from bicameral_agent.config import SEARCH_PROVIDER_NAMES, HyperConfig, ModelConfig
from bicameral_agent.gap_scanner import SearchProvider
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
    ``--judge-model``, ``--episode-budget`` and ``--search-provider``;
    consume them with :func:`resolve_runner_clients` and
    :func:`resolve_search_provider`.
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
    parser.add_argument("--search-provider", choices=list(SEARCH_PROVIDER_NAMES),
                        default=None,
                        help="Search backend for the research gap scanner: "
                             "'mock' (built-in snippets, offline) or 'brave' "
                             "(Brave Web Search API; requires BRAVE_API_KEY). "
                             "Overrides the config's [tools] search_provider; "
                             "default mock.")


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


def _search_provider_name(args: argparse.Namespace, hyper: HyperConfig) -> str:
    """Resolve the gap scanner's search backend name: CLI > config > mock."""
    return args.search_provider or hyper.tools.search_provider


def resolve_search_provider(
    args: argparse.Namespace, hyper: HyperConfig
) -> SearchProvider | None:
    """Resolve the gap scanner's search backend: CLI > config > mock.

    Returns ``None`` for "mock" (the scanner constructs its own
    ``MockSearchProvider`` default, keeping offline runs untouched). For
    "brave", returns one ``BraveSearchProvider`` shared by every episode,
    so its client-side ~1 req/s throttle is process-wide under
    ``--parallel-episodes``; a missing ``BRAVE_API_KEY`` fails fast here,
    at script startup, rather than mid-run.
    """
    name = _search_provider_name(args, hyper)
    if name == "brave":
        logger.info("Gap scanner search backend: brave")
        return BraveSearchProvider()
    return None


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


def effective_hyper_config(
    args: argparse.Namespace,
    hyper: HyperConfig,
    provenance: dict[str, dict[str, str]],
) -> HyperConfig:
    """Overlay the CLI-resolved runtime values onto *hyper* (issue #103).

    ``EpisodeRunner`` stamps ``hyper_config.to_dict()`` into every episode's
    ``metadata.hyperparameters``, so handing it the loaded config verbatim
    records the config-file defaults whenever a CLI flag overrides them.
    Pass this copy -- answerer ``[model]``, ``[measurement_model]``,
    ``run.parallel_episodes`` and ``tools.search_provider`` replaced by the
    resolved values -- as the runner's ``hyper_config`` instead.
    ``provenance`` is the mapping returned by :func:`resolve_runner_clients`.
    """
    answerer = provenance["answerer"]
    measurement = provenance["measurement"]
    return hyper.model_copy(update={
        "model": hyper.model.model_copy(
            update={"provider": answerer["provider"], "name": answerer["model"]}
        ),
        "measurement_model": (hyper.measurement_model or ModelConfig()).model_copy(
            update={"provider": measurement["provider"], "name": measurement["model"]}
        ),
        "run": hyper.run.model_copy(
            update={"parallel_episodes": resolve_parallel_episodes(args, hyper)}
        ),
        "tools": hyper.tools.model_copy(
            update={"search_provider": _search_provider_name(args, hyper)}
        ),
    })
