"""MCTS training loop: the self-improvement cycle of issue #29.

Each :meth:`MCTSTrainer.run_iteration` performs:

1. **Collect** ``n_episodes`` episodes with the current policy via
   :class:`~bicameral_agent.episode_runner.EpisodeRunner` +
   :class:`~bicameral_agent.learned_controller.LearnedPolicyController`
   (or consumes pre-collected episodes for offline runs).
2. **Targets**: run :class:`~bicameral_agent.mcts.MCTSEngine` search on
   every decision-point state; the normalized visit counts are the
   improved policy targets.
3. **Train**: policy head cross-entropy against the MCTS visit
   distributions (soft targets), value head MSE against the pipeline's
   discounted returns.
4. Optionally **retrain the transition model** on all stored examples.
5. **Evaluate** the updated policy against the heuristic baseline on
   held-out tasks (requires a live runner; skipped offline).
6. **Checkpoint** the networks and metrics; append to the metrics
   history JSON.

Live acceptance criteria (monotonic eval improvement, KL shift from the
heuristic, no catastrophic forgetting, entropy convergence, budget) are
data/LLM-dependent: this module implements their *measurement* — every
iteration records training loss, policy entropy, value accuracy, eval
scores (overall and per difficulty), and KL/agreement vs the heuristic —
so the criteria can be verified from ``metrics_history.json`` once live
training data exists (post-#46). Verification itself is pending.

Determinism: for a fixed config seed, fixed initial weights, and a fixed
episode list, an iteration is fully reproducible — MCTS noise, batch
shuffling, and collection-time sampling all draw from seeded generators,
and no torch RNG is consumed (the networks have no stochastic layers).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from bicameral_agent.concurrency import submit_in_context
from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.episode_runner import EpisodeRunner
from bicameral_agent.heuristic_controller import FullState, HeuristicController
from bicameral_agent.learned_controller import LearnedPolicyController
from bicameral_agent.mcts import MCTSEngine
from bicameral_agent.policy_value_net import NUM_ACTIONS, PolicyValueNetwork
from bicameral_agent.pretrain import evaluate_policy_value
from bicameral_agent.replay import EpisodeReplayer
from bicameral_agent.schema import Episode
from bicameral_agent.serialization import episodes_to_parquet
from bicameral_agent.signal_classifier import SignalClassifier
from bicameral_agent.training_data_store import TrainingDataStore
from bicameral_agent.training_pipeline import (
    _ACTION_INDEX,  # shared action indexing (cross-checked with ACTION_ORDER)
    STATE_DIM,
    TrainingDataPipeline,
    TrainingExample,
)
from bicameral_agent.transition_model import (
    TransitionModel,
    TransitionTrainingConfig,
    fit_transition_model,
    split_examples,
)

logger = logging.getLogger(__name__)

_HISTORY_FILENAME = "metrics_history.json"
_EPS = 1e-8  # smoothing for entropy logs
_KL_SMOOTHING = 1e-3
"""Additive smoothing for the policy/heuristic action marginals before the
KL divergence. Larger than machine epsilon on purpose: actions the
deterministic heuristic never takes would otherwise contribute
``log(p / eps)`` terms that dwarf the actual distribution shift the
"KL from heuristic increases" AC tracks."""


@dataclasses.dataclass(frozen=True, slots=True)
class MCTSTrainerConfig:
    """Hyperparameters for :class:`MCTSTrainer`.

    ``max_turns`` must match the ``EpisodeConfig.max_turns`` used by the
    runner so the pipeline's completion-fraction features are consistent
    between collection and training.

    ``parallel_episodes`` bounds how many collection episodes run
    concurrently (issue #91; 1 = sequential). Per-episode seeds and the
    returned episode order are index-based, so parallelism does not change
    what is collected — only the wall clock.
    """

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    value_loss_weight: float = 0.5
    collect_with_search: bool = True
    collect_temperature: float = 1.0
    collect_root_noise: bool = True
    target_root_noise: bool = True
    eval_with_search: bool = True
    retrain_transition: bool = False
    transition_config: TransitionTrainingConfig | None = None
    train_ratio: float = 0.8
    max_turns: int = 25
    seed: int = 0
    parallel_episodes: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Per-iteration metrics; JSON-serializable via :meth:`to_dict`.

    ``eval_score`` / ``heuristic_eval_score`` / ``eval_scores_by_difficulty``
    are *None* when no evaluation ran (offline mode). ``holdout`` is the
    :func:`~bicameral_agent.pretrain.evaluate_policy_value` dict on the
    episode-grouped validation split (*None* with fewer than 2 episodes).
    """

    iteration: int
    n_episodes: int
    n_examples: int
    epoch_losses: list[dict[str, float]]
    train_loss: float
    train_policy_loss: float
    train_value_loss: float
    policy_entropy: float
    value_mse: float
    value_correlation: float | None
    holdout: dict | None
    kl_from_heuristic: float
    heuristic_agreement: float
    mcts_root_value_mean: float
    eval_score: float | None
    heuristic_eval_score: float | None
    eval_scores_by_difficulty: dict[str, float] | None
    transition_metrics: dict | None
    collect_seconds: float
    train_seconds: float
    eval_seconds: float

    def to_dict(self) -> dict:
        """Plain-dict form for JSON serialization."""
        return dataclasses.asdict(self)


class MCTSTrainer:
    """Self-improvement training loop over learned policy/value/transition models.

    Parameters
    ----------
    policy_value_net:
        The network being trained (``input_dim`` must be ``STATE_DIM``).
    transition_model:
        Environment model for MCTS search; replaced in place when
        ``config.retrain_transition`` is set.
    checkpoint_dir:
        Root directory for per-iteration checkpoints, the metrics history,
        and (by default) the training-data store.
    runner, train_tasks, eval_tasks:
        Live-collection dependencies. ``runner`` drives real episodes (its
        client makes paid LLM calls); ``train_tasks`` feed collection and
        ``eval_tasks`` the held-out evaluation. All optional: offline
        iterations pass pre-collected ``episodes`` to
        :meth:`run_iteration` instead.
    heuristic_factory:
        Zero-arg factory for the heuristic baseline controller (defaults
        to a stock :class:`HeuristicController`).
    pipeline:
        Episode-to-example pipeline; defaults to one built with
        ``config.max_turns``. The same pipeline instance is handed to the
        learned controllers so serve-time encoding matches training.
    store:
        Append-only example store; defaults to ``checkpoint_dir / "store"``.
    """

    def __init__(
        self,
        policy_value_net: PolicyValueNetwork,
        transition_model: TransitionModel,
        *,
        checkpoint_dir: str | Path,
        config: MCTSTrainerConfig | None = None,
        runner: EpisodeRunner | None = None,
        train_tasks: Sequence[ResearchQATask] | None = None,
        eval_tasks: Sequence[ResearchQATask] | None = None,
        heuristic_factory: Callable[[], HeuristicController] | None = None,
        pipeline: TrainingDataPipeline | None = None,
        store: TrainingDataStore | None = None,
    ) -> None:
        if policy_value_net.input_dim != STATE_DIM:
            msg = (
                f"policy_value_net.input_dim must be STATE_DIM ({STATE_DIM}), "
                f"got {policy_value_net.input_dim}"
            )
            raise ValueError(msg)
        self._policy = policy_value_net
        self._transition = transition_model
        self._config = config or MCTSTrainerConfig()
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._train_tasks = list(train_tasks) if train_tasks else []
        self._eval_tasks = list(eval_tasks) if eval_tasks else []
        self._heuristic_factory = heuristic_factory or HeuristicController
        self._pipeline = pipeline or TrainingDataPipeline(max_turns=self._config.max_turns)
        self._store = store or TrainingDataStore(self._checkpoint_dir / "store")
        self._history_path = self._checkpoint_dir / _HISTORY_FILENAME
        self._iteration = len(self._load_history())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def iteration(self) -> int:
        """Index of the next iteration (== completed iterations so far)."""
        return self._iteration

    @property
    def transition_model(self) -> TransitionModel:
        """The current transition model (replaced by retraining)."""
        return self._transition

    def run_iteration(
        self,
        n_episodes: int,
        n_simulations: int,
        episodes: Sequence[Episode] | None = None,
    ) -> TrainingMetrics:
        """Run one collect → target → train → evaluate → checkpoint cycle.

        Parameters
        ----------
        n_episodes:
            Number of episodes to collect with the current policy.
            Ignored when ``episodes`` is given.
        n_simulations:
            MCTS budget per decision point, used for target generation
            and (per config) for search-based collection and evaluation.
        episodes:
            Pre-collected episodes for offline iterations. When given, no
            collection happens and ``runner``/``train_tasks`` are not
            required.

        Raises
        ------
        ValueError
            If a live collection is requested without a runner and train
            tasks, or if the episodes yield no training examples.
        """
        cfg = self._config
        iteration = self._iteration
        base_seed = cfg.seed + 1000 * iteration

        collect_start = time.perf_counter()
        collected_live = episodes is None
        if episodes is None:
            if self._runner is None or not self._train_tasks:
                msg = (
                    "live collection needs a runner and train_tasks; pass "
                    "pre-collected episodes for offline iterations"
                )
                raise ValueError(msg)
            episodes = self._collect(n_episodes, n_simulations, base_seed)
        episodes = list(episodes)
        collect_seconds = time.perf_counter() - collect_start
        if collected_live:
            # Live episodes are expensive; persist them for offline reuse.
            episodes_dir = self._checkpoint_dir / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            episodes_to_parquet(
                episodes, str(episodes_dir / f"iteration-{iteration:03d}.parquet")
            )

        examples = self._pipeline.process_episodes(episodes)
        if not examples:
            msg = "episodes produced no training examples"
            raise ValueError(msg)
        self._store.save_examples(
            examples,
            iteration,
            metadata={"controller": "learned_policy", "n_episodes": len(episodes)},
        )

        targets, root_values = self.build_mcts_targets(
            examples, n_simulations, seed=base_seed + 1
        )

        train_start = time.perf_counter()
        epoch_losses = self._train(examples, targets, seed=base_seed + 2)
        transition_metrics: dict | None = None
        if cfg.retrain_transition:
            transition_metrics = self._retrain_transition()
        train_seconds = time.perf_counter() - train_start

        # Post-training measurements on the collected decision points.
        states = torch.from_numpy(np.stack([e.state for e in examples])).float()
        policy_entropy = self._policy_entropy(states)
        kl_from_heuristic, heuristic_agreement = self._heuristic_comparison(
            episodes, states
        )
        value_metrics = self._value_metrics(examples)

        eval_start = time.perf_counter()
        eval_score, heuristic_eval_score, by_difficulty = self._evaluate(
            n_simulations, base_seed + 3
        )
        eval_seconds = time.perf_counter() - eval_start

        metrics = TrainingMetrics(
            iteration=iteration,
            n_episodes=len(episodes),
            n_examples=len(examples),
            epoch_losses=epoch_losses,
            train_loss=epoch_losses[-1]["total"],
            train_policy_loss=epoch_losses[-1]["policy"],
            train_value_loss=epoch_losses[-1]["value"],
            policy_entropy=policy_entropy,
            value_mse=value_metrics["value_mse"],
            value_correlation=value_metrics["value_correlation"],
            holdout=value_metrics["holdout"],
            kl_from_heuristic=kl_from_heuristic,
            heuristic_agreement=heuristic_agreement,
            mcts_root_value_mean=float(np.mean(root_values)),
            eval_score=eval_score,
            heuristic_eval_score=heuristic_eval_score,
            eval_scores_by_difficulty=by_difficulty,
            transition_metrics=transition_metrics,
            collect_seconds=collect_seconds,
            train_seconds=train_seconds,
            eval_seconds=eval_seconds,
        )

        self._checkpoint(metrics)
        self._iteration += 1
        return metrics

    def build_mcts_targets(
        self,
        examples: Sequence[TrainingExample],
        n_simulations: int,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """MCTS-improved action targets for every example state.

        Returns ``(distributions, root_values)`` with shapes
        ``(n, NUM_ACTIONS)`` and ``(n,)`` (float32). Deterministic for a
        fixed seed and fixed model weights.
        """
        engine = self._make_engine(seed)
        distributions = np.zeros((len(examples), NUM_ACTIONS), dtype=np.float32)
        root_values = np.zeros(len(examples), dtype=np.float32)
        for i, example in enumerate(examples):
            result = engine.search(
                example.state,
                num_simulations=n_simulations,
                add_root_noise=self._config.target_root_noise,
            )
            distributions[i] = result.action_distribution
            root_values[i] = result.root_value
        return distributions, root_values

    # ------------------------------------------------------------------
    # Collection / evaluation
    # ------------------------------------------------------------------

    def _make_engine(self, seed: int) -> MCTSEngine:
        return MCTSEngine(self._policy, self._transition, seed=seed)

    def _make_controller(
        self, *, training: bool, n_simulations: int, seed: int
    ) -> LearnedPolicyController:
        cfg = self._config
        with_search = cfg.collect_with_search if training else cfg.eval_with_search
        return LearnedPolicyController(
            self._policy,
            mcts_engine=self._make_engine(seed) if with_search else None,
            num_simulations=n_simulations,
            add_root_noise=training and cfg.collect_root_noise,
            temperature=cfg.collect_temperature if training else 0.0,
            pipeline=self._pipeline,
            seed=seed,
        )

    def _collect(
        self, n_episodes: int, n_simulations: int, base_seed: int
    ) -> list[Episode]:
        if n_episodes < 1:
            msg = f"n_episodes must be >= 1, got {n_episodes}"
            raise ValueError(msg)
        parallel = self._config.parallel_episodes
        if parallel < 1:
            msg = f"parallel_episodes must be >= 1, got {parallel}"
            raise ValueError(msg)

        def _run_one(i: int) -> Episode:
            task = self._train_tasks[i % len(self._train_tasks)]
            controller = self._make_controller(
                training=True, n_simulations=n_simulations, seed=base_seed + 10 + i
            )
            episode = self._runner.run_episode(task, controller)
            logger.info(
                "collected episode %d/%d (task %s)", i + 1, n_episodes, task.task_id
            )
            return episode

        if parallel == 1:
            return [_run_one(i) for i in range(n_episodes)]

        # Issue #91: bounded episode concurrency. Results are keyed by
        # episode index so the returned list is in collection order (and
        # per-episode seeds stay index-based) regardless of completion
        # order. Each episode runs in a copy of the caller's contextvars
        # context so per-episode counters/cost never cross episodes. Any
        # failure cancels unstarted episodes and propagates once in-flight
        # ones settle.
        episodes: dict[int, Episode] = {}
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                submit_in_context(pool, _run_one, i): i for i in range(n_episodes)
            }
            try:
                for future in as_completed(futures):
                    episodes[futures[future]] = future.result()
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
        return [episodes[i] for i in range(n_episodes)]

    def _evaluate(
        self, n_simulations: int, seed: int
    ) -> tuple[float | None, float | None, dict[str, float] | None]:
        """Mean quality of learned vs heuristic policy on held-out tasks."""
        if self._runner is None or not self._eval_tasks:
            return None, None, None

        def _mean(scores: list[float]) -> float | None:
            return float(np.mean(scores)) if scores else None

        learned_scores: list[float] = []
        by_difficulty: dict[str, list[float]] = {}
        for i, task in enumerate(self._eval_tasks):
            controller = self._make_controller(
                training=False, n_simulations=n_simulations, seed=seed + i
            )
            episode = self._runner.run_episode(task, controller)
            if episode.outcome.quality_score is not None:
                learned_scores.append(episode.outcome.quality_score)
                by_difficulty.setdefault(task.difficulty.value, []).append(
                    episode.outcome.quality_score
                )

        heuristic_scores: list[float] = []
        for task in self._eval_tasks:
            episode = self._runner.run_episode(task, self._heuristic_factory())
            if episode.outcome.quality_score is not None:
                heuristic_scores.append(episode.outcome.quality_score)

        difficulty_means = {
            k: float(np.mean(v)) for k, v in sorted(by_difficulty.items())
        }
        return (
            _mean(learned_scores),
            _mean(heuristic_scores),
            difficulty_means or None,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(
        self,
        examples: Sequence[TrainingExample],
        targets: np.ndarray,
        seed: int,
    ) -> list[dict[str, float]]:
        """Fit the policy head to MCTS targets and the value head to returns.

        Soft-target cross-entropy: ``-(target * log_softmax(logits)).sum()``
        averaged over the batch, plus ``value_loss_weight`` × MSE of the
        value head against the discounted returns.
        """
        cfg = self._config
        states = torch.from_numpy(np.stack([e.state for e in examples])).float()
        returns = torch.tensor(
            [e.discounted_return for e in examples], dtype=torch.float32
        )
        target_dists = torch.from_numpy(np.asarray(targets, dtype=np.float32))

        optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg.learning_rate)
        rng = np.random.default_rng(seed)
        n = len(examples)

        self._policy.train()
        epoch_losses: list[dict[str, float]] = []
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            sums = {"total": 0.0, "policy": 0.0, "value": 0.0}
            n_batches = 0
            for lo in range(0, n, cfg.batch_size):
                idx = torch.from_numpy(perm[lo : lo + cfg.batch_size])
                logits, values = self._policy._shared_forward(states[idx])  # noqa: SLF001
                policy_loss = -(
                    target_dists[idx] * F.log_softmax(logits, dim=-1)
                ).sum(dim=-1).mean()
                value_loss = F.mse_loss(values, returns[idx])
                loss = policy_loss + cfg.value_loss_weight * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sums["total"] += loss.item()
                sums["policy"] += policy_loss.item()
                sums["value"] += value_loss.item()
                n_batches += 1
            epoch_losses.append({k: v / n_batches for k, v in sums.items()})
        self._policy.eval()
        return epoch_losses

    def _retrain_transition(self) -> dict:
        """Refit the transition model on all stored examples."""
        all_examples = self._store.load_examples()
        result = fit_transition_model(
            all_examples,
            self._config.transition_config
            or TransitionTrainingConfig(seed=self._config.seed),
        )
        self._transition = result.model
        return {
            "n_train": result.n_train,
            "n_val": result.n_val,
            "state_mse_per_dim_mean": result.metrics.get("state_mse_per_dim_mean"),
            "reward_mse": result.metrics.get("reward_mse"),
            "reward_correlation": result.metrics.get("reward_correlation"),
        }

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _policy_entropy(self, states: torch.Tensor) -> float:
        """Mean policy entropy (nats) over the given states."""
        probs, _ = self._policy(states)
        entropy = -(probs * torch.log(probs + _EPS)).sum(dim=-1)
        return float(entropy.mean())

    def _value_metrics(self, examples: list[TrainingExample]) -> dict:
        """Value accuracy on the episode-grouped holdout split when possible."""
        holdout: dict | None = None
        eval_examples = examples
        if len({e.episode_id for e in examples}) >= 2:
            _, val = split_examples(examples, self._config.train_ratio, self._config.seed)
            holdout = evaluate_policy_value(
                self._policy, val, self._config.value_loss_weight
            )
            eval_examples = val
        in_sample = evaluate_policy_value(
            self._policy, list(eval_examples), self._config.value_loss_weight
        )
        return {
            "value_mse": in_sample["value_mse"],
            "value_correlation": in_sample["value_correlation"],
            "holdout": holdout,
        }

    @torch.no_grad()
    def _heuristic_comparison(
        self, episodes: Sequence[Episode], states: torch.Tensor
    ) -> tuple[float, float]:
        """(KL, agreement) between the trained policy and the heuristic.

        The heuristic decides on FullStates reconstructed from each
        episode's decision points; the policy's distributions come from
        the pipeline states for the same points (``states`` row order
        matches ``process_episodes``). KL is computed between the
        policy's mean action distribution and the heuristic's empirical
        action frequencies (with light smoothing); agreement is the
        fraction of decision points where the policy argmax matches the
        heuristic's action. Tracking measures the "distributions shift
        away from the heuristic" AC over iterations.
        """
        heuristic_actions: list[int] = []
        for episode in episodes:
            controller = self._heuristic_factory()
            for dp in EpisodeReplayer(episode).iter_decision_points():
                signals = SignalClassifier.classify(
                    list(dp.state.messages), list(dp.state.user_events)
                )
                full_state = FullState(
                    turn_number=dp.state.turn_number,
                    stop_count=signals.stop_count.value,
                    followup_type=signals.followup_type,
                    queue_depth=len(dp.state.pending_injections),
                    executing_tools=(),
                    predicted_latencies={},
                )
                heuristic_actions.append(_ACTION_INDEX[controller.decide(full_state)])

        probs, _ = self._policy(states)
        probs_np = probs.numpy()
        actions = np.asarray(heuristic_actions, dtype=np.int64)

        agreement = float(np.mean(probs_np.argmax(axis=1) == actions))

        policy_marginal = probs_np.mean(axis=0).astype(np.float64) + _KL_SMOOTHING
        policy_marginal /= policy_marginal.sum()
        heuristic_marginal = (
            np.bincount(actions, minlength=NUM_ACTIONS).astype(np.float64) / len(actions)
            + _KL_SMOOTHING
        )
        heuristic_marginal /= heuristic_marginal.sum()
        kl = float(np.sum(policy_marginal * np.log(policy_marginal / heuristic_marginal)))
        return kl, agreement

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint(self, metrics: TrainingMetrics) -> None:
        it_dir = self._checkpoint_dir / f"iteration-{metrics.iteration:03d}"
        it_dir.mkdir(parents=True, exist_ok=True)
        self._policy.save(it_dir / "policy_value.pt")
        self._transition.save(it_dir / "transition.pt")
        (it_dir / "metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
        )
        history = self._load_history()
        history.append(metrics.to_dict())
        self._history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def _load_history(self) -> list[dict]:
        if not self._history_path.exists():
            return []
        return json.loads(self._history_path.read_text(encoding="utf-8"))
