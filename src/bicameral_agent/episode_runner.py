"""End-to-end episode runner wiring all components into a single orchestration loop.

Combines ConsciousLoop, Controller, SimulatedUser, SignalClassifier,
ConversationLogger, tool primitives, and ContextQueue to execute a complete
research QA episode from start to finish.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Protocol, runtime_checkable

from bicameral_agent.assumption_auditor import AssumptionAuditor
from bicameral_agent.config import HyperConfig
from bicameral_agent.conscious_loop import AssistantResponse, ConsciousLoop
from bicameral_agent.context_refresher import ContextRefresher
from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.encoder import StateEncoder
from bicameral_agent.gap_scanner import ResearchGapScanner
from bicameral_agent.heuristic_controller import Action, DecisionLog, FullState, TOOL_IDS
from bicameral_agent.llm_output import DegradationCounter, count_degradations
from bicameral_agent.logger import ConversationLogger
from bicameral_agent.model_client import ModelClient
from bicameral_agent.queue import ContextQueue, InterruptConfig
from bicameral_agent.schema import (
    Episode,
    Message,
    UserEvent,
    UserEventType,
    estimate_text_tokens,
)
from bicameral_agent.signal_classifier import SignalClassifier
from bicameral_agent.simulated_user import ActionType, Patience, SimulatedUser, Strictness
from bicameral_agent.token_estimator import ContextFeatures
from bicameral_agent.tool_latency import ToolLatencyModel
from bicameral_agent.cost_tracker import CostBudgetExceeded, CostTrackedClient, CostTracker
from bicameral_agent.tool_primitive import BudgetExceededError, TokenBudget
from bicameral_agent.verifiers import build_verifier


class InjectionMode(enum.Enum):
    """Context injection strategy for the episode runner.

    SYNCHRONOUS: Same-turn regeneration after tool deposit.
    BREAKPOINT: Queue consumed at next turn (default, backward-compatible).
    INTERRUPT: Queue with threshold-triggered regeneration.
    """

    SYNCHRONOUS = "synchronous"
    BREAKPOINT = "breakpoint"
    INTERRUPT = "interrupt"

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's research question thoroughly "
    "and accurately. Use any context provided to improve your answer. Be specific "
    "and cite evidence when available."
)

_DEFAULT_TOKEN_BUDGET = TokenBudget(
    max_calls=10,
    max_input_tokens=50_000,
    max_output_tokens=20_000,
)



@runtime_checkable
class Controller(Protocol):
    """Protocol for episode controllers that decide tool invocations."""

    def decide(self, state: FullState) -> Action: ...

    @property
    def decisions(self) -> list[DecisionLog]: ...


@dataclasses.dataclass(frozen=True)
class EpisodeConfig:
    """Configuration for an episode run."""

    max_turns: int = 25
    tool_token_budget: TokenBudget = _DEFAULT_TOKEN_BUDGET
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    thinking_level: str = "medium"
    temperature: float | None = None
    interrupt_config: InterruptConfig | None = None
    queue_expiry_turns: int | None = None
    patience: Patience = Patience.MEDIUM
    strictness: Strictness = Strictness.MEDIUM
    score_episode: bool = False
    metric: str = "llm_judge"
    """Verifier-registry key used when score_episode is set (Issue #56).

    Replaces the former ``use_lexical_scorer`` boolean: ``metric="lexical"``
    is the old True, ``"llm_judge"`` (default) the old False.
    """
    injection_mode: InjectionMode = InjectionMode.BREAKPOINT
    persistent_injection: bool = True


class EpisodeRunner:
    """Orchestrates a complete research QA episode from start to finish.

    Wires together ConsciousLoop, Controller, SimulatedUser, SignalClassifier,
    ConversationLogger, tool primitives, and ContextQueue.

    Per-role clients (issue #53): ``client`` drives the system under test --
    the answerer (ConsciousLoop) *and* the subconscious tools, which are part
    of that system and deliberately follow the answerer. ``judge_client`` and
    ``sim_user_client`` drive the measurement apparatus (TaskScorer LLM judge
    and SimulatedUser); hold them fixed while ``client`` varies so
    cross-model comparisons stay on one judging scale. Both default to
    ``client`` for back-compat.
    """

    def __init__(
        self,
        client: ModelClient,
        config: EpisodeConfig | None = None,
        hyper_config: HyperConfig | None = None,
        cost_tracker: CostTracker | None = None,
        judge_client: ModelClient | None = None,
        sim_user_client: ModelClient | None = None,
    ) -> None:
        self._client = client
        self._judge_client = judge_client if judge_client is not None else client
        self._sim_user_client = (
            sim_user_client if sim_user_client is not None else client
        )
        if config is not None:
            self._config = config
        elif hyper_config is not None:
            self._config = hyper_config.to_episode_config()
        else:
            self._config = EpisodeConfig()
        self._hyper_config = hyper_config
        self._cost_tracker = cost_tracker

    def run_episode(
        self,
        task: ResearchQATask,
        controller: Controller,
    ) -> Episode:
        """Execute a complete episode for the given task.

        Parameters
        ----------
        task:
            The research QA task to work on.
        controller:
            Controller that decides which tools to invoke each turn.

        Returns
        -------
        Episode
            A validated Episode capturing the full conversation.
        """
        # Episode-scoped degradation counting (issue #82): every
        # structured-output site funnels through safe_parse_json, which
        # reports to this context-local counter -- no counter threading
        # through the six components (sim-user, tools, scorer/verifiers).
        # Context-local (issue #91) so concurrent episodes, each run in its
        # own contextvars context, never count each other's degradations.
        with count_degradations() as degradations:
            return self._run_episode(task, controller, degradations)

    def _run_episode(
        self,
        task: ResearchQATask,
        controller: Controller,
        degradations: DegradationCounter,
    ) -> Episode:
        cfg = self._config

        # Cost tracking: reset episode, wrap every role's client so judge and
        # sim-user calls hit the same budget and accounting as the answerer.
        active_client: ModelClient = self._client
        judge_client: ModelClient = self._judge_client
        sim_user_client: ModelClient = self._sim_user_client
        if self._cost_tracker is not None:
            self._cost_tracker.reset_episode()
            active_client = CostTrackedClient(self._client, self._cost_tracker)

            def _tracked(role_client: ModelClient) -> ModelClient:
                """Reuse the answerer's wrapper when the role shares its client."""
                if role_client is self._client:
                    return active_client
                return CostTrackedClient(role_client, self._cost_tracker)

            judge_client = _tracked(self._judge_client)
            sim_user_client = _tracked(self._sim_user_client)

        # Initialize components
        queue = ContextQueue()
        log = ConversationLogger(metadata={"task_id": task.task_id})
        loop = ConsciousLoop(
            active_client,
            queue,
            system_prompt=cfg.system_prompt,
            thinking_level=cfg.thinking_level,
            temperature=cfg.temperature,
            persistent_injection=cfg.persistent_injection,
        )
        sim_user = SimulatedUser(
            client=sim_user_client,
            patience=cfg.patience,
            strictness=cfg.strictness,
        )
        encoder = StateEncoder()
        latency_model = ToolLatencyModel()

        tools = {
            TOOL_IDS[Action.SCANNER]: ResearchGapScanner(),
            TOOL_IDS[Action.AUDITOR]: AssumptionAuditor(),
            TOOL_IDS[Action.REFRESHER]: ContextRefresher(),
        }

        # Tracking state
        schema_messages: list[Message] = []
        user_events: list[UserEventType] = []
        pending_injection_indices: list[int] = []
        interrupt_count = 0
        expired_count = 0

        user_message = task.question

        for turn in range(1, cfg.max_turns + 1):
            # (a) Log user message
            user_token_count = estimate_text_tokens(user_message)
            log.log_message("user", user_message, user_token_count)

            # (b) Track in schema_messages
            schema_messages.append(
                Message(
                    role="user",
                    content=user_message,
                    timestamp_ms=0,
                    token_count=user_token_count,
                )
            )

            # (c) Expire stale queue items
            expired_count += len(queue.expire_stale(turn))

            # (c2) Snapshot queue state before run_turn's breakpoint drain.
            # Taken after the drain it is always empty, which made the
            # controller's queue_depth feature and the avg_queue_depth
            # metric structurally zero (issue #45); pre-drain it reflects
            # deposits pending from earlier turns.
            queue_snapshot = queue.get_state()

            # (d) Run conscious loop turn
            try:
                response: AssistantResponse = loop.run_turn(user_message)
            except CostBudgetExceeded:
                logger.warning(
                    "CostBudgetExceeded on turn %d, ending episode",
                    turn,
                )
                break

            # (e) Log assistant message at generation time so that any tool
            # events later this turn come after it in reconstructed timelines.
            log.log_message("assistant", response.content, response.output_tokens)

            # (f) Mark pending injections as consumed
            if response.context_injected:
                for inj_idx in pending_injection_indices:
                    log.log_injection_consumed(inj_idx, turn)
                pending_injection_indices.clear()

            # (g) Classify signals (using temporary schema_messages + response)
            temp_messages = schema_messages + [
                Message(
                    role="assistant",
                    content=response.content,
                    timestamp_ms=0,
                    token_count=response.output_tokens,
                )
            ]
            schema_user_events = [
                UserEvent(event_type=evt, timestamp_ms=0) for evt in user_events
            ]
            signals = SignalClassifier.classify(temp_messages, schema_user_events)

            # (h) Build FullState
            total_tokens = sum(m.token_count for m in temp_messages)
            ctx_features = ContextFeatures(
                conversation_length_tokens=total_tokens,
                conversation_turn_count=turn,
            )

            predicted_latencies = {
                tool_id: latency_model.predict_tool_duration(tool_id, ctx_features).mean_ms
                for tool_id in tools
            }

            state = FullState(
                turn_number=turn,
                stop_count=signals.stop_count.value,
                followup_type=signals.followup_type,
                queue_depth=queue_snapshot.depth,
                executing_tools=(),
                predicted_latencies=predicted_latencies,
            )

            # (i) Controller decides. Controllers that encode the full
            # episode context at decision time (LearnedPolicyController,
            # issue #29) expose an ``observe_episode`` hook; give them a
            # snapshot of everything logged so far, which at this point
            # ends with the current assistant message — exactly the last
            # decision point a post-hoc EpisodeReplayer would yield.
            observe = getattr(controller, "observe_episode", None)
            if observe is not None:
                observe(log.snapshot())
            action = controller.decide(state)

            # Set when a cost-budget trip mid-turn must end the episode after
            # the assistant message is recorded (mirrors the run_turn path).
            end_episode = False

            # (j) Execute tool if action != DO_NOTHING
            if action != Action.DO_NOTHING:
                tool_id = TOOL_IDS[action]
                tool = tools[tool_id]
                reasoning_state = encoder.encode(
                    temp_messages,
                    queue_state=queue_snapshot,
                    latency_predictions=predicted_latencies,
                    turn_number=turn,
                    max_turns=cfg.max_turns,
                )

                inv_idx = log.log_tool_invocation(tool_id, 0, turn=turn)
                try:
                    result = tool.execute(
                        conversation_history=temp_messages,
                        reasoning_state=reasoning_state,
                        budget=cfg.tool_token_budget,
                        client=active_client,
                    )
                    log.log_tool_completion(
                        inv_idx,
                        result.metadata.tokens_consumed,
                        result_deposited=result.queue_deposit is not None,
                    )

                    # Deposit to queue and log context injection
                    if result.queue_deposit is not None:
                        deposit_update: dict = {"enqueued_at_turn": turn}
                        if cfg.queue_expiry_turns is not None:
                            deposit_update["expiry_turns"] = cfg.queue_expiry_turns
                        deposit = result.queue_deposit.model_copy(
                            update=deposit_update
                        )
                        queue.enqueue(deposit)
                        inj_idx = log.log_context_injection(deposit)
                        pending_injection_indices.append(inj_idx)

                        # (j2) Mode-specific handling after tool deposit
                        def _drain_and_regenerate():
                            nonlocal response
                            ctx = queue.drain_at_breakpoint()
                            if ctx is not None:
                                regen = loop.regenerate_with_context(ctx)
                                discarded = (
                                    response.input_tokens + response.output_tokens
                                )
                                log.log_wasted_tokens(discarded)
                                response = regen
                                log.replace_last_message(
                                    regen.content, regen.output_tokens
                                )
                                for idx in pending_injection_indices:
                                    log.log_injection_consumed(idx, turn)
                                pending_injection_indices.clear()

                        if cfg.injection_mode == InjectionMode.SYNCHRONOUS:
                            _drain_and_regenerate()

                        elif cfg.injection_mode == InjectionMode.INTERRUPT:
                            int_cfg = cfg.interrupt_config or InterruptConfig()
                            if queue.check_interrupt_threshold(int_cfg):
                                interrupt_count += 1
                                _drain_and_regenerate()

                except BudgetExceededError:
                    logger.warning(
                        "BudgetExceededError for tool %s on turn %d",
                        tool_id,
                        turn,
                    )
                    log.log_tool_completion(
                        inv_idx, 0, result_deposited=False, budget_exceeded=True
                    )
                except CostBudgetExceeded:
                    logger.warning(
                        "CostBudgetExceeded in tool %s on turn %d, ending episode",
                        tool_id,
                        turn,
                    )
                    log.log_tool_completion(
                        inv_idx, 0, result_deposited=False, budget_exceeded=True
                    )
                    end_episode = True

            schema_messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    timestamp_ms=0,
                    token_count=response.output_tokens,
                )
            )

            if end_episode:
                break

            # (k) Simulated user responds. schema_messages already contains
            # the current exchange, so the runner's turn is passed explicitly.
            try:
                user_action = sim_user.respond(
                    task, response.content, schema_messages, turn_number=turn
                )
            except CostBudgetExceeded:
                logger.warning(
                    "CostBudgetExceeded in simulated user on turn %d, ending episode",
                    turn,
                )
                break

            # (l) STOP
            if user_action.action_type == ActionType.STOP:
                log.log_user_event(UserEventType.STOP)
                user_events.append(UserEventType.STOP)
                break

            # (m) TASK_COMPLETE
            if user_action.action_type == ActionType.TASK_COMPLETE:
                log.log_user_event(UserEventType.TASK_COMPLETE)
                user_events.append(UserEventType.TASK_COMPLETE)
                break

            # (n) FOLLOW_UP
            if user_action.action_type == ActionType.FOLLOW_UP:
                log.log_user_event(
                    UserEventType.FOLLOW_UP,
                    metadata={"followup_type": user_action.followup_type.value},
                )
                user_events.append(UserEventType.FOLLOW_UP)
                user_message = user_action.message

        # Store metadata
        log.set_metadata("interrupt_count", interrupt_count)
        log.set_metadata("expired_queue_items", expired_count)
        log.set_metadata("wasted_tokens", log.wasted_tokens)
        log.set_metadata("injection_mode", cfg.injection_mode.value)
        if self._hyper_config is not None:
            log.set_metadata("hyperparameters", self._hyper_config.to_dict())

        # Score if requested. LLM-backed verifiers use the (cost-tracked)
        # judge client, so their calls count toward budgets and episode cost.
        quality_score: float | None = None
        if cfg.score_episode:
            last_assistant = next(
                (m.content for m in reversed(schema_messages) if m.role == "assistant"),
                None,
            )
            if last_assistant is not None:
                verifier = build_verifier(cfg.metric, client=judge_client)
                try:
                    task_score = verifier.score(task, last_assistant)
                    quality_score = task_score.overall
                    log.set_metadata(
                        "verification",
                        {"metric": cfg.metric, "detail": task_score.detail},
                    )
                except CostBudgetExceeded:
                    logger.warning(
                        "CostBudgetExceeded while scoring, leaving quality_score unset"
                    )

        # Capture episode cost after scoring so judge calls are included.
        if self._cost_tracker is not None:
            episode_cost = self._cost_tracker.get_episode_cost()
            log.set_metadata("episode_cost", {
                "input_cost": episode_cost.input_cost,
                "output_cost": episode_cost.output_cost,
                "total": episode_cost.total,
                "call_count": episode_cost.call_count,
            })

        # After scoring, so judge/verifier degradations are included. Always
        # present (empty dict on a clean episode) so runs can report a rate.
        log.set_metadata("parse_degradations", dict(degradations.counts))

        return log.finalize(quality_score)
