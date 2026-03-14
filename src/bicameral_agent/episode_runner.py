"""End-to-end episode runner wiring all components into a single orchestration loop.

Combines ConsciousLoop, Controller, SimulatedUser, SignalClassifier,
ConversationLogger, tool primitives, and ContextQueue to execute a complete
research QA episode from start to finish.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import random
import time
from typing import Protocol, runtime_checkable


class InjectionMode(enum.Enum):
    """Context injection strategy for the episode runner.

    SYNCHRONOUS: Same-turn regeneration after tool deposit.
    BREAKPOINT: Queue consumed at next turn (default, backward-compatible).
    INTERRUPT: Queue with threshold-triggered regeneration.
    """

    SYNCHRONOUS = "synchronous"
    BREAKPOINT = "breakpoint"
    INTERRUPT = "interrupt"

from bicameral_agent.assumption_auditor import AssumptionAuditor
from bicameral_agent.conscious_loop import AssistantResponse, ConsciousLoop
from bicameral_agent.context_refresher import ContextRefresher
from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.encoder import StateEncoder
from bicameral_agent.gap_scanner import ResearchGapScanner
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.heuristic_controller import Action, DecisionLog, FullState
from bicameral_agent.logger import ConversationLogger
from bicameral_agent.queue import ContextQueue, InterruptConfig
from bicameral_agent.schema import Episode, Message, UserEvent, UserEventType
from bicameral_agent.scorer import LexicalScorer, TaskScorer
from bicameral_agent.signal_classifier import SignalClassifier
from bicameral_agent.simulated_user import ActionType, Patience, SimulatedUser, Strictness
from bicameral_agent.token_estimator import ContextFeatures
from bicameral_agent.tool_latency import ToolLatencyModel
from bicameral_agent.tool_primitive import BudgetExceededError, TokenBudget

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

_TOOL_ID_MAP = {
    Action.SCANNER: "research_gap_scanner",
    Action.AUDITOR: "assumption_auditor",
    Action.REFRESHER: "context_refresher",
}


@runtime_checkable
class Controller(Protocol):
    """Protocol for episode controllers that decide tool invocations."""

    def decide(self, state: FullState) -> Action: ...

    @property
    def decisions(self) -> list[DecisionLog]: ...


class RandomController:
    """Controller that randomly selects tool actions.

    With probability ``action_probability``, picks uniformly from
    {SCANNER, AUDITOR, REFRESHER}. Respects queue depth guard (depth >= 3).
    Uses ``random.Random(seed)`` for reproducibility.
    """

    def __init__(
        self,
        action_probability: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self._action_probability = action_probability
        self._rng = random.Random(seed)
        self._decisions: list[DecisionLog] = []

    def decide(self, state: FullState) -> Action:
        # Queue depth guard (matches heuristic controller rule 7)
        if state.queue_depth >= 3:
            action = Action.DO_NOTHING
        elif self._rng.random() < self._action_probability:
            action = self._rng.choice(
                [Action.SCANNER, Action.AUDITOR, Action.REFRESHER]
            )
        else:
            action = Action.DO_NOTHING

        self._decisions.append(
            DecisionLog(
                action=action,
                rule_fired=0,
                state=state,
                timestamp_ms=time.time() * 1000,
            )
        )
        return action

    @property
    def decisions(self) -> list[DecisionLog]:
        return list(self._decisions)


@dataclasses.dataclass(frozen=True)
class EpisodeConfig:
    """Configuration for an episode run."""

    max_turns: int = 25
    tool_token_budget: TokenBudget = _DEFAULT_TOKEN_BUDGET
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    thinking_level: str = "medium"
    interrupt_config: InterruptConfig | None = None
    patience: Patience = Patience.MEDIUM
    strictness: Strictness = Strictness.MEDIUM
    score_episode: bool = False
    use_lexical_scorer: bool = False
    injection_mode: InjectionMode = InjectionMode.BREAKPOINT


class EpisodeRunner:
    """Orchestrates a complete research QA episode from start to finish.

    Wires together ConsciousLoop, Controller, SimulatedUser, SignalClassifier,
    ConversationLogger, tool primitives, and ContextQueue.
    """

    def __init__(
        self,
        client: GeminiClient,
        config: EpisodeConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or EpisodeConfig()

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
        cfg = self._config

        # Initialize components
        queue = ContextQueue()
        log = ConversationLogger(metadata={"task_id": task.task_id})
        loop = ConsciousLoop(
            self._client,
            queue,
            system_prompt=cfg.system_prompt,
            thinking_level=cfg.thinking_level,
            interrupt_config=cfg.interrupt_config,
        )
        sim_user = SimulatedUser(
            client=self._client,
            patience=cfg.patience,
            strictness=cfg.strictness,
        )
        encoder = StateEncoder()
        latency_model = ToolLatencyModel()

        tools = {
            "research_gap_scanner": ResearchGapScanner(),
            "assumption_auditor": AssumptionAuditor(),
            "context_refresher": ContextRefresher(),
        }

        # Tracking state
        schema_messages: list[Message] = []
        user_events: list[UserEventType] = []
        pending_injection_indices: list[int] = []
        interrupt_count = 0

        user_message = task.question

        for turn in range(1, cfg.max_turns + 1):
            # (a) Log user message
            user_token_count = len(user_message.split())
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
            queue.expire_stale(turn)

            # (d) Run conscious loop turn
            response: AssistantResponse = loop.run_turn(user_message)

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
                queue_depth=queue.get_state().depth,
                executing_tools=(),
                predicted_latencies=predicted_latencies,
            )

            # (i) Controller decides
            action = controller.decide(state)

            # (j) Execute tool if action != DO_NOTHING
            if action != Action.DO_NOTHING:
                tool_id = _TOOL_ID_MAP[action]
                tool = tools[tool_id]
                reasoning_state = encoder.encode(temp_messages)

                inv_idx = log.log_tool_invocation(tool_id, 0)
                try:
                    result = tool.execute(
                        conversation_history=temp_messages,
                        reasoning_state=reasoning_state,
                        budget=cfg.tool_token_budget,
                        client=self._client,
                    )
                    log.log_tool_completion(
                        inv_idx,
                        result.metadata.tokens_consumed,
                        result_deposited=result.queue_deposit is not None,
                    )

                    # Deposit to queue and log context injection
                    if result.queue_deposit is not None:
                        deposit = result.queue_deposit.model_copy(
                            update={"enqueued_at_turn": turn}
                        )
                        queue.enqueue(deposit)
                        inj_idx = log.log_context_injection(
                            content=deposit.content,
                            source_tool_id=deposit.source_tool_id,
                            priority=int(deposit.priority),
                            token_count=deposit.token_count,
                        )
                        pending_injection_indices.append(inj_idx)

                        # (j2) Mode-specific handling after tool deposit
                        def _drain_and_regenerate():
                            nonlocal response
                            ctx = queue.drain_at_breakpoint()
                            if ctx is not None:
                                regen = loop.regenerate_with_context(ctx)
                                queue.report_wasted_tokens(
                                    response.input_tokens + response.output_tokens
                                )
                                response = regen
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
                    log.log_tool_completion(inv_idx, 0, result_deposited=False)

            # (e') Log assistant message (deferred until after mode handling)
            log.log_message("assistant", response.content, response.output_tokens)

            schema_messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    timestamp_ms=0,
                    token_count=response.output_tokens,
                )
            )

            # (k) Simulated user responds
            user_action = sim_user.respond(
                task, response.content, schema_messages
            )

            # (l) STOP
            if user_action.action_type == ActionType.STOP:
                log.log_user_event(UserEventType.STOP)
                user_events.append(UserEventType.STOP)
                break

            # (m) TASK_COMPLETE
            if user_action.action_type == ActionType.TASK_COMPLETE:
                break

            # (n) FOLLOW_UP
            if user_action.action_type == ActionType.FOLLOW_UP:
                log.log_user_event(UserEventType.FOLLOW_UP)
                user_events.append(UserEventType.FOLLOW_UP)
                user_message = user_action.message

        # Store metadata
        log.set_metadata("interrupt_count", interrupt_count)
        log.set_metadata("injection_mode", cfg.injection_mode.value)

        # Score if requested
        quality_score: float | None = None
        if cfg.score_episode:
            last_assistant = next(
                (m.content for m in reversed(schema_messages) if m.role == "assistant"),
                None,
            )
            if last_assistant is not None:
                scorer = LexicalScorer() if cfg.use_lexical_scorer else TaskScorer(client=self._client)
                quality_score = scorer.score(task, last_assistant).overall

        return log.finalize(quality_score)
