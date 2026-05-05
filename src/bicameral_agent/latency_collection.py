"""Latency data collection harness for tool primitives and the conscious loop.

Drives synthetic conversations of varied length through each tool primitive
(plus the conscious loop), captures per-API-call latency observations, and
feeds them back into the ToolLatencyModel. Persists observations as Parquet
for future re-training.

The collection harness uses :class:`bicameral_agent.gemini.GeminiClient`'s
``on_completion`` callback to capture every inner Gemini call without
monkey-patching the client.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bicameral_agent.assumption_auditor import AssumptionAuditor
from bicameral_agent.conscious_loop import ConsciousLoop
from bicameral_agent.context_refresher import ContextRefresher
from bicameral_agent.encoder import FEATURE_DIM
from bicameral_agent.gap_scanner import ResearchGapScanner
from bicameral_agent.gemini import ChatMessage, GeminiClient
from bicameral_agent.heuristic_controller import TOOL_IDS, Action
from bicameral_agent.queue import ContextQueue
from bicameral_agent.schema import Message
from bicameral_agent.token_estimator import ContextFeatures
from bicameral_agent.tool_latency import ToolLatencyModel
from bicameral_agent.tool_primitive import TokenBudget, ToolPrimitive

# Standard set of conversation length buckets (token counts).
DEFAULT_CONV_LENGTHS: tuple[int, ...] = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000)

# Generous budget — collection harness deliberately does not enforce caps.
_COLLECTION_BUDGET = TokenBudget(
    max_calls=10,
    max_input_tokens=200_000,
    max_output_tokens=20_000,
)

CONSCIOUS_LOOP_TOOL_ID = "conscious_loop"


@dataclass(frozen=True, slots=True)
class LatencyObservation:
    """One observed Gemini API call with the prediction made before it ran."""

    tool_id: str
    sub_call_label: str
    conversation_length_bucket: int
    run_index: int
    input_tokens: int
    output_tokens: int
    actual_duration_ms: float
    predicted_mean_ms: float
    predicted_p25_ms: float
    predicted_p75_ms: float
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """Outer-tool aggregate used to evaluate Layer 1 (token estimator) accuracy."""

    tool_id: str
    conversation_length_bucket: int
    run_index: int
    actual_conversation_tokens: int
    conversation_turn_count: int
    predicted_input_tokens: int
    predicted_output_tokens: int
    predicted_num_calls: int
    actual_input_tokens: int
    actual_output_tokens: int
    actual_num_calls: int
    actual_total_duration_ms: float
    timestamp_ms: int


# ---------------------------------------------------------------------------
# Synthetic conversation generation
# ---------------------------------------------------------------------------

# Realistic-ish vocabulary so the LLM has substantive content to reason about.
_SAMPLE_TOPICS = (
    "Researchers have investigated whether high-temperature superconductors "
    "could be synthesized at ambient pressure, and recent experimental work "
    "suggests several promising hydride compounds.",
    "Clinical trials of psilocybin therapy for treatment-resistant depression "
    "have shown rapid antidepressant effects after a single 25mg dose, with "
    "improvements sustained at three weeks.",
    "The ITER tokamak project aims to achieve net energy gain from fusion "
    "around 2035, while the National Ignition Facility already demonstrated "
    "ignition with 3.15 MJ output from 2.05 MJ of laser input.",
    "CAR-T cell therapies have driven complete-remission rates of 40-54% in "
    "diffuse large B-cell lymphoma, though cytokine release syndrome remains "
    "a frequent adverse effect.",
    "A 2022 study detected microplastics in 77% of human blood samples, with "
    "PET and polystyrene the most common polymers, raising concerns about "
    "systemic exposure pathways.",
    "Quantum error-correction milestones at Google demonstrated that scaling "
    "up surface-code qubit counts can actually reduce logical error rates, a "
    "key threshold for practical fault tolerance.",
    "Casgevy, the first CRISPR-based therapy, received FDA approval in "
    "December 2023 for sickle cell disease and transfusion-dependent "
    "beta-thalassemia.",
    "Compact tokamak designs from Commonwealth Fusion Systems use "
    "high-temperature superconducting magnets to enable smaller, cheaper "
    "fusion plants than traditional ITER-scale designs.",
)


def _approx_token_count(text: str) -> int:
    """Approximate token count using the project's len(text)//4 convention."""
    return len(text) // 4


def synthesize_conversation(target_tokens: int, seed: int = 0) -> list[Message]:
    """Build a deterministic synthetic conversation sized to ``target_tokens``.

    Alternates ``user``/``assistant`` messages, padding with content drawn
    from ``_SAMPLE_TOPICS`` until the cumulative ``token_count`` reaches the
    target. Tokens are estimated as ``len(text) // 4`` (matching project
    convention).

    Args:
        target_tokens: Approximate total tokens across all messages.
        seed: Random seed for deterministic content selection.

    Returns:
        Ordered list of :class:`Message` objects.
    """
    rng = random.Random(seed)
    topics = list(_SAMPLE_TOPICS)
    rng.shuffle(topics)

    messages: list[Message] = []
    accumulated_tokens = 0
    base_ts = 1_000_000
    idx = 0
    while accumulated_tokens < target_tokens:
        topic = topics[idx % len(topics)]
        chunks = [topic]
        chunks.extend(topics[(idx + k) % len(topics)] for k in range(1, 1 + (idx % 3)))
        content = " ".join(chunks)
        remaining = target_tokens - accumulated_tokens
        max_chars = remaining * 4
        if len(content) > max_chars and idx > 0:
            content = content[:max_chars]
        token_count = max(_approx_token_count(content), 1)
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append(
            Message(
                role=role,
                content=content,
                timestamp_ms=base_ts + idx * 1000,
                token_count=token_count,
            )
        )
        accumulated_tokens += token_count
        idx += 1
    return messages


# ---------------------------------------------------------------------------
# Collection harness
# ---------------------------------------------------------------------------


@dataclass
class _PendingPrediction:
    """One queued prediction waiting to be matched with an actual call."""

    sub_call_label: str
    predicted_mean_ms: float
    predicted_p25_ms: float
    predicted_p75_ms: float


class LatencyCollector:
    """Drives tool primitives and the conscious loop, recording observations.

    The collector wraps a :class:`GeminiClient` whose ``on_completion``
    callback feeds latency data into the configured :class:`ToolLatencyModel`
    and appends a :class:`LatencyObservation` to ``api_observations``.

    Predictions are made *before* each measurement and recorded alongside
    the actual values, so the resulting dataset contains both predicted and
    observed quantities for every API call.
    """

    def __init__(
        self,
        tool_latency_model: ToolLatencyModel,
        tools: dict[str, ToolPrimitive] | None = None,
    ) -> None:
        self._tool_latency_model = tool_latency_model
        self._tools: dict[str, ToolPrimitive] = tools or _default_tools()
        self._client: GeminiClient | None = None
        # Mutable shared state for the current measurement; the GeminiClient
        # already calls our on_completion synchronously after each call.
        self._current_tool_id: str | None = None
        self._current_conv_length: int = 0
        self._current_run_index: int = 0
        self._call_index_in_measurement: int = 0
        self._pending_predictions: list[_PendingPrediction] = []
        self._actual_input_tokens: int = 0
        self._actual_output_tokens: int = 0
        self._call_count: int = 0

        self.api_observations: list[LatencyObservation] = []
        self.tool_observations: list[ToolObservation] = []

    @property
    def on_completion(self) -> Callable[[int, int, float], None]:
        """Return the callback to register on a fresh GeminiClient.

        The collector owns the callback; users construct a client with
        ``GeminiClient(..., on_completion=collector.on_completion)`` and then
        call :meth:`bind_client` so subsequent collection calls have a client
        to invoke tools with.
        """
        return self._record_api_call

    def bind_client(self, client: GeminiClient) -> None:
        """Attach the GeminiClient that tool/loop calls will use."""
        self._client = client

    def _record_api_call(
        self, input_tokens: int, output_tokens: int, duration_ms: float
    ) -> None:
        if self._current_tool_id is None:
            return  # Not currently inside a measurement.

        # Pop the next pending prediction for this measurement (if any).
        pending = self._pop_pending_prediction()
        sub_call_label = pending.sub_call_label if pending else f"call_{self._call_index_in_measurement}"
        predicted_mean = pending.predicted_mean_ms if pending else 0.0
        predicted_p25 = pending.predicted_p25_ms if pending else 0.0
        predicted_p75 = pending.predicted_p75_ms if pending else 0.0

        self.api_observations.append(
            LatencyObservation(
                tool_id=self._current_tool_id,
                sub_call_label=sub_call_label,
                conversation_length_bucket=self._current_conv_length,
                run_index=self._current_run_index,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                actual_duration_ms=duration_ms,
                predicted_mean_ms=predicted_mean,
                predicted_p25_ms=predicted_p25,
                predicted_p75_ms=predicted_p75,
                timestamp_ms=int(time.time() * 1000),
            )
        )

        # Feed the observation back into the latency model immediately, so
        # subsequent predictions improve as we collect more data.
        self._tool_latency_model.observe(input_tokens, output_tokens, duration_ms)

        self._actual_input_tokens += input_tokens
        self._actual_output_tokens += output_tokens
        self._call_count += 1
        self._call_index_in_measurement += 1

    def _pop_pending_prediction(self) -> _PendingPrediction | None:
        if not self._pending_predictions:
            return None
        return self._pending_predictions.pop(0)

    def collect_tool(self, tool_id: str, conv_length: int, run_index: int) -> None:
        """Run one tool invocation and record per-call observations.

        Args:
            tool_id: One of the tool ids in ``TOOL_IDS``.
            conv_length: Target conversation length in tokens.
            run_index: Sequential run number for this (tool, length) cell.
        """
        if tool_id not in self._tools:
            raise ValueError(f"Unknown tool_id {tool_id!r}; have {sorted(self._tools)}")
        if self._client is None:
            raise RuntimeError("LatencyCollector has no client; call bind_client() first")

        history = synthesize_conversation(conv_length, seed=run_index)
        actual_conv_tokens = sum(m.token_count for m in history)
        ctx = ContextFeatures(
            conversation_length_tokens=actual_conv_tokens,
            conversation_turn_count=len(history),
        )

        prediction = self._tool_latency_model.predict(tool_id, ctx)

        self._begin_measurement(tool_id, conv_length, run_index)
        for sub in prediction.sub_calls:
            self._pending_predictions.append(
                _PendingPrediction(
                    sub_call_label=sub.label,
                    predicted_mean_ms=sub.latency.mean_ms,
                    predicted_p25_ms=sub.latency.p25_ms,
                    predicted_p75_ms=sub.latency.p75_ms,
                )
            )

        start_ns = time.monotonic_ns()
        try:
            self._tools[tool_id].execute(
                conversation_history=history,
                reasoning_state=np.zeros(FEATURE_DIM, dtype=np.float32),
                budget=_COLLECTION_BUDGET,
                client=self._client,
            )
        finally:
            duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            self._end_measurement(
                tool_id=tool_id,
                conv_length=conv_length,
                run_index=run_index,
                duration_ms=duration_ms,
                predicted_input_tokens=prediction.token_estimate.input_tokens,
                predicted_output_tokens=prediction.token_estimate.output_tokens,
                predicted_num_calls=prediction.token_estimate.num_calls,
                ctx=ctx,
                actual_conv_tokens=actual_conv_tokens,
                turn_count=len(history),
            )

    def collect_conscious_loop(
        self, conv_length: int, run_index: int = 0, user_message: str | None = None
    ) -> None:
        """Run one conscious-loop turn at the given conversation length."""
        if self._client is None:
            raise RuntimeError("LatencyCollector has no client; call bind_client() first")

        history = synthesize_conversation(conv_length, seed=run_index)

        queue = ContextQueue()
        loop = ConsciousLoop(
            client=self._client,
            queue=queue,
            system_prompt=(
                "You are a concise assistant. Reply briefly to the user."
            ),
            thinking_level="low",
        )
        # Seed prior history into the loop so the next turn includes it.
        for msg in history:
            loop._history.append(  # noqa: SLF001 — internal seeding for benchmarking
                ChatMessage(role=_to_gemini_role(msg.role), content=msg.content)
            )

        prompt = user_message or "Please summarize the discussion so far in one sentence."

        self._begin_measurement(CONSCIOUS_LOOP_TOOL_ID, conv_length, run_index)
        # Conscious-loop predictions are not part of the ToolLatencyModel grid;
        # we emit a single placeholder prediction for the user-message call so
        # the observation row is consistent with the others.
        actual_conv_tokens = sum(m.token_count for m in history)
        latency_prediction = self._tool_latency_model.latency_model.predict(
            input_tokens=actual_conv_tokens, output_tokens=200
        )
        self._pending_predictions.append(
            _PendingPrediction(
                sub_call_label="conscious_loop_turn",
                predicted_mean_ms=latency_prediction.mean_ms,
                predicted_p25_ms=latency_prediction.p25_ms,
                predicted_p75_ms=latency_prediction.p75_ms,
            )
        )

        start_ns = time.monotonic_ns()
        try:
            loop.run_turn(prompt)
        finally:
            duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            self._end_measurement(
                tool_id=CONSCIOUS_LOOP_TOOL_ID,
                conv_length=conv_length,
                run_index=run_index,
                duration_ms=duration_ms,
                predicted_input_tokens=actual_conv_tokens,
                predicted_output_tokens=200,
                predicted_num_calls=1,
                ctx=None,
                actual_conv_tokens=actual_conv_tokens,
                turn_count=len(history) + 1,
            )

    def _begin_measurement(
        self, tool_id: str, conv_length: int, run_index: int
    ) -> None:
        self._current_tool_id = tool_id
        self._current_conv_length = conv_length
        self._current_run_index = run_index
        self._call_index_in_measurement = 0
        self._pending_predictions = []
        self._actual_input_tokens = 0
        self._actual_output_tokens = 0
        self._call_count = 0

    def _end_measurement(
        self,
        *,
        tool_id: str,
        conv_length: int,
        run_index: int,
        duration_ms: float,
        predicted_input_tokens: int,
        predicted_output_tokens: int,
        predicted_num_calls: int,
        ctx: ContextFeatures | None,
        actual_conv_tokens: int,
        turn_count: int,
    ) -> None:
        actual_input = self._actual_input_tokens
        actual_output = self._actual_output_tokens
        actual_calls = self._call_count

        self.tool_observations.append(
            ToolObservation(
                tool_id=tool_id,
                conversation_length_bucket=conv_length,
                run_index=run_index,
                actual_conversation_tokens=actual_conv_tokens,
                conversation_turn_count=turn_count,
                predicted_input_tokens=predicted_input_tokens,
                predicted_output_tokens=predicted_output_tokens,
                predicted_num_calls=predicted_num_calls,
                actual_input_tokens=actual_input,
                actual_output_tokens=actual_output,
                actual_num_calls=actual_calls,
                actual_total_duration_ms=duration_ms,
                timestamp_ms=int(time.time() * 1000),
            )
        )

        # Conscious-loop and other ids outside the TokenEstimator registry are
        # excluded from Layer 1 updates (observe_tool would raise ValueError).
        if ctx is not None and tool_id != CONSCIOUS_LOOP_TOOL_ID:
            try:
                self._tool_latency_model.observe_tool(tool_id, ctx, actual_output)
            except ValueError:
                pass

        self._current_tool_id = None
        self._pending_predictions = []


def _to_gemini_role(schema_role: str) -> str:
    """Map schema ``role`` strings into Gemini roles."""
    return "model" if schema_role == "assistant" else schema_role


def _default_tools() -> dict[str, ToolPrimitive]:
    return {
        TOOL_IDS[Action.SCANNER]: ResearchGapScanner(),
        TOOL_IDS[Action.AUDITOR]: AssumptionAuditor(),
        TOOL_IDS[Action.REFRESHER]: ContextRefresher(),
    }


# ---------------------------------------------------------------------------
# Post-collection re-prediction (Layer 1/Layer 2 evaluation)
# ---------------------------------------------------------------------------


def recompute_predictions(
    api_obs: list[LatencyObservation],
    tool_obs: list[ToolObservation],
    trained_model: ToolLatencyModel,
) -> tuple[list[LatencyObservation], list[ToolObservation]]:
    """Return new observation lists with predictions recomputed by ``trained_model``.

    The acceptance criteria of issue #35 evaluate the model "after ingesting
    all data", so this is the canonical way to compare predictions to actuals
    after the online OLS / EMA updates have converged.
    """
    new_api: list[LatencyObservation] = []
    for obs in api_obs:
        pred = trained_model.latency_model.predict(obs.input_tokens, obs.output_tokens)
        new_api.append(
            replace(
                obs,
                predicted_mean_ms=pred.mean_ms,
                predicted_p25_ms=pred.p25_ms,
                predicted_p75_ms=pred.p75_ms,
            )
        )

    new_tool: list[ToolObservation] = []
    for obs in tool_obs:
        if obs.tool_id == CONSCIOUS_LOOP_TOOL_ID:
            new_tool.append(obs)
            continue
        ctx = ContextFeatures(
            conversation_length_tokens=obs.actual_conversation_tokens,
            conversation_turn_count=obs.conversation_turn_count,
        )
        try:
            est = trained_model.token_estimator.estimate(obs.tool_id, ctx)
        except ValueError:
            new_tool.append(obs)
            continue
        new_tool.append(
            replace(
                obs,
                predicted_input_tokens=est.input_tokens,
                predicted_output_tokens=est.output_tokens,
                predicted_num_calls=est.num_calls,
            )
        )
    return new_api, new_tool


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_API_OBS_FILENAME = "latency_observations.parquet"
_TOOL_OBS_FILENAME = "tool_observations.parquet"


def save_observations(
    output_dir: str | os.PathLike,
    api_obs: list[LatencyObservation],
    tool_obs: list[ToolObservation],
) -> tuple[str, str]:
    """Persist observations to two Parquet files under ``output_dir``.

    Returns the (api_path, tool_path) tuple of file paths written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    api_path = out / _API_OBS_FILENAME
    tool_path = out / _TOOL_OBS_FILENAME

    pq.write_table(_records_to_table(api_obs, LatencyObservation), str(api_path))
    pq.write_table(_records_to_table(tool_obs, ToolObservation), str(tool_path))

    return str(api_path), str(tool_path)


def load_observations(
    output_dir: str | os.PathLike,
) -> tuple[list[LatencyObservation], list[ToolObservation]]:
    """Read observations back from a directory written by ``save_observations``."""
    out = Path(output_dir)
    api_path = out / _API_OBS_FILENAME
    tool_path = out / _TOOL_OBS_FILENAME
    api_obs = _table_to_records(pq.read_table(str(api_path)), LatencyObservation)
    tool_obs = _table_to_records(pq.read_table(str(tool_path)), ToolObservation)
    return api_obs, tool_obs


def _records_to_table(records, cls) -> pa.Table:
    fields = list(cls.__dataclass_fields__)
    return pa.table({f: [getattr(rec, f) for rec in records] for f in fields})


def _table_to_records(table: pa.Table, cls):
    rows = table.to_pylist()
    return [cls(**row) for row in rows]
