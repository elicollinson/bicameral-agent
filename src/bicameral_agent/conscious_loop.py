"""Main execution loop driving multi-turn Gemini conversations with context injection.

Orchestrates the conscious loop: runs generation turns, injects context from the
ContextQueue at breakpoints, and handles interrupts when critical context arrives
mid-generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from bicameral_agent.gemini import ChatMessage, GeminiClient, GeminiResponse
from bicameral_agent.queue import ContextQueue, InterruptConfig


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """Result of a single conscious loop turn."""

    content: str
    turn_number: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_ms: float
    interrupted: bool
    context_injected: bool


class ConsciousLoop:
    """Drives multi-turn Gemini conversations with context injection and interrupts.

    Each call to run_turn() sends a user message, checks for queued context
    at breakpoints, generates a response, and optionally interrupts and
    retries if critical context arrives during generation.
    """

    def __init__(
        self,
        client: GeminiClient,
        queue: ContextQueue,
        *,
        system_prompt: str | None = None,
        thinking_level: str = "medium",
        interrupt_config: InterruptConfig | None = None,
        on_completion: Callable[[AssistantResponse], None] | None = None,
    ) -> None:
        self._client = client
        self._queue = queue
        self._system_prompt = system_prompt
        self._thinking_level = thinking_level
        self._interrupt_config = interrupt_config or InterruptConfig()
        self._on_completion = on_completion
        self._history: list[ChatMessage] = []
        self._turn_count = 0

    @property
    def history(self) -> list[ChatMessage]:
        """Return a copy of the conversation history."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        """Return the number of completed turns."""
        return self._turn_count

    def run_turn(self, user_message: str) -> AssistantResponse:
        """Execute a single conversation turn.

        1. Increment turn number, append user message to history
        2. Drain context at breakpoint
        3. Build messages and generate
        4. Check interrupt threshold — if triggered, report wasted tokens,
           freeze, drain again, regenerate
        5. Unfreeze, append assistant response to history
        6. Fire on_completion callback, return AssistantResponse
        """
        self._turn_count += 1
        self._history.append(ChatMessage(role="user", content=user_message))

        start_ns = time.monotonic_ns()

        # Breakpoint drain
        context_str = self._queue.drain_at_breakpoint()
        context_injected = context_str is not None

        # Build messages with context prepended to user message
        response = self._generate(user_message, context_str)

        interrupted = False
        wasted_input = 0
        wasted_output = 0

        # Check interrupt threshold
        if self._queue.check_interrupt_threshold(self._interrupt_config):
            interrupted = True
            wasted_input = response.input_tokens
            wasted_output = response.output_tokens
            self._queue.report_wasted_tokens(wasted_input + wasted_output)
            self._queue.freeze()

            # Drain again and combine contexts
            new_context = self._queue.drain_at_breakpoint()
            if new_context is not None:
                if context_str is not None:
                    context_str = context_str + "\n" + new_context
                else:
                    context_str = new_context
                context_injected = True

            # Regenerate
            response = self._generate(user_message, context_str)

        if interrupted:
            self._queue.unfreeze()

        duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        self._history.append(ChatMessage(role="model", content=response.content))

        total_tokens = (
            response.input_tokens
            + response.output_tokens
            + wasted_input
            + wasted_output
        )

        result = AssistantResponse(
            content=response.content,
            turn_number=self._turn_count,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            interrupted=interrupted,
            context_injected=context_injected,
        )

        if self._on_completion is not None:
            self._on_completion(result)

        return result

    def regenerate_with_context(self, context_str: str) -> AssistantResponse:
        """Re-generate the last assistant response with additional context.

        Pops the last model message from history, finds the last user message,
        and regenerates with the provided context. Does NOT increment turn count.
        """
        if not self._history or self._history[-1].role != "model":
            raise ValueError("No model message to replace in history")

        # Pop the last model message
        self._history.pop()

        # Find the last user message
        last_user_msg = None
        for msg in reversed(self._history):
            if msg.role == "user":
                last_user_msg = msg.content
                break
        if last_user_msg is None:
            raise ValueError("No user message found in history")

        start_ns = time.monotonic_ns()
        response = self._generate(last_user_msg, context_str)
        duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        self._history.append(ChatMessage(role="model", content=response.content))

        return AssistantResponse(
            content=response.content,
            turn_number=self._turn_count,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.input_tokens + response.output_tokens,
            duration_ms=duration_ms,
            interrupted=False,
            context_injected=True,
        )

    def _generate(
        self, user_message: str, context_str: str | None
    ) -> GeminiResponse:
        """Build messages and call the Gemini API."""
        prior = self._history[:-1]

        if context_str is not None:
            augmented_content = context_str + "\n\n" + user_message
        else:
            augmented_content = user_message

        messages = prior + [ChatMessage(role="user", content=augmented_content)]
        return self._client.generate(
            messages,
            system_prompt=self._system_prompt,
            thinking_level=self._thinking_level,
        )
