"""Main execution loop driving multi-turn Gemini conversations with context injection.

Orchestrates the conscious loop: runs generation turns and injects context
from the ContextQueue at breakpoints. Interrupt handling lives in the
EpisodeRunner (InjectionMode.INTERRUPT): generation is synchronous, so
nothing can enqueue while ``generate()`` runs — an in-loop post-generation
interrupt check could never fire in production and was removed (issue #54).

Injection persistence semantics (issue #49): by default injected context is
*persistent* — the context-augmented user message is what enters conversation
history, so injected findings remain visible to every subsequent generation.
Because the queue drain is destructive and the augmented message is stored
exactly once, the injected text is token-accounted once at injection and then
carried as ordinary history. Passing ``persistent_injection=False`` restores
the *transient* "whisper" behavior: context is prepended only for the
immediate API call and vanishes from history afterwards, so it influences
exactly one generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from bicameral_agent.gemini import ChatMessage, GeminiClient, GeminiResponse
from bicameral_agent.queue import ContextQueue


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """Result of a single conscious loop turn."""

    content: str
    turn_number: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    context_injected: bool


class ConsciousLoop:
    """Drives multi-turn Gemini conversations with context injection.

    Each call to run_turn() sends a user message, checks for queued context
    at breakpoints, and generates a response.

    ``persistent_injection`` controls visibility of injected context across
    turns. When True (default), the context-augmented user message is stored
    in history, so injected context stays visible to all later generations
    and is token-accounted exactly once. When False, injected context is
    used only for the immediate API call (transient "whisper" mode) and
    vanishes from history after one generation.
    """

    def __init__(
        self,
        client: GeminiClient,
        queue: ContextQueue,
        *,
        system_prompt: str | None = None,
        thinking_level: str = "medium",
        temperature: float | None = None,
        on_completion: Callable[[AssistantResponse], None] | None = None,
        persistent_injection: bool = True,
    ) -> None:
        self._client = client
        self._queue = queue
        self._system_prompt = system_prompt
        self._thinking_level = thinking_level
        self._temperature = temperature
        self._on_completion = on_completion
        self._persistent_injection = persistent_injection
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
        4. In persistent mode, replace the stored user message with
           the context-augmented one; append assistant response to history
        5. Fire on_completion callback, return AssistantResponse
        """
        self._turn_count += 1
        self._history.append(ChatMessage(role="user", content=user_message))

        start_ns = time.monotonic_ns()

        # Breakpoint drain
        context_str = self._queue.drain_at_breakpoint()
        context_injected = context_str is not None

        # Build messages with context prepended to user message
        response = self._generate(user_message, context_str)

        duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        if context_str is not None and self._persistent_injection:
            # Persist the augmented message so injected context stays visible
            # on later turns. It enters history exactly once (the drain was
            # destructive), so its tokens are accounted once at injection.
            self._history[-1] = ChatMessage(
                role="user", content=_augment(user_message, context_str)
            )

        self._history.append(ChatMessage(role="model", content=response.content))

        result = AssistantResponse(
            content=response.content,
            turn_number=self._turn_count,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
            context_injected=context_injected,
        )

        if self._on_completion is not None:
            self._on_completion(result)

        return result

    def regenerate_with_context(self, context_str: str) -> AssistantResponse:
        """Re-generate the last assistant response with additional context.

        Pops the last model message from history, then regenerates the
        preceding user message with the provided context. Does NOT increment
        turn count. In persistent mode, the stored user message is replaced
        with the context-augmented one so the context stays visible on later
        turns.
        """
        if not self._history or self._history[-1].role != "model":
            raise ValueError("No model message to replace in history")

        # Pop the last model message
        self._history.pop()

        # History must now end with the user message that prompted the
        # replaced response — _generate builds its prompt as history[:-1]
        # plus the (augmented) last message, so any other shape would
        # silently duplicate messages.
        if not self._history or self._history[-1].role != "user":
            raise ValueError("No user message found in history")
        last_user_msg = self._history[-1].content

        start_ns = time.monotonic_ns()
        response = self._generate(last_user_msg, context_str)
        duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        if self._persistent_injection:
            self._history[-1] = ChatMessage(
                role="user", content=_augment(last_user_msg, context_str)
            )

        self._history.append(ChatMessage(role="model", content=response.content))

        return AssistantResponse(
            content=response.content,
            turn_number=self._turn_count,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
            context_injected=True,
        )

    def _generate(
        self, user_message: str, context_str: str | None
    ) -> GeminiResponse:
        """Build messages and call the Gemini API."""
        prior = self._history[:-1]

        augmented_content = _augment(user_message, context_str)

        messages = prior + [ChatMessage(role="user", content=augmented_content)]
        return self._client.generate(
            messages,
            system_prompt=self._system_prompt,
            thinking_level=self._thinking_level,
            temperature=self._temperature,
        )


def _augment(user_message: str, context_str: str | None) -> str:
    """Prepend injected context to a user message."""
    if context_str is None:
        return user_message
    return context_str + "\n\n" + user_message
