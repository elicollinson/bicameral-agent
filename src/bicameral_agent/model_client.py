"""Provider-neutral model-client layer (issues #43, #52).

Single source of truth for:

- the provider registry (``PROVIDERS``): provider names, per-provider default
  model tags, and plausible-tag patterns, consumed by config validation and
  CLI ``choices``;
- the neutral ``ChatMessage`` / ``ModelResponse`` dataclasses that every
  client accepts and returns (``gemini.GeminiResponse`` is a back-compat
  alias);
- the shared retry/backoff scaffold (``RetryingClientBase``) with a
  per-provider ``_is_retryable`` hook;
- the ``build_client`` factory that turns a ``provider`` string into a
  concrete client, failing fast on unknown providers, implausible
  provider/model pairings, and unpriced model tags -- before any paid call.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0
_MAX_JITTER = 0.5

VALID_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A message in the conversation sent to a model API.

    Lighter than schema.Message -- no timestamp/token_count fields,
    since those are logging concerns, not API input concerns.
    """

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Provider-neutral response from a model API call with metadata."""

    content: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    finish_reason: str
    function_calls: list[dict[str, Any]] | None = field(default=None)


class ModelClient(Protocol):
    """Structural contract satisfied by every model client and wrapper."""

    @property
    def model(self) -> str:
        """The model name used for API calls."""
        ...

    def generate(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        thinking_level: str = "medium",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> ModelResponse:
        """Generate a response from the model API."""
        ...


def validate_thinking_level(thinking_level: str) -> str:
    """Return *thinking_level* lowercased, or raise for unknown levels."""
    lowered = thinking_level.lower()
    if lowered not in VALID_THINKING_LEVELS:
        raise ValueError(
            f"Invalid thinking_level {thinking_level!r}; "
            f"must be one of {sorted(VALID_THINKING_LEVELS)}"
        )
    return lowered


class TransportExhausted(RuntimeError):
    """A retryable transport error survived the whole client retry budget.

    Raised by ``RetryingClientBase._execute_with_retry`` in place of the
    final provider exception (preserved as ``__cause__``) so callers can
    contain exhausted-retry transport failures narrowly, without also
    catching non-transient errors such as bad requests or parsing bugs
    (issue #81).
    """

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"transport error persisted after {attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )


class RetryingClientBase:
    """Shared retry/backoff scaffold for concrete model clients.

    Subclasses implement ``_is_retryable`` (their typed, provider-specific
    transient-error check) and route each API call through
    ``_execute_with_retry`` with a zero-arg attempt callable that performs
    one timed request/parse round trip. A retryable error that survives
    the whole budget is raised as ``TransportExhausted``; non-retryable
    errors propagate unchanged.
    """

    def _execute_with_retry(self, attempt: Callable[[], ModelResponse]) -> ModelResponse:
        for attempt_idx in range(_MAX_RETRIES + 1):
            if attempt_idx > 0:
                delay = _BASE_DELAY * (_BACKOFF_FACTOR ** (attempt_idx - 1))
                jitter = random.uniform(0, _MAX_JITTER)
                time.sleep(delay + jitter)

            try:
                return attempt()
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise
                if attempt_idx == _MAX_RETRIES:
                    raise TransportExhausted(_MAX_RETRIES + 1, exc) from exc

        raise AssertionError("unreachable: retry loop always returns or raises")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        raise NotImplementedError


def _load_gemini_client() -> type:
    from bicameral_agent.gemini import GeminiClient

    return GeminiClient


def _load_ollama_client() -> type:
    from bicameral_agent.ollama_cloud import OllamaCloudClient

    return OllamaCloudClient


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Registry entry for a model provider.

    Attributes:
        default_model: Model tag used when none is configured.
        model_pattern: Shape of this provider's model tags, used to reject
            implausible provider/model pairings (a tag that matches another
            provider's pattern but not this one's).
        load_client: Lazy import of the concrete client class.
        flat_rate: Subscription-flat pricing: unknown tags cost $0/token
            (with a warning) instead of failing pricing validation.
    """

    default_model: str
    model_pattern: re.Pattern[str]
    load_client: Callable[[], type]
    flat_rate: bool = False


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        default_model="gemini-3.1-flash-lite-preview",
        model_pattern=re.compile(r"gemini-\S+"),
        load_client=_load_gemini_client,
    ),
    "ollama": ProviderSpec(
        default_model="gemma4:31b-cloud",
        model_pattern=re.compile(r"[^:\s]+:[^:\s]+"),
        load_client=_load_ollama_client,
        flat_rate=True,
    ),
}


def provider_names() -> tuple[str, ...]:
    """Known provider names, for CLI choices and config validation."""
    return tuple(PROVIDERS)


def _spec(provider: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(
            f"Unknown provider {provider!r}; known providers: {sorted(PROVIDERS)}"
        )
    return spec


def default_model(provider: str) -> str:
    """The default model tag for *provider*."""
    return _spec(provider).default_model


def validate_provider_model(provider: str, model: str) -> None:
    """Reject a *model* tag that plausibly belongs to a different *provider*.

    Catches the config footgun of switching ``provider`` while leaving the
    model name at another provider's tag (e.g. provider='ollama' with the
    Gemini default name). Unrecognisable tags are allowed through; pricing
    validation at client-build time is the backstop.

    Raises:
        ValueError: If *provider* is unknown, or *model* matches another
            provider's tag pattern but not this provider's.
    """
    spec = _spec(provider)
    if spec.model_pattern.fullmatch(model):
        return
    for other, other_spec in PROVIDERS.items():
        if other != provider and other_spec.model_pattern.fullmatch(model):
            raise ValueError(
                f"Model {model!r} looks like a {other!r} tag but provider is "
                f"{provider!r}; set a {provider} model or switch provider."
            )


def build_client(
    provider: str = "gemini",
    model: str | None = None,
    *,
    api_key: str | None = None,
    on_completion: Callable[[int, int, float], None] | None = None,
):
    """Construct a model client for *provider*, failing fast on bad configs.

    Args:
        provider: A key of ``PROVIDERS`` ('gemini' or 'ollama').
        model: Model id/tag; None uses the provider's default from the registry.
        api_key: Optional explicit API key (else read from the provider's env var).
        on_completion: Optional ``(input_tokens, output_tokens, duration_ms)`` callback.

    Returns:
        A ``GeminiClient`` or ``OllamaCloudClient``.

    Raises:
        ValueError: If *provider* is unknown, *model* is implausible for
            *provider*, or *model* has no resolvable pricing (fail fast,
            before any paid call could be destroyed post-hoc).
    """
    from bicameral_agent.cost_tracker import resolve_pricing

    spec = _spec(provider)
    resolved_model = model if model is not None else spec.default_model
    validate_provider_model(provider, resolved_model)
    resolve_pricing(resolved_model, provider=provider)

    client_cls = spec.load_client()
    return client_cls(
        api_key=api_key, on_completion=on_completion, model=resolved_model
    )
