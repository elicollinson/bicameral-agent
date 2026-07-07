"""Model-client factory: pick a backend by provider name (issue #43).

Single source of truth for translating a ``provider`` string into a concrete
client (``GeminiClient`` or ``OllamaCloudClient``). Both satisfy the same
duck-typed contract, so callers downstream are provider-agnostic.
"""

from __future__ import annotations

from typing import Callable

from bicameral_agent.gemini import GeminiClient
from bicameral_agent.ollama_cloud import OllamaCloudClient

_PROVIDERS = ("gemini", "ollama")


def build_client(
    provider: str = "gemini",
    model: str | None = None,
    *,
    api_key: str | None = None,
    on_completion: Callable[[int, int, float], None] | None = None,
):
    """Construct a model client for *provider*.

    Args:
        provider: 'gemini' or 'ollama'.
        model: Model id/tag; None uses the client's own default.
        api_key: Optional explicit API key (else read from the provider's env var).
        on_completion: Optional ``(input_tokens, output_tokens, duration_ms)`` callback.

    Returns:
        A ``GeminiClient`` or ``OllamaCloudClient``.

    Raises:
        ValueError: If *provider* is not recognised.
    """
    kwargs: dict = {"api_key": api_key, "on_completion": on_completion}
    if model is not None:
        kwargs["model"] = model

    if provider == "gemini":
        return GeminiClient(**kwargs)
    if provider == "ollama":
        return OllamaCloudClient(**kwargs)
    raise ValueError(f"Unknown provider {provider!r}; known providers: {list(_PROVIDERS)}")
