"""Tests for provider/backend selection wiring (Issue #43).

Covers the ``build_client`` factory, the ``ModelConfig.provider`` field, the
``HyperConfig.to_model_client`` adapter, and that the Ollama Gemma tag is a
known (flat-rate) model for ``CostTracker``.
"""

from __future__ import annotations

import pytest

from bicameral_agent.config import HyperConfig, ModelConfig
from bicameral_agent.cost_tracker import MODEL_PRICING, CostTracker
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.model_client import build_client
from bicameral_agent.ollama_cloud import OllamaCloudClient


class TestBuildClient:
    def test_gemini_provider(self):
        client = build_client("gemini", api_key="k")
        assert isinstance(client, GeminiClient)

    def test_ollama_provider(self):
        client = build_client("ollama", api_key="k")
        assert isinstance(client, OllamaCloudClient)
        assert client.model == "gemma4:31b-cloud"

    def test_model_override(self):
        client = build_client("ollama", "gemma3:27b-cloud", api_key="k")
        assert client.model == "gemma3:27b-cloud"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            build_client("anthropic", api_key="k")


class TestModelConfigProvider:
    def test_default_provider_is_gemini(self):
        assert ModelConfig().provider == "gemini"

    def test_ollama_provider_allowed(self):
        assert ModelConfig(provider="ollama").provider == "ollama"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValueError, match="provider must be one of"):
            ModelConfig(provider="openai")

    def test_to_model_client_uses_provider(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "k")
        config = HyperConfig(model=ModelConfig(provider="ollama", name="gemma4:31b-cloud"))
        client = config.to_model_client()
        assert isinstance(client, OllamaCloudClient)
        assert client.model == "gemma4:31b-cloud"


class TestGemmaPricing:
    def test_gemma_is_known_flat_rate_model(self):
        assert "gemma4:31b-cloud" in MODEL_PRICING
        pricing = MODEL_PRICING["gemma4:31b-cloud"]
        assert pricing.input_cost_per_token == 0.0
        assert pricing.output_cost_per_token == 0.0

    def test_record_call_counts_without_cost(self):
        tracker = CostTracker()
        tracker.record_call(1000, 500, "gemma4:31b-cloud")
        report = tracker.get_total()
        assert report.call_count == 1
        assert report.total == 0.0
