"""Tests for provider/backend selection wiring (Issues #43, #52).

Covers the provider registry, the ``build_client`` factory (including
fail-fast pricing and provider/model cross-validation), the
``ModelConfig.provider`` field, the ``HyperConfig.to_model_client``
adapter, and that the Ollama Gemma tag is a known (flat-rate) model for
``CostTracker``.
"""

from __future__ import annotations

import logging

import pytest

from bicameral_agent.config import HyperConfig, ModelConfig
from bicameral_agent.cost_tracker import MODEL_PRICING, CostTracker, resolve_pricing
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.model_client import build_client, default_model, provider_names
from bicameral_agent.ollama_cloud import OllamaCloudClient


class TestProviderRegistry:
    def test_provider_names(self):
        assert set(provider_names()) == {"gemini", "ollama"}

    def test_default_model_per_provider(self):
        assert default_model("gemini") == "gemini-3.1-flash-lite-preview"
        assert default_model("ollama") == "gemma4:31b-cloud"

    def test_default_model_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            default_model("anthropic")


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


class TestFailFastPricing:
    def test_unpriced_gemini_tag_rejected_at_build_time(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_client("gemini", "gemini-9-ultra", api_key="k")

    def test_unknown_ollama_tag_builds_with_flat_rate_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bicameral_agent.cost_tracker"):
            client = build_client("ollama", "qwen9:9b-cloud", api_key="k")
        assert isinstance(client, OllamaCloudClient)
        assert "subscription-flat" in caplog.text
        pricing = resolve_pricing("qwen9:9b-cloud", provider="ollama")
        assert pricing.input_cost_per_token == 0.0
        assert pricing.output_cost_per_token == 0.0

    def test_unknown_tag_without_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_pricing("not-a-real-model")


class TestProviderModelCrossValidation:
    def test_ollama_provider_with_gemini_tag_rejected(self):
        with pytest.raises(ValueError, match="looks like a 'gemini' tag"):
            build_client("ollama", "gemini-3.1-flash-lite-preview", api_key="k")

    def test_gemini_provider_with_ollama_tag_rejected(self):
        with pytest.raises(ValueError, match="looks like a 'ollama' tag"):
            build_client("gemini", "gemma4:31b-cloud", api_key="k")

    def test_model_config_rejects_mismatched_pair(self):
        with pytest.raises(ValueError, match="looks like a 'gemini' tag"):
            ModelConfig(provider="ollama", name="gemini-3.1-flash-lite-preview")


class TestModelConfigProvider:
    def test_default_provider_is_gemini(self):
        assert ModelConfig().provider == "gemini"

    def test_ollama_provider_allowed(self):
        assert ModelConfig(provider="ollama").provider == "ollama"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValueError, match="provider must be one of"):
            ModelConfig(provider="openai")

    def test_name_defaults_to_provider_default(self):
        assert ModelConfig().name == "gemini-3.1-flash-lite-preview"
        assert ModelConfig(provider="ollama").name == "gemma4:31b-cloud"

    def test_env_provider_override_switches_default_name(self, monkeypatch):
        monkeypatch.setenv("BICAMERAL_MODEL__PROVIDER", "ollama")
        cfg = HyperConfig().with_env_overrides()
        assert cfg.model.provider == "ollama"
        assert cfg.model.name == "gemma4:31b-cloud"

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
