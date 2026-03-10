"""Tests for the Context Refresher tool primitive."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from bicameral_agent.gemini import GeminiResponse
from bicameral_agent.queue import Priority
from bicameral_agent.schema import Message
from bicameral_agent.tool_primitive import BudgetExceededError, TokenBudget
from bicameral_agent.context_refresher import (
    ContextRefresher,
    DriftCategory,
    _drift_priority,
    _extract_first_user_message,
    _make_dedup_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BUDGET = TokenBudget(max_calls=1, max_input_tokens=2000, max_output_tokens=500)


def _msg(role: str, content: str, ts: int = 1000) -> Message:
    return Message(role=role, content=content, timestamp_ms=ts, token_count=len(content.split()))


def _make_history(
    task: str = "Build a REST API with pagination, rate limiting, and auth",
    recent: list[tuple[str, str]] | None = None,
    middle_padding: int = 0,
) -> list[Message]:
    """Build a conversation history with a user task and recent messages."""
    msgs = [_msg("user", task, ts=1000)]
    for i in range(middle_padding):
        msgs.append(_msg("assistant", f"Middle message {i}", ts=2000 + i))
    if recent is None:
        recent = [
            ("assistant", "I'll start by setting up the database schema."),
            ("user", "Looks good, continue."),
            ("assistant", "Now implementing the endpoints with pagination support."),
        ]
    for role, content in recent:
        msgs.append(_msg(role, content, ts=5000 + len(msgs)))
    return msgs


def _make_state():
    return np.zeros(53, dtype=np.float32)


def _fake_response(content: str, input_tokens: int = 50, output_tokens: int = 80) -> GeminiResponse:
    return GeminiResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=40.0,
        finish_reason="STOP",
    )


def _mock_client(response: GeminiResponse | None = None) -> MagicMock:
    client = MagicMock(spec=["generate"])
    client.generate.return_value = response or _fake_response(
        json.dumps({"drift_detected": False, "drifts": [], "reminder": None})
    )
    return client


def _drift_response(
    drifts: list[dict] | None = None,
    reminder: str = "Refocus: you must implement rate limiting and auth, not just pagination.",
) -> GeminiResponse:
    if drifts is None:
        drifts = [
            {"category": "requirement_ignored", "description": "Rate limiting not addressed"},
            {"category": "scope_creep", "description": "Added unnecessary caching layer"},
        ]
    payload = {"drift_detected": True, "drifts": drifts, "reminder": reminder}
    return _fake_response(json.dumps(payload))


def _no_drift_response() -> GeminiResponse:
    payload = {"drift_detected": False, "drifts": [], "reminder": None}
    return _fake_response(json.dumps(payload))


# ---------------------------------------------------------------------------
# TestDriftDetection
# ---------------------------------------------------------------------------


class TestDriftDetection:
    def test_constraint_violation_high_priority(self):
        drifts = [{"category": "constraint_violation", "description": "Exceeded token limit"}]
        client = _mock_client(_drift_response(drifts=drifts))
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.queue_deposit.priority == Priority.HIGH

    def test_scope_creep_medium_priority(self):
        drifts = [{"category": "scope_creep", "description": "Added caching"}]
        client = _mock_client(_drift_response(drifts=drifts))
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.queue_deposit.priority == Priority.MEDIUM

    def test_requirement_ignored_high_priority(self):
        drifts = [{"category": "requirement_ignored", "description": "Auth not implemented"}]
        client = _mock_client(_drift_response(drifts=drifts))
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.queue_deposit.priority == Priority.HIGH

    def test_mixed_drifts_take_max_priority(self):
        drifts = [
            {"category": "scope_creep", "description": "Added caching"},
            {"category": "constraint_violation", "description": "Exceeded limit"},
        ]
        client = _mock_client(_drift_response(drifts=drifts))
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.queue_deposit.priority == Priority.HIGH

    def test_deposit_has_expiry_turns_3(self):
        client = _mock_client(_drift_response())
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        assert result.queue_deposit.expiry_turns == 3


# ---------------------------------------------------------------------------
# TestNoDrift
# ---------------------------------------------------------------------------


class TestNoDrift:
    def test_on_track_returns_none(self):
        client = _mock_client(_no_drift_response())
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is None

    def test_short_conversation_returns_none_without_llm(self):
        """A conversation with <2 messages should skip the LLM call entirely."""
        client = _mock_client()
        refresher = ContextRefresher()
        history = [_msg("user", "Hello")]
        result = refresher.execute(history, _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is None
        assert client.generate.call_count == 0

    def test_no_user_message_returns_none(self):
        client = _mock_client()
        refresher = ContextRefresher()
        history = [
            _msg("system", "You are helpful"),
            _msg("assistant", "Hello!"),
        ]
        result = refresher.execute(history, _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is None
        assert client.generate.call_count == 0


# ---------------------------------------------------------------------------
# TestConciseness
# ---------------------------------------------------------------------------


class TestConciseness:
    def test_reminder_within_100_words(self):
        client = _mock_client(_drift_response())
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        word_count = len(result.queue_deposit.content.split())
        assert word_count <= 100

    def test_long_reminder_truncated(self):
        long_reminder = " ".join(["word"] * 150)
        client = _mock_client(_drift_response(reminder=long_reminder))
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is not None
        word_count = len(result.queue_deposit.content.split())
        assert word_count <= 100


# ---------------------------------------------------------------------------
# TestBudgetCompliance
# ---------------------------------------------------------------------------


class TestBudgetCompliance:
    def test_exactly_one_llm_call(self):
        client = _mock_client(_no_drift_response())
        refresher = ContextRefresher()
        refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert client.generate.call_count == 1

    def test_budget_exceeded_returns_none(self):
        client = MagicMock(spec=["generate"])
        client.generate.side_effect = BudgetExceededError("budget exceeded")
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.queue_deposit is None


# ---------------------------------------------------------------------------
# TestMessageExtraction
# ---------------------------------------------------------------------------


class TestMessageExtraction:
    def test_uses_first_user_message(self):
        client = _mock_client(_no_drift_response())
        refresher = ContextRefresher()
        history = _make_history(task="Specific task XYZ")
        refresher.execute(history, _make_state(), _DEFAULT_BUDGET, client)

        call_args = client.generate.call_args
        prompt = call_args[0][0][0]["content"]
        assert "Specific task XYZ" in prompt

    def test_uses_last_3_messages(self):
        client = _mock_client(_no_drift_response())
        refresher = ContextRefresher()
        history = _make_history(middle_padding=5)
        refresher.execute(history, _make_state(), _DEFAULT_BUDGET, client)

        call_args = client.generate.call_args
        prompt = call_args[0][0][0]["content"]
        # Last 3 messages should be in the recent section
        assert "pagination support" in prompt
        # Middle messages should NOT be in the prompt
        assert "Middle message 0" not in prompt

    def test_ignores_middle_messages(self):
        client = _mock_client(_no_drift_response())
        refresher = ContextRefresher()
        history = _make_history(middle_padding=10)
        refresher.execute(history, _make_state(), _DEFAULT_BUDGET, client)

        call_args = client.generate.call_args
        prompt = call_args[0][0][0]["content"]
        for i in range(7):  # Middle messages 0-6 are not in last 3
            assert f"Middle message {i}" not in prompt


# ---------------------------------------------------------------------------
# TestDedupKey
# ---------------------------------------------------------------------------


class TestDedupKey:
    def test_order_independent(self):
        a = [(DriftCategory.SCOPE_CREEP, "desc1"), (DriftCategory.CONSTRAINT_VIOLATION, "desc2")]
        b = [(DriftCategory.CONSTRAINT_VIOLATION, "desc2"), (DriftCategory.SCOPE_CREEP, "desc1")]
        assert _make_dedup_key(a) == _make_dedup_key(b)

    def test_different_drifts_different_key(self):
        a = [(DriftCategory.SCOPE_CREEP, "desc1")]
        b = [(DriftCategory.SCOPE_CREEP, "desc2")]
        assert _make_dedup_key(a) != _make_dedup_key(b)

    def test_prefix(self):
        a = [(DriftCategory.SCOPE_CREEP, "desc1")]
        assert _make_dedup_key(a).startswith("context_refresher:")


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_all_fields_populated(self):
        client = _mock_client(_drift_response())
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        meta = result.metadata
        assert meta.tool_id == "context_refresher"
        assert meta.action_taken != ""
        assert 0.0 <= meta.confidence <= 1.0
        assert meta.items_found >= 0
        assert 0.0 <= meta.estimated_relevance <= 1.0
        assert meta.tokens_consumed >= 0
        assert meta.execution_duration_ms >= 0

    def test_no_drift_metadata(self):
        client = _mock_client(_no_drift_response())
        refresher = ContextRefresher()
        result = refresher.execute(_make_history(), _make_state(), _DEFAULT_BUDGET, client)

        assert result.metadata.items_found == 0
        assert result.metadata.confidence == 0.8


# ---------------------------------------------------------------------------
# TestPriorityMapping
# ---------------------------------------------------------------------------


class TestPriorityMapping:
    def test_constraint_violation_is_high(self):
        assert _drift_priority(DriftCategory.CONSTRAINT_VIOLATION) == Priority.HIGH

    def test_scope_creep_is_medium(self):
        assert _drift_priority(DriftCategory.SCOPE_CREEP) == Priority.MEDIUM

    def test_requirement_ignored_is_high(self):
        assert _drift_priority(DriftCategory.REQUIREMENT_IGNORED) == Priority.HIGH


# ---------------------------------------------------------------------------
# TestFirstUserMessage
# ---------------------------------------------------------------------------


class TestFirstUserMessage:
    def test_finds_first_user(self):
        history = [
            _msg("system", "sys"),
            _msg("user", "first"),
            _msg("assistant", "reply"),
            _msg("user", "second"),
        ]
        result = _extract_first_user_message(history)
        assert result is not None
        assert result.content == "first"

    def test_no_user_returns_none(self):
        history = [_msg("system", "sys"), _msg("assistant", "reply")]
        assert _extract_first_user_message(history) is None

    def test_empty_returns_none(self):
        assert _extract_first_user_message([]) is None


# ---------------------------------------------------------------------------
# Integration tests (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestIntegration:
    def _make_drifted_conversation(self, task: str, drift_messages: list[tuple[str, str]]) -> list[Message]:
        msgs = [_msg("user", task, ts=1000)]
        for i, (role, content) in enumerate(drift_messages):
            msgs.append(_msg(role, content, ts=2000 + i * 1000))
        return msgs

    def test_catches_drift_in_planted_conversations(self):
        from bicameral_agent.gemini import GeminiClient

        client = GeminiClient()
        budget = TokenBudget(max_calls=1, max_input_tokens=2000, max_output_tokens=500)

        drifted_convos = [
            ("Build a REST API with auth and rate limiting", [
                ("assistant", "Setting up database models for user profiles."),
                ("assistant", "Now I'll add a caching layer with Redis."),
                ("assistant", "Implementing the Redis pub/sub messaging system."),
            ]),
            ("Write unit tests for the payment module", [
                ("assistant", "Looking at the payment module code."),
                ("assistant", "I think we should refactor the payment module first."),
                ("assistant", "Let me redesign the entire payment architecture."),
            ]),
            ("Fix the login bug where users get 403 errors", [
                ("assistant", "Investigating the auth middleware."),
                ("assistant", "The auth system could use some improvements."),
                ("assistant", "Let me add OAuth2 support and social login."),
            ]),
            ("Summarize this document in 3 bullet points", [
                ("assistant", "Reading the document carefully."),
                ("assistant", "This document covers many interesting topics."),
                ("assistant", "Here's a 500-word detailed analysis of each section with historical context."),
            ]),
            ("Add input validation to the signup form (email, password min 8 chars)", [
                ("assistant", "Looking at the signup form."),
                ("assistant", "I'll add a full form library with i18n support."),
                ("assistant", "Now implementing a custom date picker component."),
            ]),
            ("List the top 5 Python web frameworks", [
                ("assistant", "Let me research Python web frameworks."),
                ("assistant", "Actually, let me compare all 50+ Python web frameworks."),
                ("assistant", "Here's a deep dive into the history of web development since 1995."),
            ]),
            ("Convert this function from sync to async", [
                ("assistant", "Analyzing the function."),
                ("assistant", "I think we need to restructure the whole module."),
                ("assistant", "Let me implement a custom event loop from scratch."),
            ]),
            ("Add a dark mode toggle to the settings page", [
                ("assistant", "Looking at the settings page."),
                ("assistant", "The UI could use a complete redesign."),
                ("assistant", "Implementing a full theming engine with CSS-in-JS."),
            ]),
            ("Write a docstring for the calculate_tax function", [
                ("assistant", "Reading the function."),
                ("assistant", "The tax calculation logic seems wrong."),
                ("assistant", "Let me rewrite the entire tax calculation system."),
            ]),
            ("Rename the variable 'x' to 'user_count' in utils.py", [
                ("assistant", "Found the variable in utils.py."),
                ("assistant", "While I'm here, let me refactor the whole file."),
                ("assistant", "Splitting utils.py into 5 separate modules."),
            ]),
            ("Update the README with installation instructions", [
                ("assistant", "Reading the current README."),
                ("assistant", "The project structure needs reorganization."),
                ("assistant", "Let me set up a documentation site with Sphinx."),
            ]),
            ("Add a loading spinner when the API call is in progress", [
                ("assistant", "Looking at the API call code."),
                ("assistant", "We should implement a global state management solution."),
                ("assistant", "Setting up Redux with middleware and sagas."),
            ]),
            ("Fix the off-by-one error in the pagination logic", [
                ("assistant", "Found the pagination code."),
                ("assistant", "The pagination approach is outdated."),
                ("assistant", "Implementing cursor-based pagination with GraphQL."),
            ]),
            ("Add a 'created_at' timestamp column to the users table", [
                ("assistant", "Looking at the database schema."),
                ("assistant", "The schema design has several issues."),
                ("assistant", "Redesigning the entire database with a new ORM."),
            ]),
            ("Remove the deprecated 'legacy_login' endpoint", [
                ("assistant", "Found the legacy_login endpoint."),
                ("assistant", "The auth system needs modernization."),
                ("assistant", "Building a new auth service with JWT, OAuth2, SAML."),
            ]),
        ]

        caught = 0
        for task, msgs in drifted_convos:
            history = self._make_drifted_conversation(task, msgs)
            refresher = ContextRefresher()
            result = refresher.execute(history, _make_state(), budget, client)
            if result.queue_deposit is not None:
                caught += 1

        assert caught >= 11, f"Expected >=11/15 drift detections, got {caught}"

    def test_on_track_returns_none(self):
        from bicameral_agent.gemini import GeminiClient

        client = GeminiClient()
        budget = TokenBudget(max_calls=1, max_input_tokens=2000, max_output_tokens=500)

        on_track_convos = [
            ("Build a REST API with pagination", [
                ("assistant", "Setting up the Flask project structure."),
                ("assistant", "Implementing the pagination logic for the list endpoint."),
                ("assistant", "Adding query parameters for page and page_size."),
            ]),
            ("Write unit tests for the sort function", [
                ("assistant", "Reading the sort function implementation."),
                ("assistant", "Writing test for empty array input."),
                ("assistant", "Writing test for already sorted array."),
            ]),
            ("Fix the 404 error on the /users endpoint", [
                ("assistant", "Checking the route configuration."),
                ("assistant", "Found the issue: the route prefix was missing."),
                ("assistant", "Adding the correct route prefix to fix the 404."),
            ]),
            ("Add input validation for email field", [
                ("assistant", "Looking at the form component."),
                ("assistant", "Adding regex validation for email format."),
                ("assistant", "Adding error message display for invalid emails."),
            ]),
            ("Refactor the database connection to use connection pooling", [
                ("assistant", "Reviewing current database connection code."),
                ("assistant", "Configuring SQLAlchemy connection pool settings."),
                ("assistant", "Updating the connection factory to use the pool."),
            ]),
            ("Add a logout button to the navbar", [
                ("assistant", "Finding the navbar component."),
                ("assistant", "Adding the logout button with onClick handler."),
                ("assistant", "Implementing the logout API call and redirect."),
            ]),
            ("Convert the callback-based code to use promises", [
                ("assistant", "Identifying the callback patterns in the code."),
                ("assistant", "Wrapping the first callback in a Promise."),
                ("assistant", "Converting the remaining callbacks to async/await."),
            ]),
            ("Add error handling to the file upload endpoint", [
                ("assistant", "Reading the file upload endpoint code."),
                ("assistant", "Adding try-catch for file size validation."),
                ("assistant", "Adding error response for unsupported file types."),
            ]),
            ("Update the copyright year in the footer", [
                ("assistant", "Found the footer component."),
                ("assistant", "Updating the year from 2024 to 2025."),
                ("assistant", "Done. The footer now shows the correct year."),
            ]),
            ("Add a health check endpoint that returns 200 OK", [
                ("assistant", "Adding a /health route to the app."),
                ("assistant", "Implementing the handler to return 200 with status info."),
                ("assistant", "Adding a basic database connectivity check."),
            ]),
            ("Sort the imports alphabetically in main.py", [
                ("assistant", "Reading the current imports in main.py."),
                ("assistant", "Reordering the imports alphabetically."),
                ("assistant", "Grouped by standard lib, third-party, and local."),
            ]),
            ("Add type hints to the calculate_total function", [
                ("assistant", "Reading the calculate_total function."),
                ("assistant", "Adding parameter type hints: items: list[Item], tax_rate: float."),
                ("assistant", "Adding return type hint: -> Decimal."),
            ]),
            ("Increase the request timeout from 30s to 60s", [
                ("assistant", "Finding the timeout configuration."),
                ("assistant", "Updating the timeout value from 30 to 60 seconds."),
                ("assistant", "Verified the change in the config file."),
            ]),
            ("Remove unused imports from utils.py", [
                ("assistant", "Scanning utils.py for unused imports."),
                ("assistant", "Found 3 unused imports: os, sys, json."),
                ("assistant", "Removed the unused imports."),
            ]),
            ("Add a default value of 0 for the quantity parameter", [
                ("assistant", "Finding the quantity parameter definition."),
                ("assistant", "Setting the default value to 0."),
                ("assistant", "Updated the docstring to mention the default."),
            ]),
        ]

        none_count = 0
        for task, msgs in on_track_convos:
            history = self._make_drifted_conversation(task, msgs)
            refresher = ContextRefresher()
            result = refresher.execute(history, _make_state(), budget, client)
            if result.queue_deposit is None:
                none_count += 1

        assert none_count >= 12, f"Expected >=12/15 None for on-track, got {none_count}"

    def test_reminders_within_100_words(self):
        from bicameral_agent.gemini import GeminiClient

        client = GeminiClient()
        budget = TokenBudget(max_calls=1, max_input_tokens=2000, max_output_tokens=500)

        # Use a few clearly drifted conversations
        drifted = [
            ("Fix the login bug", [
                ("assistant", "Let me redesign the entire auth system."),
                ("assistant", "Adding microservices architecture."),
                ("assistant", "Setting up Kubernetes deployment."),
            ]),
            ("Add a button to the form", [
                ("assistant", "The form needs a complete rewrite."),
                ("assistant", "Implementing a design system."),
                ("assistant", "Building a component library from scratch."),
            ]),
        ]

        for task, msgs in drifted:
            history = self._make_drifted_conversation(task, msgs)
            refresher = ContextRefresher()
            result = refresher.execute(history, _make_state(), budget, client)
            if result.queue_deposit is not None:
                word_count = len(result.queue_deposit.content.split())
                assert word_count <= 100, f"Reminder too long: {word_count} words"
