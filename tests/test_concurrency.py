"""Tests for context-propagating pool submission (issue #91)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

from bicameral_agent.concurrency import submit_in_context

_var: ContextVar[str] = ContextVar("test_concurrency_var", default="unset")


class TestSubmitInContext:
    def test_worker_sees_submitters_context(self):
        token = _var.set("episode-A")
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                # Plain submit does not inherit the submitter's context...
                assert pool.submit(_var.get).result() == "unset"
                # ...submit_in_context does.
                assert submit_in_context(pool, _var.get).result() == "episode-A"
        finally:
            _var.reset(token)

    def test_context_captured_per_submission(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            token = _var.set("first")
            future_first = submit_in_context(pool, _var.get)
            _var.reset(token)
            token = _var.set("second")
            future_second = submit_in_context(pool, _var.get)
            _var.reset(token)
            assert future_first.result() == "first"
            assert future_second.result() == "second"

    def test_worker_mutations_do_not_leak_back(self):
        token = _var.set("outer")
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                submit_in_context(pool, _var.set, "inner").result()
            assert _var.get() == "outer"
        finally:
            _var.reset(token)

    def test_args_and_kwargs_forwarded(self):
        def combine(a, b, *, sep):
            return f"{a}{sep}{b}"

        with ThreadPoolExecutor(max_workers=1) as pool:
            assert submit_in_context(pool, combine, "x", "y", sep="-").result() == "x-y"
