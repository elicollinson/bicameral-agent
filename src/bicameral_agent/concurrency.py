"""Context-propagating thread-pool submission (issue #91).

Per-episode state lives in ``contextvars`` ContextVars: the degradation
counters in ``llm_output`` and the episode cost accumulators in
``cost_tracker``. ``ThreadPoolExecutor`` worker threads do NOT inherit the
submitter's context -- each thread runs in its own context, created empty at
thread start -- so work submitted with a plain ``pool.submit(fn, ...)``
silently escapes the submitting episode's counters and cost accounting.
This module names and enforces that contract in one place.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import Executor, Future
from typing import Callable, TypeVar

T = TypeVar("T")


def submit_in_context(
    pool: Executor, fn: Callable[..., T], /, *args, **kwargs
) -> Future[T]:
    """Submit ``fn(*args, **kwargs)`` to ``pool``, run in a copy of the
    caller's ``contextvars`` context.

    Use this for ALL episode-scoped pool work: it is what makes
    context-local per-episode state (degradation counters, episode cost
    accumulators) attribute worker-thread activity to the submitting
    episode. A fresh ``copy_context()`` is captured per submission, so
    concurrent tasks never share a Context object (``Context.run`` may not
    be entered concurrently) and mutations of context variables inside the
    worker never leak back to the submitter.
    """
    return pool.submit(contextvars.copy_context().run, fn, *args, **kwargs)
