"""Utility functions for the parallel execution framework.

Provides ``as_completed`` and ``wait`` that work uniformly across all
backends (serial, multiprocessing, ray, dask, executorlib).
"""

from __future__ import annotations

import concurrent.futures as _cf
import time
from collections.abc import Iterator, Sequence
from typing import Any

from graphatoms.enterpoint.parallel.base import BaseFuture


def _detect_backend(futures: Sequence[BaseFuture]) -> str:
    """Detect which backend the futures belong to."""
    if not futures:
        return "unknown"
    first = futures[0]
    # Ray futures
    if hasattr(first, "_obj_ref"):
        return "ray"
    # Dask futures
    if hasattr(first, "_future") and hasattr(first, "_client"):
        return "dask"
    # Standard library futures (concurrent.futures.Future)
    if isinstance(first, _cf.Future):
        return "cf"
    return "unknown"


def _poll_done(f: BaseFuture) -> bool:
    """Return True if the future has finished."""
    return f.done()


def as_completed(
    futures: Sequence[BaseFuture],
    timeout: float | None = None,
) -> Iterator[BaseFuture]:
    """Yield futures as they complete.

    Similar to ``concurrent.futures.as_completed`` but works across all
    backends. Futures are yielded in completion order, not submission
    order.
    """
    backend = _detect_backend(futures)
    pending = list(futures)

    if backend == "ray":
        import ray

        # ray.wait operates on ObjectRefs, not the wrapper futures.
        pending_wrappers = list(pending)
        pending_refs = [f._obj_ref for f in pending_wrappers]  # type: ignore[attr-defined]
        while pending_refs:
            ready_refs, pending_refs = ray.wait(
                pending_refs, num_returns=1, timeout=timeout
            )
            if not ready_refs:
                break
            ready_ref = ready_refs[0]
            for f in list(pending_wrappers):
                if f._obj_ref == ready_ref and not f._fetched:  # type: ignore[attr-defined]
                    pending_wrappers.remove(f)
                    yield f
                    break
        return

    # Standard library, dask, or unknown: use polling. The native
    # dask.distributed.wait API uses sentinel constants (not the
    # ``return_when`` string) and is version-sensitive, so polling is
    # used for portability across all remaining backends.
    deadline = time.monotonic() + timeout if timeout else None
    while pending:
        if deadline and time.monotonic() > deadline:
            break
        for f in list(pending):
            if _poll_done(f):
                pending.remove(f)
                yield f
        if pending:
            time.sleep(0.01)


def wait_one(
    futures: Sequence[BaseFuture],
    timeout: float | None = None,
) -> tuple[Any, list[BaseFuture]]:
    done, pending = wait(futures, timeout=timeout, num_returns=1)
    assert isinstance(done, list) and len(done) == 1
    result = done[0]

    if isinstance(result, BaseFuture):
        result = result.result()
    assert not isinstance(result, BaseFuture)
    return result, pending


def wait(
    futures: Sequence[BaseFuture],
    timeout: float | None = None,
    num_returns: int = 1,
) -> tuple[list[BaseFuture], list[BaseFuture]]:
    """Wait until at least ``num_returns`` futures complete.

    Returns a tuple of (done, not_done) lists.
    """
    backend = _detect_backend(futures)
    pending = list(futures)

    if backend == "ray":
        import ray

        refs = [f._obj_ref for f in pending]  # type: ignore[attr-defined]
        if not refs:
            return [], []
        n = max(1, min(num_returns, len(refs)))
        ready_refs, _ = ray.wait(refs, num_returns=n, timeout=timeout)
        done = [f for f in futures if f._obj_ref in ready_refs]  # type: ignore[attr-defined]
        not_done = [
            f
            for f in futures
            if f._obj_ref not in ready_refs  # type: ignore[attr-defined]
        ]
        return done, not_done

    # Standard library, dask, or unknown: poll for completion.
    deadline = time.monotonic() + timeout if timeout else None
    done: list[BaseFuture] = []
    while len(done) < num_returns and pending:
        if deadline and time.monotonic() > deadline:
            break
        for f in list(pending):
            if _poll_done(f):
                done.append(f)
                pending.remove(f)
                if len(done) >= num_returns:
                    break
        if len(done) < num_returns and pending:
            time.sleep(0.01)
    return done, pending
