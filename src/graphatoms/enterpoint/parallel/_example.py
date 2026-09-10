"""Example: KMC batch submit → consume first N → cancel rest.

This module demonstrates the canonical KMC parallel pattern: submit a
large batch of candidate calculations, consume the first ``n_results``
that complete, then cancel the remaining tasks.

The pattern works identically across all backends (serial,
multiprocessing, ray, dask, executorlib) thanks to the unified
``BaseExecutor`` / ``BaseFuture`` interface.
"""

from __future__ import annotations

import time
from typing import Any

from graphatoms.enterpoint.parallel import as_completed, get_executor


def _sample_task(index: int) -> int:
    """A sample task that simulates varying computation time."""
    time.sleep(0.01 * (index % 10))
    return index * 2


def run_batch_consume_cancel(
    backend: str = "serial",
    n_tasks: int = 100,
    n_results: int = 5,
    **kwargs: Any,
) -> list[int]:
    """Submit *n_tasks* tasks, consume first *n_results*, cancel rest.

    Args:
        backend: Backend name (serial, multiprocessing, ray, dask,
            executorlib).
        n_tasks: Total number of tasks to submit.
        n_results: Number of results to consume before cancelling.
        **kwargs: Passed to ``get_executor``.

    Returns:
        List of results from the first *n_results* completed tasks.
    """
    with get_executor(backend, **kwargs) as executor:
        futures = [executor.submit(_sample_task, i) for i in range(n_tasks)]

        results: list[int] = []
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) >= n_results:
                break

        # Cancel remaining tasks
        for f in futures:
            if not f.done():
                f.cancel()

    return results


def run_batch_wait_cancel(
    backend: str = "serial",
    n_tasks: int = 100,
    n_results: int = 5,
    **kwargs: Any,
) -> list[int]:
    """Submit tasks, use ``wait`` to drain results, cancel rest.

    Uses :func:`graphatoms.parallel.wait` instead of ``as_completed`` to
    show the alternative non-blocking pattern.
    """
    from graphatoms.enterpoint.parallel import wait

    with get_executor(backend, **kwargs) as executor:
        futures = [executor.submit(_sample_task, i) for i in range(n_tasks)]

        results: list[int] = []
        pending = list(futures)
        while len(results) < n_results and pending:
            done, pending = wait(pending, num_returns=1)
            for f in done:
                results.append(f.result())
                if len(results) >= n_results:
                    break

        for f in pending:
            if not f.done():
                f.cancel()

    return results


if __name__ == "__main__":
    # Demo with serial backend (no extra dependencies needed)
    print("=== as_completed pattern ===")
    res = run_batch_consume_cancel("multiprocessing", n_tasks=20, n_results=3)
    print(f"Results: {res}")

    print("\n=== wait pattern ===")
    res2 = run_batch_wait_cancel("multiprocessing", n_tasks=20, n_results=3)
    print(f"Results: {res2}")
