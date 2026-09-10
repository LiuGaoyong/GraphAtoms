"""Abstract base classes for the parallel execution framework.

These mirror the public API of :mod:`concurrent.futures` so that every
backend (serial, multiprocessing, ray, dask, executorlib) can be used
through a single, uniform interface.
"""

from __future__ import annotations

from concurrent.futures import Executor as BaseExecutor
from concurrent.futures import Future as BaseFuture
from concurrent.futures import ProcessPoolExecutor

__all__ = [
    "BaseExecutor",
    "BaseFuture",
    "SerialExecutor",
    "ProcessPoolExecutor",
]


class SerialExecutor(ProcessPoolExecutor):
    """Serial executor: a ProcessPoolExecutor with a single worker."""

    def __init__(self, **kwargs) -> None:
        ProcessPoolExecutor.__init__(
            self,
            max_workers=1,
            **kwargs,
        )
