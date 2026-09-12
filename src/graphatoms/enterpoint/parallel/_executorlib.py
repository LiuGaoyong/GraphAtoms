"""executorlib-backed execution backend.

executorlib is an *optional* dependency. If it cannot be imported the module
still loads, but instantiating :class:`ExecutorLibExecutor` raises a clear
:class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graphatoms.enterpoint.parallel._base import ParallelExecutorABC

try:
    from executorlib import (
        SingleNodeExecutor as _ExecutorLibExecutorBase,  # type: ignore
    )

    _EXECUTORLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when executorlib absent
    _ExecutorLibExecutorBase = None  # type: ignore[assignment]
    _EXECUTORLIB_AVAILABLE = False


class ExecutorLibExecutor(ParallelExecutorABC):
    """An executor backed by :class:`executorlib.Executor`."""

    raise NotImplementedError("ExecutorLibExecutor is not implemented.")

    def __init__(self, *, nworkers: int | None = None, **kwargs) -> None:
        if not _EXECUTORLIB_AVAILABLE:
            raise ImportError(
                "executorlib is not installed. Install it with: "
                "pip install executorlib"
            )
        self._executor = _ExecutorLibExecutorBase(  # type: ignore
            max_workers=nworkers, **kwargs
        )

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> object:
        raise NotImplementedError(
            "ExecutorLibExecutor.submit is not implemented."
        )
