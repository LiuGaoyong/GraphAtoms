"""Parallel execution framework for graphatoms.

Provides a unified ``concurrent.futures``-compatible API across multiple
backends: serial, multiprocessing, ray, dask, executorlib.
"""

from __future__ import annotations

from graphatoms.enterpoint.parallel._utils import as_completed, wait
from graphatoms.enterpoint.parallel.base import (
    BaseExecutor,
    BaseFuture,
    ProcessPoolExecutor,
    SerialExecutor,
)

__all__ = [
    "BaseExecutor",
    "BaseFuture",
    "SerialExecutor",
    "ProcessPoolExecutor",
    "as_completed",
    "wait",
    "get_executor",
]

_BACKENDS: dict[str, type[BaseExecutor]] = {
    "serial": SerialExecutor,
    "multiprocessing": ProcessPoolExecutor,
}


def get_executor(
    name: str, *, max_workers: int | None = None, **kwargs: object
) -> BaseExecutor:
    """Create an executor by backend name.

    Args:
        name: Backend name. One of 'serial', 'multiprocessing', 'ray',
            'dask', 'executorlib'.
        max_workers: Maximum number of workers.
        **kwargs: Backend-specific arguments.

    Returns:
        An executor instance.
    """
    if name in ("ray",):
        from graphatoms.enterpoint.parallel.ray import RayExecutor

        return RayExecutor(max_workers=max_workers, **kwargs)  # type: ignore[arg-type]
    elif name in ("dask",):
        from graphatoms.enterpoint.parallel.dask import DaskExecutor

        return DaskExecutor(max_workers=max_workers, **kwargs)  # type: ignore[arg-type]
    elif name in ("executorlib",):
        from graphatoms.enterpoint.parallel.executorlib import (
            ExecutorLibExecutor,
        )

        return ExecutorLibExecutor(  # type: ignore[arg-type]
            max_workers=max_workers, **kwargs
        )

    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: "
            f"{list(_BACKENDS.keys()) + ['ray', 'dask', 'executorlib']}"
        )

    cls = _BACKENDS[name]
    if name in ("serial",):
        return cls(**kwargs)  # type: ignore[call-arg]
    return cls(max_workers=max_workers or 1, **kwargs)  # type: ignore[call-arg]


if __name__ == "__main__":
    from concurrent.futures import Executor

    from graphatoms.enterpoint.parallel.dask import DaskExecutor
    from graphatoms.enterpoint.parallel.ray import RayExecutor

    for cls in [SerialExecutor, ProcessPoolExecutor, RayExecutor, DaskExecutor]:
        assert issubclass(cls, Executor)
