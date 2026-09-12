"""Parallel execution framework for graphatoms.

Provides a unified ``concurrent.futures``-compatible API across multiple
backends: serial, multiprocessing, ray, dask, executorlib.
"""

from __future__ import annotations

from graphatoms.enterpoint.parallel._utils import as_completed, wait, wait_one
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
    "wait_one",
    "get_executor",
]

_BACKENDS: dict[str, type[BaseExecutor]] = {
    "serial": SerialExecutor,
    "multiprocessing": ProcessPoolExecutor,
}


def get_executor(
    name: str,
    *,
    max_workers: int | None = None,
    **kwargs,
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

    elif name in ("serial",):
        return SerialExecutor(**kwargs)  # type: ignore[call-arg]

    elif name in ("multiprocessing",):
        if max_workers is None or max_workers <= 0:
            max_workers = None
        return ProcessPoolExecutor(max_workers=max_workers, **kwargs)  # type: ignore[call-arg]

    else:
        raise ValueError(
            f"Unknown backend: {name} Available: "
            f"{['serial', 'multiprocessing', 'ray', 'dask', 'executorlib']}"
        )


if __name__ == "__main__":
    from concurrent.futures import Executor

    from graphatoms.enterpoint.parallel.dask import DaskExecutor
    from graphatoms.enterpoint.parallel.ray import RayExecutor

    for cls in [SerialExecutor, ProcessPoolExecutor, RayExecutor, DaskExecutor]:
        assert issubclass(cls, Executor)
