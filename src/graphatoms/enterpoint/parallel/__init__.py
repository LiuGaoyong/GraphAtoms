import random
import time

from graphatoms.enterpoint.parallel._base import (
    MultiprocessingExecutor,
    ParallelExecutorABC,
    SerialExecutor,
)

__all__ = [
    "SerialExecutor",
    "MultiprocessingExecutor",
    "ParallelExecutorABC",
    "get_executor",
]


def get_executor(
    backend: str,
    nworkers: int | None = None,
    *args,
    **kwargs,
) -> ParallelExecutorABC:
    """Create an executor by backend backend.

    Args:
        backend: Backend backend. One of 'serial', 'multiprocessing', 'ray',
            'dask', 'executorlib'.
        nworkers: Maximum number of workers.
        **kwargs: Backend-specific arguments.

    Returns:
        An executor instance.
    """
    if backend in ("ray",):
        from graphatoms.enterpoint.parallel._ray import RayExecutor

        return RayExecutor(nworkers=nworkers, **kwargs)  # type: ignore[arg-type]

    elif backend in ("dask",):
        from graphatoms.enterpoint.parallel._dask import DaskExecutor

        return DaskExecutor(nworkers=nworkers, **kwargs)  # type: ignore[arg-type]

    elif backend in ("executorlib",):
        from graphatoms.enterpoint.parallel._executorlib import (
            ExecutorLibExecutor,
        )

        return ExecutorLibExecutor(  # type: ignore[arg-type]
            max_workers=nworkers, **kwargs
        )

    elif backend in ("serial",):
        return SerialExecutor(**kwargs)  # type: ignore[call-arg]

    elif backend in ("multiprocessing",):
        if nworkers is None or nworkers <= 0:
            nworkers = None
        return MultiprocessingExecutor(nworkers=nworkers, **kwargs)  # type: ignore[call-arg]

    else:
        raise ValueError(
            f"Unknown backend: {backend} Available: "
            f"{['serial', 'multiprocessing', 'ray', 'dask', 'executorlib']}"
        )
