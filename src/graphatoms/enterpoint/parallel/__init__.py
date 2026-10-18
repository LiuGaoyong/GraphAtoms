from collections.abc import Callable, Sequence
from typing import Any

from joblib import Parallel, delayed

from graphatoms.enterpoint.parallel._base import (
    MultiprocessingExecutor,
    ParallelExecutorABC,
    SerialExecutor,
)
from graphatoms.enterpoint.parallel._dask import DaskExecutor
from graphatoms.enterpoint.parallel._executorlib import ExecutorLibExecutor
from graphatoms.enterpoint.parallel._ray import _RAY_AVAILABLE, RayExecutor, ray

__all__ = [
    "SerialExecutor",
    "MultiprocessingExecutor",
    "ParallelExecutorABC",
    "get_executor",
]


def get_executor(
    backend: str,
    *args,
    nworkers: int | None = None,
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
    backend = str(backend).lower()
    if backend in ("ray",):
        return RayExecutor(nworkers=nworkers, **kwargs)  # type: ignore[arg-type]
    elif backend in ("dask",):
        return DaskExecutor(nworkers=nworkers, **kwargs)  # type: ignore[arg-type]
    elif backend in ("executorlib",):
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


def get_executor_map(
    backend: str,
    fn: Callable[..., Any],
    fn_params: Sequence[dict[str, Any]],
    *args,
    nworkers: int | None = None,
    **kwargs,
) -> Sequence[Any]:
    backend = str(backend).lower()
    if backend in ("joblib",):
        lst = [delayed(fn)(**params) for params in fn_params]
        return list(Parallel(n_jobs=nworkers, **kwargs)(lst))
    elif backend in ("ray",):
        if not _RAY_AVAILABLE:
            raise ImportError(
                "ray is not installed. Install it with: pip install ray"
            )
        get_executor(backend, nworkers=nworkers, **kwargs)
        dct: dict[int, ray.ObjectRef] = {}
        remote_params: list[dict[str, ray.ObjectRef]] = []
        for params in fn_params:
            item: dict[str, ray.ObjectRef] = {}
            for k, v in params.items():
                if id(v) not in dct:
                    dct[id(v)] = ray.put(v)
                item[k] = dct[id(v)]
            remote_params.append(item)
        remote_fn = ray.remote(fn).options(num_cpus=1)
        print(dct)
        return ray.get([remote_fn.remote(**d) for d in remote_params])
    else:
        executor = get_executor(backend, nworkers=nworkers, **kwargs)
        futures = [executor.submit(fn, **d) for d in fn_params]
        results = []
        while len(futures) > 0:
            result, futures = executor.wait(futures)
            results.append(result)
        return results


def _add(x, y) -> Any:
    import time

    time.sleep(0.011)
    return x + y


if __name__ == "__main__":
    import time

    for backend in ["serial", "multiprocessing", "joblib", "ray"]:
        for i in range(4):
            start_time = time.perf_counter()
            result = get_executor_map(
                backend,
                _add,
                [dict(x=1, y=i) for i in range(4)],
                nworkers=4,
            )
            end_time = time.perf_counter()
            print(
                result,
                end_time - start_time,
            )
            print("=" * 50)
        print("#" * 60)
