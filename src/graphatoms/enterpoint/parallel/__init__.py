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
from graphatoms.enterpoint.parallel._ray import RayExecutor, ray

__all__ = [
    "SerialExecutor",
    "MultiprocessingExecutor",
    "ParallelExecutorABC",
    "get_executor",
]
BACKENDS = [
    "serial",
    "multiprocessing",
    "ray",
    "dask",
    "executorlib",
    "joblib",
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
    # `ncpus_for_tasks` only applies to the ray backend; `RunnerABC` passes
    # it unconditionally, so drop it for the other backends instead of
    # forwarding it to `ProcessPoolExecutor`.
    kwargs.pop("ncpus_for_tasks", None)
    if backend in ("dask",):
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
        raise ValueError(f"Unknown backend: {backend} Available: {BACKENDS}")


def executor_map(
    fn: Callable[..., Any],
    fn_params: Sequence[dict[str, Any]],
    executor: ParallelExecutorABC,
    *args,
    only_submit: bool = False,
    **kwargs,
) -> Sequence[Any]:
    if isinstance(executor, RayExecutor):
        dct: dict[int, ray.ObjectRef] = {}
        # Hold a strong reference to every deduplicated object.  Keying the
        # cache on `id(v)` alone is only safe while the objects stay alive;
        # if an intermediate object were collected, CPython could hand its
        # `id` to a *different* object and the wrong value would be
        # substituted into a task.
        keepalive: list[Any] = []
        remote_params: list[dict[str, ray.ObjectRef]] = []
        for params in fn_params:
            item: dict[str, ray.ObjectRef] = {}
            for k, v in params.items():
                if id(v) not in dct:
                    dct[id(v)] = ray.put(v)
                    keepalive.append(v)
                item[k] = dct[id(v)]
            remote_params.append(item)
        fn_params = remote_params
    futures = [executor.submit(fn, **d) for d in fn_params]
    if only_submit:
        return futures
    else:
        results = []
        while len(futures) > 0:
            result, futures = executor.wait(futures)
            results.append(result)
        return results


def get_executor_map(
    backend: str,
    fn: Callable[..., Any],
    fn_params: Sequence[dict[str, Any]],
    *args,
    nworkers: int | None = None,
    only_submit: bool = False,
    **kwargs,
) -> Sequence[Any]:
    backend = str(backend).lower()
    if backend in ("joblib",):
        if only_submit:
            raise RuntimeError(
                "joblib backend does not support `only_submit=True`."
            )
        lst = [delayed(fn)(**params) for params in fn_params]
        return list(Parallel(n_jobs=nworkers, **kwargs)(lst))
    else:
        return executor_map(
            fn=fn,
            fn_params=fn_params,
            executor=get_executor(
                backend=backend,
                nworkers=nworkers,
                **kwargs,
            ),
            only_submit=only_submit,
        )


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
        if backend == "joblib":
            continue
        print(
            get_executor_map(
                backend,
                _add,
                [dict(x=1, y=i) for i in range(4)],
                nworkers=4,
                only_submit=True,
            )
        )
        print("#" * 60)
