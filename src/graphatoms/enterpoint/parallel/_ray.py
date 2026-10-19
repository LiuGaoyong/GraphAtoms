import os
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

from graphatoms.enterpoint.parallel._base import ParallelExecutorABC

try:
    os.environ["RAY_DEDUP_LOGS"] = "0"

    import ray

    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when ray absent
    ray = None  # type: ignore[assignment]
    _RAY_AVAILABLE = False
if TYPE_CHECKING:
    import ray  # type: ignore


class RayExecutor(ParallelExecutorABC):
    """An executor backed by :mod:`ray`."""

    @override
    def __init__(
        self,
        *,
        nworkers: int | None = None,
        ncpus_for_tasks: int = 1,
        **kwargs,
    ) -> None:
        self.ncpus_for_tasks = ncpus_for_tasks
        self._remote_fn_cache: dict[int, tuple[weakref.ref, Any]] = {}
        if not _RAY_AVAILABLE:
            raise ImportError(
                "ray is not installed. Install it with: pip install ray"
            )
        try:
            if not ray.is_initialized():  # type: ignore
                ray.init(  # type: ignore
                    ignore_reinit_error=True,
                    num_cpus=nworkers,
                    **kwargs,
                )
        except Exception:
            pass

    @override
    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> ray.ObjectRef:  # type: ignore
        key = id(func)
        entry = self._remote_fn_cache.get(key)
        if entry is not None:
            func_ref, remote_fn = entry  # type: ignore
            if func_ref() is func:  # func alive and cache hit
                return remote_fn.remote(*args, **kwargs)
            # id reused or func recycled, clean up cache
            del self._remote_fn_cache[key]

        remote_fn = ray.remote(func).options(  # type: ignore
            num_cpus=self.ncpus_for_tasks,
        )
        try:
            func_ref = weakref.ref(func)  # type: ignore
        except TypeError:

            def func_ref():  # type: ignore
                return func

        self._remote_fn_cache[key] = (func_ref, remote_fn)  # type: ignore
        return remote_fn.remote(*args, **kwargs)

    @override
    def get(self, future: ray.ObjectRef) -> Any:  # type: ignore
        return ray.get(future)  # type: ignore

    @override
    def cancel(self, future: ray.ObjectRef) -> None:  # type: ignore
        ray.cancel(future, force=True, recursive=True)  # type: ignore

    @override
    def wait(
        self,
        futures: list[ray.ObjectRef],  # type: ignore
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> tuple[Any, list[ray.ObjectRef]]:  # type: ignore
        done, unfinished = ray.wait(futures, timeout=timeout)  # type: ignore
        return ray.get(done)[0], unfinished  # type: ignore

    @override
    def map(self, fn, *iterables, timeout=None, chunksize=1) -> list[Any]:
        return ray.get([self.submit(fn, *args) for args in zip(*iterables)])  # type: ignore
