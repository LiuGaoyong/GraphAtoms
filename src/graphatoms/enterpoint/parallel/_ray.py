import os
from collections.abc import Callable, Sequence
from typing import Any, override

from graphatoms.enterpoint.parallel._base import ParallelExecutorABC

try:
    os.environ["RAY_DEDUP_LOGS"] = "0"

    import ray

    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when ray absent
    ray = None  # type: ignore[assignment]
    _RAY_AVAILABLE = False


class RayExecutor(ParallelExecutorABC):
    """An executor backed by :mod:`ray`."""

    @override
    def __init__(self, *, nworkers: int | None = None, **kwargs) -> None:
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
        remote_fn = ray.remote(func)  # type: ignore
        remote_fn = remote_fn.options(num_cpus=1)
        obj_ref = remote_fn.remote(*args, **kwargs)
        return obj_ref

    @override
    def get(self, future: ray.ObjectRef) -> Any:  # type: ignore
        return ray.get(future)  # type: ignore

    @override
    def cancel(self, future: ray.ObjectRef) -> None:  # type: ignore
        ray.cancel(future, force=True, recursive=True)  # type: ignore

    @override
    def wait(
        self,
        futures: Sequence[ray.ObjectRef],  # type: ignore
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> tuple[Any, Sequence[ray.ObjectRef]]:  # type: ignore
        done, unfinished = ray.wait(futures, timeout=timeout)  # type: ignore
        return ray.get(done)[0], unfinished  # type: ignore

    @override
    def map(self, fn, *iterables, timeout=None, chunksize=1) -> list[Any]:
        return ray.get([self.submit(fn, *args) for args in zip(*iterables)])  # type: ignore
