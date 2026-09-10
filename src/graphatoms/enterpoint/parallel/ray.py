"""Ray-backed execution backend.

Ray is an *optional* dependency. If it cannot be imported the module still
loads, but instantiating :class:`RayExecutor` raises a clear
:class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from typing import Any

from graphatoms.enterpoint.parallel.base import BaseExecutor, BaseFuture

try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when ray absent
    ray = None  # type: ignore[assignment]
    _RAY_AVAILABLE = False


class RayFuture(BaseFuture):
    """A future wrapping a Ray ``ObjectRef``."""

    def __init__(self, obj_ref: Any) -> None:
        self._obj_ref = obj_ref
        self._result: Any = None
        self._exception: BaseException | None = None
        self._fetched = False
        self._cancelled = False
        self._callbacks: list[Callable[[BaseFuture], Any]] = []

    def _is_ready(self) -> bool:
        ready, _ = ray.wait([self._obj_ref], num_returns=1, timeout=0)  # type: ignore
        return bool(ready)

    def _fetch(self, timeout: float | None = None) -> None:
        """Retrieve the result from Ray and populate state."""
        if self._fetched:
            return
        try:
            self._result = ray.get(self._obj_ref, timeout=timeout)  # type: ignore
        except Exception as exc:  # noqa: BLE001
            # ray raises GetTimeoutError on timeout; normalise to the
            # standard library TimeoutError for API consistency.
            if timeout is not None and type(exc).__name__ == "GetTimeoutError":
                raise TimeoutError(str(exc)) from exc
            # ray wraps task exceptions in RayTaskError; unwrap to the
            # original exception so callers see the expected type.
            if type(exc).__name__.startswith("RayTaskError"):
                exc = exc.args[0] if exc.args else exc
            self._exception = exc
        finally:
            self._fetched = True
            for cb in self._callbacks:
                cb(self)

    def result(self, timeout: float | None = None) -> Any:
        if self._cancelled:
            raise CancelledError()
        self._fetch(timeout=timeout)
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if self._cancelled:
            raise CancelledError()
        self._fetch(timeout=timeout)
        return self._exception

    def cancel(
        self,
        *,
        force: bool = True,
        recursive: bool = True,
    ) -> bool:
        if self._fetched:
            return False
        try:
            ray.cancel(  # type: ignore
                self._obj_ref,
                force=force,
                recursive=recursive,
            )
        except Exception:  # noqa: BLE001
            pass
        self._cancelled = True
        self._fetched = True
        for cb in self._callbacks:
            cb(self)
        return True

    def cancelled(self) -> bool:
        return self._cancelled

    def running(self) -> bool:
        if self._fetched or self._cancelled:
            return False
        return not self._is_ready()

    def done(self) -> bool:
        if self._fetched or self._cancelled:
            return True
        return self._is_ready()

    def add_done_callback(self, fn: Callable[[BaseFuture], Any]) -> None:
        if self.done():
            fn(self)
        else:
            self._callbacks.append(fn)


class RayExecutor(BaseExecutor):
    """An executor backed by :mod:`ray`."""

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        if not _RAY_AVAILABLE:
            raise ImportError(
                "ray is not installed. Install it with: pip install ray"
            )
        if not ray.is_initialized():  # type: ignore
            ray.init(ignore_reinit_error=True, **kwargs)  # type: ignore
        self._max_workers = max_workers

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> RayFuture:
        remote_fn = ray.remote(fn)  # type: ignore
        if self._max_workers is not None:
            remote_fn = remote_fn.options(num_cpus=1)
        obj_ref = remote_fn.remote(*args, **kwargs)
        return RayFuture(obj_ref)

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Any,
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[Any]:
        del chunksize
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        return (f.result(timeout=timeout) for f in futures)

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        # Do not shut down the global Ray runtime: it may be shared with
        # other parts of the program.
        del wait, cancel_futures


