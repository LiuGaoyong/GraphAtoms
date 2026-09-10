"""executorlib-backed execution backend.

executorlib is an *optional* dependency. If it cannot be imported the module
still loads, but instantiating :class:`ExecutorLibExecutor` raises a clear
:class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from typing import Any

from graphatoms.parallel.base import BaseExecutor, BaseFuture

try:
    from executorlib import (
        SingleNodeExecutor as _ExecutorLibExecutorBase,  # type: ignore
    )

    _EXECUTORLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when executorlib absent
    _ExecutorLibExecutorBase = None  # type: ignore[assignment]
    _EXECUTORLIB_AVAILABLE = False


class ExecutorLibFuture(BaseFuture):
    """A future wrapping the future returned by executorlib."""

    def __init__(self, future: Any) -> None:
        self._future = future
        self._cancelled = False

    def result(self, timeout: float | None = None) -> Any:
        if self._cancelled:
            raise CancelledError()
        return self._future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if self._cancelled:
            raise CancelledError()
        return self._future.exception(timeout=timeout)

    def cancel(self) -> bool:
        if self._cancelled or self.done():
            return False
        try:
            cancelled = self._future.cancel()
        except Exception:  # noqa: BLE001
            cancelled = False
        self._cancelled = bool(cancelled)
        return self._cancelled

    def cancelled(self) -> bool:
        return self._cancelled

    def running(self) -> bool:
        return self._future.running()

    def done(self) -> bool:
        return self._future.done()

    def add_done_callback(self, fn: Callable[[BaseFuture], Any]) -> None:
        self._future.add_done_callback(lambda _f: fn(self))


class ExecutorLibExecutor(BaseExecutor):
    """An executor backed by :class:`executorlib.Executor`."""

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        if not _EXECUTORLIB_AVAILABLE:
            raise ImportError(
                "executorlib is not installed. Install it with: "
                "pip install executorlib"
            )
        self._executor = _ExecutorLibExecutorBase(  # type: ignore
            max_workers=max_workers, **kwargs
        )

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> ExecutorLibFuture:
        future = self._executor.submit(fn, *args, **kwargs)  # type: ignore
        return ExecutorLibFuture(future)

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Any,
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[Any]:
        return self._executor.map(
            fn, *iterables, timeout=timeout, chunksize=chunksize
        )

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        self._executor.shutdown(wait=wait)
        del cancel_futures
