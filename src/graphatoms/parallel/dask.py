"""Dask-backed execution backend.

Dask is an *optional* dependency. If it cannot be imported the module still
loads, but instantiating :class:`DaskExecutor` raises a clear
:class:`ImportError`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from typing import Any

from graphatoms.parallel.abc import BaseExecutor, BaseFuture

try:
    import dask.config  # type: ignore[import]
    from dask.distributed import Client  # type: ignore[import]

    _DASK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when dask absent
    Client = None  # type: ignore[assignment]
    _DASK_AVAILABLE = False


class DaskFuture(BaseFuture):
    """A future wrapping a :class:`dask.distributed.Future`."""

    def __init__(self, future: Any, client: Any = None) -> None:
        self._future = future
        self._client = client
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
        with dask.config.set({"logging.distributed": "error"}):  # type: ignore
            if self._client is not None:
                try:
                    self._client.cancel(self._future, force=True)
                except Exception:  # noqa: BLE001
                    return False
            else:
                try:
                    self._future.cancel()
                except Exception:  # noqa: BLE001
                    return False
        # dask's Future.cancel() has no return value; check the status
        # to determine whether cancellation actually succeeded.
        cancelled = self._future.status == "cancelled"
        self._cancelled = cancelled
        return cancelled

    def cancelled(self) -> bool:
        return self._cancelled

    def running(self) -> bool:
        # dask status: "pending" (queued), "processing" (running),
        # "finished", "error", "cancelled". "processing" means actively
        # executing.
        return self._future.status == "processing"

    def done(self) -> bool:
        return self._future.done()

    def add_done_callback(self, fn: Callable[[BaseFuture], Any]) -> None:
        self._future.add_done_callback(lambda _f: fn(self))


class DaskExecutor(BaseExecutor):
    """An executor backed by :class:`dask.distributed.Client`."""

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        if not _DASK_AVAILABLE:
            raise ImportError(
                "dask is not installed. Install it with: "
                "pip install dask distributed"
            )
        self._client = Client(
            n_workers=max_workers,
            silence_logs=logging.ERROR,
            **kwargs,
        )  # type: ignore

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> DaskFuture:
        future = self._client.submit(fn, *args, **kwargs)
        return DaskFuture(future, self._client)

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Any,
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[Any]:
        futures = self._client.map(fn, *iterables)
        return (f.result(timeout=timeout) for f in futures)

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        self._client.close()
        del wait, cancel_futures

    def cancel(
        self,
        future: BaseFuture,
        *,
        force: bool = False,
        recursive: bool = False,
    ) -> bool:
        """Cancel a Dask future via the client."""
        del recursive
        if not isinstance(future, DaskFuture):
            return future.cancel()
        if future._cancelled or future.done():
            return False

        with dask.config.set({"logging.distributed": "error"}):  # type: ignore
            try:
                future._client.cancel(future._future, force=force)
            except Exception:  # noqa: BLE001
                return False

        future._cancelled = True
        return True
