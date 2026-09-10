"""Process-pool backend.

Backed by :class:`concurrent.futures.ProcessPoolExecutor`.

The native :class:`concurrent.futures.Future` already satisfies the
:class:`~graphatoms.parallel.base.BaseFuture` interface, so it is returned
directly from :meth:`ProcessPoolExecutor.submit`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from typing import Any

from graphatoms.parallel.abc import BaseExecutor, BaseFuture


class ProcessPoolExecutor(BaseExecutor):
    """An executor backed by
    :class:`concurrent.futures.ProcessPoolExecutor`."""

    def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
        self._executor = _ProcessPoolExecutor(max_workers=max_workers, **kwargs)

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> BaseFuture:
        return self._executor.submit(fn, *args, **kwargs)  # type: ignore

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
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
