"""Abstract base classes for the parallel execution framework.

These mirror the public API of :mod:`concurrent.futures` so that every
backend (serial, multiprocessing, ray, dask, executorlib) can be used
through a single, uniform interface.
"""

from __future__ import annotations

import concurrent.futures as _cf
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable, Iterator


class BaseFuture(ABC):
    """Abstract base class for Future objects.

    Mirrors :class:`concurrent.futures.Future`.
    """

    @abstractmethod
    def result(self, timeout: float | None = None) -> Any:
        """Return the result of the call that the future represents.

        Args:
            timeout: The maximum number of seconds to wait for the result.
                ``None`` means wait forever.

        Returns:
            The result of the call.

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future did not complete within ``timeout``.
            Exception: If the call raised, this method raises the same
                exception.
        """

    @abstractmethod
    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Return the exception raised by the call.

        Args:
            timeout: The maximum number of seconds to wait. ``None`` means
                wait forever.

        Returns:
            The exception raised by the call, or ``None`` if the call
            completed without raising.

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future did not complete within ``timeout``.
        """

    @abstractmethod
    def cancel(self) -> bool:
        """Attempt to cancel the call.

        Returns:
            ``True`` if the call was successfully cancelled, ``False``
            otherwise (for example if the call has already finished or is
            currently running and cannot be cancelled).
        """

    @abstractmethod
    def cancelled(self) -> bool:
        """Return ``True`` if the call was successfully cancelled."""

    @abstractmethod
    def running(self) -> bool:
        """Return ``True`` if the call is currently being executed."""

    @abstractmethod
    def done(self) -> bool:
        """Return ``True`` if the call has finished (cancelled or ran)."""

    @abstractmethod
    def add_done_callback(self, fn: Callable[[BaseFuture], Any]) -> None:
        """Attach a callback that will be called when the future is done.

        The callback receives the future as its only argument. If the future
        is already done, the callback is called immediately.
        """

    def set_result(self, result: Any) -> None:  # noqa: D401
        """Set the result of the future.

        Not all backends support setting the result externally. The default
        implementation raises :class:`NotImplementedError`.
        """
        raise NotImplementedError

    def set_exception(self, exception: BaseException) -> None:  # noqa: D401
        """Set the exception of the future.

        Not all backends support setting the exception externally. The
        default implementation raises :class:`NotImplementedError`.
        """
        raise NotImplementedError

    def set_running_or_notify_cancel(self) -> bool:  # noqa: D401
        """Mark the future as running or notify cancellation.

        Not all backends support this internal hook. The default
        implementation raises :class:`NotImplementedError`.
        """
        raise NotImplementedError


class BaseExecutor(ABC):
    """Abstract base class for Executor objects.

    Mirrors :class:`concurrent.futures.Executor`.
    """

    @abstractmethod
    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> BaseFuture:
        """Schedule ``fn`` to be executed as ``fn(*args, **kwargs)``.

        Returns:
            A :class:`BaseFuture` representing the execution of the call.
        """

    @abstractmethod
    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Any,
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[Any]:
        """Apply ``fn`` to every element of ``iterables`` in parallel.

        Returns an iterator that yields the results in the same order as the
        input iterables.
        """

    @abstractmethod
    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        """Free any resources used by the executor.

        Args:
            wait: If ``True``, wait for all pending futures to finish.
            cancel_futures: If ``True``, cancel all pending futures that have
                not yet started running.
        """

    def cancel(
        self,
        future: BaseFuture,
        *,
        force: bool = False,
        recursive: bool = False,
    ) -> bool:
        """Attempt to cancel a submitted future.

        Args:
            future: The future to cancel.
            force: If True, forcefully cancel even if the task is running.
            recursive: If True, recursively cancel dependent tasks.

        Returns:
            True if the future was successfully cancelled, False otherwise.

        The default implementation delegates to ``future.cancel()``.
        Backends with native force/recursive support (Ray, Dask) override
        this method.
        """
        return future.cancel()

    def __enter__(self) -> BaseExecutor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


# Register the standard library Future as a virtual subclass of BaseFuture
# so that backends returning native concurrent.futures.Future objects
# (multiprocessing) satisfy isinstance checks.
BaseFuture.register(_cf.Future)
