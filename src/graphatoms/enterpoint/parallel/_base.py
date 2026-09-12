import concurrent.futures as cf
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, override


class ParallelExecutorABC(ABC):
    @abstractmethod
    def __init__(self, *, nworkers: int | None = None, **kwargs) -> None:
        """Initialize the parallel executor.

        Args:
            nworkers: The number of workers to use. If None, then
                the number of workers is determined automatically.
        """

    @abstractmethod
    def submit(
        self,
        func,
        *args,
        **kwargs,
    ) -> object:
        """Schedules the callable, fn, to be executed as fn(*args, **kwargs).

        returns a Future object representing the execution of the callable.
        """

    @abstractmethod
    def get(self, future: object) -> Any:
        """Returns the result of the remote execution."""

    @abstractmethod
    def cancel(self, future: object) -> None:
        """Cancel the execution of the remote task."""

    @abstractmethod
    def wait(
        self,
        futures: Sequence[object],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> tuple[Any, Sequence[object]]:
        """Wait for the completion of the remote tasks.

        Returns: A tuple of result and remaining futures.
            First element is the result of the first completed task.
            Second element is the remaining futures.
        """

    @abstractmethod
    def map(self, fn, *iterables, timeout=None, chunksize=1) -> list[Any]:
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a child process. This argument is only
                used by ProcessPoolExecutor; it is ignored by
                ThreadPoolExecutor.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """


class MultiprocessingExecutor(ParallelExecutorABC):
    @override
    def __init__(
        self,
        *,
        nworkers: int | None = None,
        **kwargs,
    ) -> None:
        self.__executor = cf.ProcessPoolExecutor(
            max_workers=nworkers,
            **kwargs,
        )

    @override
    def submit(self, func, *args, **kwargs) -> cf.Future:
        return self.__executor.submit(func, *args, **kwargs)

    @override
    def get(self, future: cf.Future) -> Any:  # type: ignore[override]
        return future.result()

    @override
    def cancel(self, future: cf.Future) -> None:  # type: ignore[override]
        future.cancel()

    @override
    def wait(  # type: ignore[override]
        self,
        futures: Sequence[cf.Future],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> tuple[Any, Sequence[cf.Future]]:
        done, undone = cf.wait(
            futures,
            return_when=cf.FIRST_COMPLETED,
            timeout=timeout,
        )
        self.__executor.map
        return self.get(list(done)[0]), undone  # type: ignore[override]

    @override
    def map(self, fn, *iterables, timeout=None, chunksize=1) -> list[Any]:
        return list(
            self.__executor.map(
                fn,
                *iterables,
                timeout=timeout,
                chunksize=chunksize,
            )
        )


class SerialExecutor(MultiprocessingExecutor):
    @override
    def __init__(self, *, nworkers=None, **kwargs) -> None:
        super().__init__(nworkers=1, **kwargs)
