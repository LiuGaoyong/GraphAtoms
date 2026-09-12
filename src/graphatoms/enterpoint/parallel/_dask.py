"""Dask-backed execution backend.

Dask is an *optional* dependency. If it cannot be imported the module still
loads, but instantiating :class:`DaskExecutor` raises a clear
:class:`ImportError`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from graphatoms.enterpoint.parallel._base import ParallelExecutorABC

try:
    from distributed import Client  # type: ignore[import]

    _DASK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when dask absent
    Client = None  # type: ignore[assignment]
    _DASK_AVAILABLE = False


class DaskExecutor(ParallelExecutorABC):
    """An executor backed by :class:`dask.distributed.Client`."""

    raise NotImplementedError("DaskExecutor is not implemented.")

    def __init__(self, *, nworkers: int | None = None, **kwargs) -> None:
        if not _DASK_AVAILABLE:
            raise ImportError(
                "dask is not installed. Install it with: "
                "pip install dask distributed"
            )
        self._client = Client(
            n_workers=nworkers,
            silence_logs=logging.ERROR,
            **kwargs,
        )  # type: ignore

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> object:
        raise NotImplementedError("DaskExecutor.submit is not implemented.")
