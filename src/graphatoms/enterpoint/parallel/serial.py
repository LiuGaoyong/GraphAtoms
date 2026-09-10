"""Serial execution backend.

A serial executor is a special case of the multiprocessing backend with
a single worker. This keeps the async semantics (submit returns
immediately, result blocks) so that cancel/wait/as_completed work
uniformly across all backends.
"""

from __future__ import annotations

from graphatoms.enterpoint.parallel.multiprocessing import ProcessPoolExecutor


class SerialExecutor(ProcessPoolExecutor):
    """Serial executor: a ProcessPoolExecutor with a single worker."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(max_workers=1, **kwargs)
