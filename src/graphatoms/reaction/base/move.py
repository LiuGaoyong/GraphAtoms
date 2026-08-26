from abc import ABC, abstractmethod

import numpy as np

from graphatoms.system import System


class MoveABC(ABC):
    """The base class for all moves."""

    @abstractmethod
    def apply(
        self,
        atoms: System,
        *,
        match: np.ndarray | None = None,
        **kwargs,
    ) -> System:
        """Apply this move to the given atoms."""
