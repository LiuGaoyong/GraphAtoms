from abc import ABC, abstractmethod
from typing import Any

from graphatoms.system import System


class MoveABC(ABC):
    """The base class for all moves."""

    @abstractmethod
    def apply(
        self,
        atoms: System,
        *args,
        **kwargs,
    ) -> Any:
        """Apply this move to the given atoms."""
