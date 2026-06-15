from abc import ABC, abstractmethod

from ase.atoms import Atoms


class MoveABC(ABC):
    @abstractmethod
    def __call__(self, atoms: Atoms, *args, **kwargs) -> Atoms: ...

    def apply(self, atoms: Atoms, *args, **kwargs) -> Atoms:
        """Apply this move to the given atoms."""
        return self.__call__(atoms, *args, **kwargs)
