from abc import abstractmethod
from collections.abc import Mapping, MutableSet
from functools import reduce
from typing import override

import numpy as np
from ase import Atoms
from ase.symbols import Symbols

from graphatoms.system import SysGraph
from graphatoms.utils.bytestool import hash_string


class DatabaseABC(Mapping[str, Atoms], MutableSet[str]):
    @abstractmethod
    @override
    def __init__(self, *, append: bool = True) -> None:
        """Initialize the database."""

    @property
    @abstractmethod
    def allthing(self) -> Mapping[str, Atoms]: ...
    @abstractmethod
    def _save(self, key: str, value: SysGraph) -> None: ...

    @override
    def add(self, value: SysGraph) -> bool:  # type: ignore
        """Add a value to the database.

        If the value is already in the database, return False.
        Otherwise, return True.
        """
        key = self.get_key_of(value)
        if not self.__contains__(key):
            self._save(key, value)
            return True
        else:
            return False

    @override
    def discard(self, *args, **kwargs) -> None:
        raise RuntimeError("The discard method is not supported.")

    @staticmethod
    def get_key_of(value: SysGraph) -> str:
        """Get the key of the value."""
        symbols: Symbols = Symbols(value.numbers)
        fml: str = symbols.get_chemical_formula("metal")
        geometry: np.ndarray = value.positions
        x = np.char.rjust(np.char.mod("%.1f", geometry[:, 0]), 20)
        y = np.char.rjust(np.char.mod("%.1f", geometry[:, 1]), 20)
        z = np.char.rjust(np.char.mod("%.1f", geometry[:, 2]), 20)
        pos_str = "".join(reduce(np.char.add, [x, y, z, " \n"]))
        uuid = hash_string(pos_str, digest_size=8)
        return f"{value.hash}-{fml}-{uuid}"
