from abc import abstractmethod
from collections.abc import Mapping, MutableSet
from typing import override

import numpy as np
from ase import Atoms

from graphatoms.system import SysGraph


class DatabaseABC(Mapping[str, Atoms], MutableSet[str]):
    @abstractmethod
    @override
    def __init__(self, *, append: bool = True) -> None:
        """Initialize the database."""

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            if isinstance(key, SysGraph):
                key = self.__get_key_of(key)
            else:
                raise TypeError(
                    "The key must be a string or a SysGraph object."
                )
        return self._contains(key)

    @abstractmethod
    def _contains(self, key: str) -> bool: ...
    @property
    @abstractmethod
    def allthing(self) -> Mapping[str, Atoms]: ...
    @abstractmethod
    def _save(self, key: str, value: SysGraph) -> None: ...

    @override
    def add(  # type: ignore
        self,
        value: SysGraph,
        check_positions: bool = True,
        check_threshold: float = 0.05,  # Angstrom
    ) -> bool:
        """Add a value to the database.

        If the value is already in the database, return False.
        Otherwise, return True.
        """
        key = self.__get_key_of(value)
        if not self.__contains__(key):
            self._save(key, value)
            return True
        else:
            if check_positions:
                old_value = self[key]
                vdiff = value.positions - old_value.positions
                if not np.all(np.abs(vdiff) < check_threshold):
                    # Save some files for debug.
                    old = SysGraph.from_ase(old_value)
                    for v, append in [
                        (old_value, False),
                        (value.to_ase(), True),
                    ]:
                        v.write(f"{key}-debug.xyz", "extxyz", append=append)
                    value.write_npz(f"{key}-new.npz")  # type: ignore
                    old.write_npz(f"{key}-old.npz")  # type: ignore
                    raise ValueError(
                        "The positions are not the same as the old value. "
                        f"Check threshold: {check_threshold} Angstrom. But "
                        f"Positions difference Max: {np.max(np.abs(vdiff))}."
                    )
            return False

    @override
    def discard(self, *args, **kwargs) -> None:
        raise RuntimeError("The discard method is not supported.")

    @staticmethod
    def __get_key_of(value: SysGraph, use_positions_uuid: bool = True) -> str:
        return value.get_key_for_metadata(use_positions_uuid)
