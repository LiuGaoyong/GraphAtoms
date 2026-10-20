from abc import abstractmethod
from collections.abc import Mapping, MutableSet
from pathlib import Path
from typing import override

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_xyz as write_extxyz

from graphatoms.system import SysGraph
from graphatoms.utils.bytestool import atomic_write


class DatabaseABC(Mapping[str, Atoms], MutableSet[str]):
    @abstractmethod
    @override
    def __init__(
        self,
        *,
        append: bool = True,
        use_positions_uuid: bool = False,
    ) -> None:
        """Initialize the database."""
        self.use_positions_uuid = bool(use_positions_uuid)

    @override
    def __contains__(self, key: str | SysGraph) -> bool:  # type: ignore
        if not isinstance(key, str):
            if isinstance(key, SysGraph):
                key = key.get_key_for_metadata(self.use_positions_uuid)
            else:
                raise TypeError(
                    "The key must be a string or a SysGraph object."
                    f" But got {type(key)}."
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
        key = value.get_key_for_metadata(self.use_positions_uuid)
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
                    lst = [old_value, value.to_ase()]
                    atomic_write(
                        Path(f"{key}-debug.xyz"),
                        writer=lambda p: write_extxyz(p, lst),
                    )
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
