"""The database class based on `dict`."""

from collections.abc import Iterator, Mapping, MutableSet
from typing import Any, override

import numpy as np
from ase import Atoms

from graphatoms.system import SysGraph


class DictDB(Mapping[str, Atoms], MutableSet[str]):
    def __init__(self) -> None:
        self.__data: dict[str, Atoms] = {}

    @override
    def __len__(self) -> int:
        return len(self.__data)

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            if isinstance(key, SysGraph):
                key = str(key.hash)
            else:
                raise TypeError(
                    "The key must be a string or a SysGraph object."
                )
        return str(key) in self.__data

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data.keys())

    def __getitem__(self, key: str) -> Atoms:
        return self.__data.__getitem__(key)

    @property
    def allthing(self) -> Mapping[str, Atoms]:
        return self.__data

    def discard(self, *args, **kwargs) -> None:
        raise RuntimeError("The discard method is not supported.")

    def add(self, value: SysGraph, event_cfg: Mapping[str, Any] = {}) -> bool:  # type: ignore
        """Return True if the value is new, False otherwise."""
        assert isinstance(value, SysGraph), "The value must be a SysGraph."
        assert value.energy is not None, "The energy of the value is None."
        assert value.fmax is not None, "The fmax of the value is None."
        assert value.hash is not None, "The hash of the value is None."
        assert value.frequencies is not None, (
            "The frequencies of the value is None."
        )  # noqa: E501
        assert value.check_minima(
            fmax=event_cfg.get("max_force", 0.05),
            fqmin=event_cfg.get("min_frequency", 30.0),
        ) or value.check_ts(
            fmax=event_cfg.get("max_force", 0.1),
            fqmin=event_cfg.get("min_frequency", 20.0),
        ), "The value is not `Minima` or `TS`."

        if not self.__contains__(value.hash):
            self.__data[value.hash] = value.to_ase(
                exclude_bond_attibutes=True,
            )
            return True
        else:
            return False


if __name__ == "__main__":
    import numpy as np
    from ase.collections import g2

    db = DictDB()
    for k in g2.names[:5]:
        v = SysGraph.from_ase(
            g2[k],
            energy=2.5,
            fmax=0.05,
            frequencies=np.array([1.0, 2.0, 3.0]) + 50,
        )
        print(v)
        print(v.hash, k)
        db.add(v)

    print(db.keys())
    for k in db:
        atoms = db[k]
        print(atoms.info)
        g = SysGraph.from_ase(atoms)
        print(g, g.hash)
        print()
