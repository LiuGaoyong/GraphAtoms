"""The database class based on ASE."""

import os
from collections.abc import Iterator, Mapping, MutableSet
from pathlib import Path
from typing import Any, override

import numpy as np
from ase import Atoms
from ase.db.core import connect, now
from ase.db.row import AtomsRow
from ase.db.sqlite import SQLite3Database

from GraphAtoms.system import SysGraph


class AseSqliteDB(Mapping[str, Atoms], MutableSet[str]):
    def __init__(self, path: Path, append: bool = True) -> None:
        assert path.name.endswith(".db"), "The filename must end with .db"
        if append:
            assert path.exists(), "The database file does not exist."
        self.__db = connect(path, type="db", append=append, serial=True)
        assert isinstance(self.__db, SQLite3Database), (
            "The database type is not SQLite3Database."
        )
        with SQLite3Database(path.as_posix()) as db:
            ks = [row.unique_id for row in db.select()]
            self.__keys: set[str] = set(ks)
        if self.__db.connection is not None:
            self.__db.connection.close()
        self.__path = path

    @override
    def __len__(self) -> int:
        return len(self.__keys)

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            if isinstance(key, SysGraph):
                key = str(key.hash)
            else:
                raise TypeError(
                    "The key must be a string or a SysGraph object."
                )
        return str(key) in self.__keys

    def __iter__(self) -> Iterator[str]:
        return iter(self.__keys)

    def __getitem__(self, key: str) -> Atoms:
        with SQLite3Database(self.__path.as_posix()) as db:
            out: AtomsRow = db.get(unique_id=key)
        atoms: Atoms = out.toatoms(False)
        atoms.info["fmax"] = out.key_value_pairs["fmax0"]
        atoms.info["hash"] = out.unique_id  # type: ignore
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.info["frequencies"] = out.data["frequencies"]
        atoms.calc = None
        return atoms

    @property
    def allthing(self) -> Mapping[str, Atoms]:
        with SQLite3Database(self.__path.as_posix()) as db:
            outs: list[AtomsRow] = list(db.select())
            result: dict[str, Atoms] = {}
            for out in outs:
                atoms: Atoms = out.toatoms(False)
                atoms.info["fmax"] = out.key_value_pairs["fmax0"]
                atoms.info["hash"] = k = out.unique_id  # type: ignore
                atoms.info["energy"] = atoms.get_potential_energy()
                atoms.info["frequencies"] = out.data["frequencies"]
                atoms.calc = None
                result[k] = atoms
        return result

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
            with SQLite3Database(self.__path.as_posix()) as db:
                row = AtomsRow(value.to_ase())
                row["user"] = os.getenv("USER")
                row["unique_id"] = value.hash
                row["energy"] = value.energy
                row["ctime"] = now()
                db.write(
                    row,
                    data={"frequencies": value.frequencies},
                    key_value_pairs={"fmax0": value.fmax},
                )
                assert db.connection is not None
                db.connection.commit()
            self.__keys.add(value.hash)
            return True
        else:
            return False


if __name__ == "__main__":
    import numpy as np
    from ase.collections import g2

    db = AseSqliteDB(Path("test.db"), False)
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

    db = AseSqliteDB(Path("test.db"), True)
    print(db.keys())
    for k in db:
        atoms = db[k]
        print(atoms.info)
        g = SysGraph.from_ase(atoms)
        print(g, g.hash)
        print()
