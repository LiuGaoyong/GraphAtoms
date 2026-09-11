"""The database class based on ASE."""

# ruff: noqa: D101 D107 D105
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import override

from ase import Atoms
from ase.db.core import connect, now
from ase.db.row import AtomsRow
from ase.db.sqlite import SQLite3Database

from graphatoms.system import SysGraph

from .abc import DatabaseABC


class AseSqliteDB(DatabaseABC):
    @override
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
    def _contains(self, key: str) -> bool:
        return key in self.__keys

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.__keys)

    @override
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
    @override
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

    @override
    def _save(self, key: str, value: SysGraph) -> None:
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
