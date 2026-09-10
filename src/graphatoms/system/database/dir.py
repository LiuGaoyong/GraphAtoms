"""The database class based on `dict`."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import override

from ase import Atoms

from graphatoms.system import SysGraph
from graphatoms.system.database.abc import DatabaseABC


class DirDB(DatabaseABC):
    """The database class based on `dict`."""

    @override
    def __init__(self, path: Path, *, append: bool = True) -> None:
        self.__path = path
        if not append:
            assert (
                not self.__path.exists()
                or len(list(self.__path.glob("*"))) == 0
            ), f"The directory {self.__path} already exists and is not empty."
        self.__path.mkdir(parents=True, exist_ok=True)

    @override
    def __len__(self) -> int:
        return len(list(self.__path.glob("*.npz")))

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            if isinstance(key, SysGraph):
                key = self.get_key_of(key)
            else:
                raise TypeError(
                    "The key must be a string or a SysGraph object."
                )
        return (self.__path / f"{key}.npz").exists()

    @override
    def __iter__(self) -> Iterator[str]:
        for p in self.__path.glob("*.npz"):
            yield p.stem

    @override
    def __getitem__(self, key: str) -> Atoms:
        p = self.__path.joinpath(f"{key}.npz")
        return SysGraph.read_npz(p).to_ase()

    @property
    @override
    def allthing(self) -> Mapping[str, Atoms]:
        return {key: self[key] for key in self.__iter__()}

    @override
    def _save(self, key: str, value: SysGraph) -> None:
        p = self.__path.joinpath(f"{key}.npz")
        value.write_npz(p)
