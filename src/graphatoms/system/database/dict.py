"""The database class based on `dict`."""

from collections.abc import Iterator, Mapping
from typing import override

from ase import Atoms

from graphatoms.system import SysGraph
from graphatoms.system.database.abc import DatabaseABC


class DictDB(DatabaseABC):
    """The database class based on `dict`."""

    @override
    def __init__(self, *, append: bool = True) -> None:
        self.__data: dict[str, Atoms] = {}

    @override
    def __len__(self) -> int:
        return len(self.__data)

    @override
    def _contains(self, key: str) -> bool:
        return key in self.__data

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.__data.keys())

    @override
    def __getitem__(self, key: str) -> Atoms:
        return self.__data.__getitem__(key)

    @property
    @override
    def allthing(self) -> Mapping[str, Atoms]:
        return self.__data

    @override
    def _save(self, key: str, value: SysGraph) -> None:
        self.__data[key] = value.to_ase()
