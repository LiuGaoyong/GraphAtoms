"""The database class based on ASE."""

from collections.abc import Iterator, Mapping, MutableSet
from pathlib import Path
from typing import Any, override

from ase import Atoms

from graphatoms.system import SysGraph


class AseH5DB(Mapping[str, Atoms], MutableSet[str]):
    def __init__(self, path: Path, append: bool = True) -> None:
        raise NotImplementedError

    @override
    def __len__(self) -> int:
        raise NotImplementedError

    @override
    def __contains__(self, key: object) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Atoms:
        raise NotImplementedError

    @property
    def allthing(self) -> Mapping[str, Atoms]:
        raise NotImplementedError

    def discard(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def add(self, value: SysGraph, event_cfg: Mapping[str, Any] = {}) -> bool:  # type: ignore
        raise NotImplementedError
