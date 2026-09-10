"""The database class based on HDF5 file."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, override

import h5py
from ase import Atoms

from graphatoms.system import SysGraph

from .abc import DatabaseABC


class AseH5DB(DatabaseABC):
    """The database class based on HDF5 file."""

    @override
    def __init__(self, path: Path, append: bool = True) -> None:
        assert path.name.endswith(".h5"), "The filename must end with .h5"
        if append:
            assert path.exists(), "The database file does not exist."
        else:
            f = h5py.File(path, "w", libver="latest")
            f.swmr_mode = True
            f.close()
        self.__path = path

    @override
    def __len__(self) -> int:
        with h5py.File(
            self.__path,
            "r",
            libver="latest",
            swmr=True,
            locking=False,
        ) as f:
            return len(f.keys())

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            if isinstance(key, SysGraph):
                key = str(key.hash)
            else:
                raise TypeError(
                    "The key must be a string or a SysGraph object."
                )
        with h5py.File(
            self.__path,
            "r",
            libver="latest",
            swmr=True,
            locking=False,
        ) as f:
            return str(key) in f.keys()

    @override
    def __iter__(self) -> Iterator[str]:
        with h5py.File(
            self.__path,
            "r",
            libver="latest",
            swmr=True,
            locking=False,
        ) as f:
            data = [str(i) for i in f.keys()]
        return iter(data)

    @override
    def __getitem__(self, key: str) -> Atoms:
        with h5py.File(
            self.__path,
            "r",
            libver="latest",
            swmr=True,
            locking=False,
        ) as f:
            value: h5py.Group = f[key]  # type: ignore
            assert isinstance(value, h5py.Group), (
                "The value must be a h5py.Group."
            )
            dct = {k: value[k][()] for k in value.keys()}  # type: ignore
        return SysGraph.from_dict(dct).to_ase()

    @property
    @override
    def allthing(self) -> Mapping[str, Atoms]:
        with h5py.File(
            self.__path,
            "r",
            libver="latest",
            swmr=True,
            locking=False,
        ) as f:
            data: dict[str, Any] = {}
            for key in f.keys():
                value: h5py.Group = f[key]  # type: ignore
                assert isinstance(value, h5py.Group), (
                    "The value must be a h5py.Group."
                )
                dct = {k: value[k][()] for k in value.keys()}  # type: ignore
                data[key] = SysGraph.from_dict(dct).to_ase()
        return data

    @override
    def _save(self, key: str, value: SysGraph) -> None:
        f = h5py.File(
            self.__path,
            "a",
            libver="latest",
            locking=False,
        )
        f.swmr_mode = True
        try:
            group = f.create_group(value.hash)
            for k, v in value.to_dict().items():
                group.create_dataset(k, data=v)
            f.flush()
        finally:
            f.close()
