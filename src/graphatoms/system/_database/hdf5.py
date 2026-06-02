"""The database class based on HDF5 file."""

from collections.abc import Iterator, Mapping, MutableSet
from pathlib import Path
from typing import Any, override

import h5py
from ase import Atoms

from graphatoms.system import SysGraph


class AseH5DB(Mapping[str, Atoms], MutableSet[str]):
    def __init__(self, path: Path, append: bool = True) -> None:
        assert path.name.endswith(".h5"), "The filename must end with .h5"
        if append:
            assert path.exists(), "The database file does not exist."
        else:
            h5py.File(path, "w", libver="latest")
        self.__path = path

    @override
    def __len__(self) -> int:
        with h5py.File(self.__path, "r", libver="latest", swmr=True) as f:
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
        with h5py.File(self.__path, "r", libver="latest", swmr=True) as f:
            return str(key) in f.keys()

    def __iter__(self) -> Iterator[str]:
        with h5py.File(self.__path, "r", libver="latest", swmr=True) as f:
            data = [str(i) for i in f.keys()]
        return iter(data)

    def __getitem__(self, key: str) -> Atoms:
        with h5py.File(self.__path, "r", libver="latest", swmr=True) as f:
            value: h5py.Group = f[key]  # type: ignore
            assert isinstance(value, h5py.Group), (
                "The value must be a h5py.Group."
            )
            dct = {k: value[k][()] for k in value.keys()}  # type: ignore
        return SysGraph.from_dict(dct).to_ase()

    @property
    def allthing(self) -> Mapping[str, Atoms]:
        with h5py.File(self.__path, "r", libver="latest", swmr=True) as f:
            data: dict[str, Any] = {}
            for key in f.keys():
                value: h5py.Group = f[key]  # type: ignore
                assert isinstance(value, h5py.Group), (
                    "The value must be a h5py.Group."
                )
                dct = {k: value[k][()] for k in value.keys()}  # type: ignore
                data[key] = SysGraph.from_dict(dct).to_ase()
        return data

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
            with h5py.File(self.__path, "a", libver="latest") as f:
                f.swmr_mode = True
                group = f.create_group(value.hash)
                for k, v in value.to_dict().items():
                    group.create_dataset(k, data=v)
                f.flush()
            return True
        else:
            return False


if __name__ == "__main__":
    import numpy as np
    from ase.collections import g2

    db = AseH5DB(Path("test.h5"), False)
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

    db = AseH5DB(Path("test.h5"), True)
    print(db.keys())
    for k in db:
        atoms = db[k]
        print(atoms.info)
        g = SysGraph.from_ase(atoms)
        print(g, g.hash)
        print()
