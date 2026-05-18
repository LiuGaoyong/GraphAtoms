import json
from abc import abstractmethod
from collections.abc import Mapping
from functools import cached_property
from typing import Annotated, Any, override

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from pydantic import model_validator
from typing_extensions import Self

from GraphAtoms.geometry import bond_list
from GraphAtoms.utils.rdutils import (
    RDMol,  # type: ignore
    get_atomic_sasa,
    get_rdmol,
)

from ...dataclasses import NDArray, OurBaseModel, numpy_validator
from .._bonds._bondsAttr import BondsAttr
from .._sys._gasMixin import GasMixin
from ._box import BoxMixin
from ._eng import EnergeticsMixin

__all__ = ["Base"]


class MoveFixTag(OurBaseModel):
    move_fix_tag: Annotated[NDArray, numpy_validator("int8")] | None = None

    @model_validator(mode="after")
    def __check_atoms(self) -> Self:
        if self.move_fix_tag is not None:
            assert self.isfix.sum() != 0, "`isfix` sum == 0"
            assert self.iscore.sum() != 0, "`iscore` sum == 0"
            assert self.isfix.sum != self.natoms, "`ismoved` sum == 0"
        return self

    @cached_property
    @abstractmethod
    def natoms(self) -> int: ...

    @property
    def nfix(self) -> int:
        return int(self.isfix.sum())

    @property
    def isfix(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag < 0  # type: ignore

    @property
    def ncore(self) -> int:
        return int(self.iscore.sum())

    @property
    def iscore(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == 0

    @property
    def nmoved(self) -> int:
        return self.natoms - self.nfix

    @property
    def isfirstmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == 1

    @property
    def islastmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == np.max(self.move_fix_tag)


class AtomsAttr(NumbersMixin, MoveFixTag):
    is_outer: Annotated[NDArray, numpy_validator(bool)] | None = None
    coordination: Annotated[NDArray, numpy_validator("uint8")] | None = None
    hashes: list[str] | None = None

    @model_validator(mode="after")
    def __check_atoms(self) -> Self:
        assert self.positions.shape == (self.natoms, 3), self.positions.shape
        for k in AtomsAttr.__pydantic_fields__.keys():
            v = getattr(self, k, None)
            if v is not None:
                assert len(v) == self.natoms, (
                    f"Invalid shape for `{k}`: Len({k})="
                    f"{len(v)} but natoms={self.natoms}."
                )
        return self

    @property
    @abstractmethod
    def CN(self) -> NDArray: ...


class Base(AtomsAttr, BondsAttr, BoxMixin, GasMixin, EnergeticsMixin):
    @override
    @classmethod
    def from_str(cls, data: str, **kw) -> Self:
        return cls.from_dict(json.loads(data), **kw)

    @override
    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        parse_bonds: Mapping[str, Any] | None = None,
        parse_bonds_distance: bool = False,
        parse_bonds_order: bool = False,
        **kwargs,
    ) -> Self:
        obj = super().from_dict(data, **kwargs)
        dct: dict[str, np.ndarray | float] = obj.to_dict()

        if parse_bonds is not None and len(parse_bonds) == 0:
            parse_bonds = None
        ndata: int = len(dct)

        # parse bonds pair index
        if obj.pair is None and parse_bonds is not None:
            atoms = Atoms(
                numbers=obj.numbers,
                positions=obj.positions,
                pbc=obj.is_periodic,
                cell=obj.ase_cell,
            )
            dct["pair"] = pair = bond_list(atoms, **parse_bonds)
            if np.any(pair[:, 0] == pair[:, 1]):
                raise RuntimeError(
                    "The `pair` should not contain self-loop bonds. It "
                    "is typically caused by the structure is periodic "
                    "and contains too less atoms (bulk? surface? etc). "
                )
        else:
            pair = obj.P
        assert isinstance(pair, np.ndarray)
        if pair.size == 0:
            pair = None

        if pair is not None:
            i, j = np.transpose(pair)
            # parse bonds distance
            if obj.distance is None and parse_bonds_distance:
                v, c = obj.positions[i] - obj.positions[j], obj.ase_cell
                _, dct["distance"] = find_mic(v, c, obj.is_periodic)
            # parse bonds order
            if obj.order is None and parse_bonds_order:
                raise NotImplementedError(
                    "Bond order parsing is not implemented yet."
                )

        if len(dct) == ndata:
            return obj
        else:
            return super().from_dict(dct, **kwargs)

    @override
    def _string(self) -> str:
        pbc = "PBC" if self.is_periodic else "NOPBC"
        lst: list[str] = [self.formula, pbc]
        lst.append(f"{self.nbonds}Bonds")
        if self.coordination is not None:
            lst.append("Sub")
        if self.energy is None:
            lst.append("NOSPE")
        else:
            if abs(self.energy) >= 1:
                lst.append(f"E={self.energy:.2f}eV")
            elif abs(self.energy * 1000) >= 1:
                lst.append(f"E={self.energy * 1000:.2f}meV")
            else:
                lst.append(f"E={self.energy * 1000:.3e}meV")
        if self.frequencies is None:
            lst.append("NOVIB")
        else:
            lst.append("VIB")
        if self.check_minima():
            lst.append("Minima")
        if self.check_ts():
            lst.append("TS")
        return ",".join(lst)

    def get_rdmol(self) -> RDMol:
        return get_rdmol(
            numbers=self.numbers,
            geometry=self.positions,
            source=self.source,
            target=self.target,
            order=self.order if self.order is not None else None,
            infer_order=False,
            charge=0,
        )

    def get_atomic_sasa(self) -> np.ndarray:
        return np.asarray(get_atomic_sasa(self.get_rdmol()))
