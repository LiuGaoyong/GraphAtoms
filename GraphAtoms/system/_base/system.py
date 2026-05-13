import json
from collections.abc import Mapping
from typing import Any, override

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from typing_extensions import Self

from GraphAtoms.geometry import bond_list
from GraphAtoms.utils.rdutils import (
    RDMol,  # type: ignore
    get_atomic_sasa,
    get_rdmol,
)

from ._atomsAttr import AtomsAttr
from ._bondsAttr import BondsAttr
from ._boxMixin import BoxMixin
from ._engMixin import EnergeticsMixin
from ._gasMixin import GasMixin

__all__ = ["Base"]


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
