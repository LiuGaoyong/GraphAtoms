from typing import override

import numpy as np
from ase import Atoms as AseAtoms
from typing_extensions import Any, Self

from GraphAtoms.common.error import NotSupportNonOrthorhombicLattice
from GraphAtoms.containner._aMixin import ATOM_KEY, AtomsMixin
from GraphAtoms.containner._aOther import OTHER_KEY, OtherMixin


class AtomsWithBoxEng(AtomsMixin, OtherMixin):
    """The atomic container."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif not OtherMixin.__eq__(self, other):
            return False
        elif not AtomsMixin.__eq__(self, other):
            return False
        else:
            return True

    @override
    def __hash__(self) -> int:
        return super().__hash__()

    @override
    def _string(self) -> str:
        return f"{super()._string()},{OtherMixin._string(self)}"

    def to_ase(self) -> AseAtoms:
        return AseAtoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.ase_cell,
            pbc=self.is_periodic,
            info=self.model_dump(
                mode="python",
                exclude_none=True,
                exclude=(
                    {
                        ATOM_KEY.NUMBER,
                        ATOM_KEY.POSITION,
                        OTHER_KEY.BOX,
                    }
                ),
            ),
        )

    @classmethod
    def from_ase(cls, atoms: AseAtoms) -> Self:
        if not atoms.cell.orthorhombic:
            raise NotSupportNonOrthorhombicLattice()
        dct: dict[str, Any] = atoms.info
        dct[ATOM_KEY.NUMBER] = atoms.numbers
        dct[ATOM_KEY.POSITION] = atoms.positions
        if np.sum(atoms.cell.array.any(1) & atoms.pbc) > 0:
            cell = atoms.cell.complete().minkowski_reduce()[0]
            dct[OTHER_KEY.BOX] = cell.array
        return cls.model_validate(dct)
