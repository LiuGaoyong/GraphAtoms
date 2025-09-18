from typing import override

import numpy as np
from ase import Atoms as AseAtoms
from typing_extensions import Any, Self

from GraphAtoms.common.error import NotSupportNonOrthorhombicLattice
from GraphAtoms.containner._atmBox import Box
from GraphAtoms.containner._atmEng import Energetics
from GraphAtoms.containner._atmMix import ATOM_KEY
from GraphAtoms.containner._atmMix import Atoms as AtomsMixin


class AtomsWithBoxEng(AtomsMixin, Energetics):
    """The atomic container."""

    box: Box = Box()

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif not np.allclose(self.box.CELL, other.box.CELL):
            return False
        elif not Energetics.__eq__(self, other):
            return False
        elif not AtomsMixin.__eq__(self, other):
            return False
        else:
            return True

    @override
    def _string(self) -> str:
        return ",".join(
            [
                AtomsMixin._string(self),
                self.box._string(),
            ]
        )

    def to_ase(self) -> AseAtoms:
        return AseAtoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.box.CELL,
            pbc=self.box.is_periodic,
            info=self.model_dump(
                mode="python",
                exclude_none=True,
                exclude={ATOM_KEY.NUMBER, ATOM_KEY.POSITION, "box"},
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
            dct["box"] = Box.from_ase(cell)
        return cls.model_validate(dct)
