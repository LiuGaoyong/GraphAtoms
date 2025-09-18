from typing import override

import numpy as np
from ase import Atoms as AseAtoms
from typing_extensions import Any, Self

from GraphAtoms.common.error import NotSupportNonOrthorhombicLattice
from GraphAtoms.containner._aBox import BOX_KEY, Box
from GraphAtoms.containner._aMixin import ATOM_KEY
from GraphAtoms.containner._aMixin import Atoms as AtomsMixin
from GraphAtoms.containner._aSpeVib import Energetics


class AtomsWithBoxEng(AtomsMixin, Energetics, Box):
    """The atomic container."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif not Box.__eq__(self, other):
            return False
        elif not Energetics.__eq__(self, other):
            return False
        elif not AtomsMixin.__eq__(self, other):
            return False
        else:
            return True

    @override
    def __hash__(self) -> int:
        return AtomsMixin.__hash__(self)

    @override
    def _string(self) -> str:
        return ",".join(
            [
                AtomsMixin._string(self),
                Box._string(self),
            ]
        )

    def to_ase(self) -> AseAtoms:
        return AseAtoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.ase_cell,
            pbc=self.is_periodic,
            info=self.model_dump(
                mode="python",
                exclude_none=True,
                exclude_defaults=True,
                exclude=(
                    {ATOM_KEY.NUMBER, ATOM_KEY.POSITION}
                    | set(BOX_KEY._DICT.values())
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
            a, b, c, alpha, beta, gamma = cell.cellpar()
            dct[BOX_KEY.A] = a
            dct[BOX_KEY.B] = b
            dct[BOX_KEY.C] = c
            dct[BOX_KEY.ALPHA] = alpha
            dct[BOX_KEY.BETA] = beta
            dct[BOX_KEY.GAMMA] = gamma
        return cls.model_validate(dct)
