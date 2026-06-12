from functools import cached_property
from typing import Annotated, Self, override

import numpy as np
from ase.cell import Cell
from ase.geometry import cell as cellutils
from pydantic import model_validator
from pymatgen.core.lattice import Lattice

from graphatoms.dataclasses import NDArray, OurBaseModel, numpy_validator


class Box(OurBaseModel):
    cell: Annotated[NDArray, numpy_validator(float, (3, 3))] | None = None
    pbc: Annotated[NDArray, numpy_validator(bool, (3,))] | None = None

    @override
    def _string(self) -> str:
        return "PBC" if self.is_periodic else "NOPBC"

    @model_validator(mode="after")
    def __check_cell(self) -> Self:
        if not self.is_periodic:
            object.__setattr__(self, "cell", None)
            object.__setattr__(self, "pbc", None)
        return self

    @cached_property
    def is_periodic(self) -> bool:
        if self.pbc is None:
            return False
        elif self.cell is None:
            return False
        elif self.ase_cell.volume < 1e-3:
            return False
        else:
            pbc = self.cell.any(1) & self.pbc
            return bool(np.sum(pbc) > 0)

    @cached_property
    def cellpar(self) -> tuple[float, float, float, float, float, float]:
        return tuple(cellutils.cell_to_cellpar(self.ase_cell.array).tolist())

    @cached_property
    def a(self) -> float:
        return self.cellpar[0]

    @cached_property
    def b(self) -> float:
        return self.cellpar[1]

    @cached_property
    def c(self) -> float:
        return self.cellpar[2]

    @cached_property
    def alpha(self) -> float:
        return self.cellpar[3]

    @cached_property
    def beta(self) -> float:
        return self.cellpar[4]

    @cached_property
    def gamma(self) -> float:
        return self.cellpar[5]

    @cached_property
    def ase_cell(self) -> Cell:
        return Cell.new(self.cell)

    @cached_property
    def pmg_lattice(self) -> Lattice:
        return Lattice(self.ase_cell.array)

    @cached_property
    def is_orthorhombic(self) -> bool:
        if self.cell is None:
            return True
        return cellutils.is_orthorhombic(self.cell)


def test_Box() -> None:
    assert len(Box.__abstractmethods__) == 0, Box.__abstractmethods__
