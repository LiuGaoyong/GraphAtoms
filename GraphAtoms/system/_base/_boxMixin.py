from functools import cached_property
from typing import Annotated

import numpy as np
from ase.cell import Cell
from ase.geometry import cell as cellutils
from pydantic import model_validator
from pymatgen.core.lattice import Lattice
from typing_extensions import Self

from GraphAtoms.dataclasses import (
    NDArray,
    OurFrozenModel,
    numpy_validator,
)


class BoxMixin(OurFrozenModel):
    box: Annotated[NDArray, numpy_validator(float, (3, 3))] | None = None

    @model_validator(mode="after")
    def __check_box(self) -> Self:
        if not self.is_periodic:
            object.__setattr__(self, "box", None)
        return self

    @cached_property
    def is_periodic(self) -> bool:
        if self.box is None:
            return False
        v = np.array([self.a, self.b, self.c])
        return bool(np.any(np.abs(v) > 1e-7))

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
        return Cell.new(self.box)

    @cached_property
    def pmg_lattice(self) -> Lattice:
        return Lattice(self.ase_cell.array)

    @cached_property
    def is_orthorhombic(self) -> bool:
        if self.box is None:
            return True
        return cellutils.is_orthorhombic(self.box)
