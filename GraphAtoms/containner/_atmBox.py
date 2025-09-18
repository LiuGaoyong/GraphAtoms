from functools import cached_property
from typing import Annotated

import numpy as np
import pydantic
from ase import Atoms
from ase.cell import Cell
from ase.geometry import cell as cellutils
from typing_extensions import Self, override

from GraphAtoms.common import BaseModel, XxxKeyMixin
from GraphAtoms.utils.ndarray import NDArray, Shape


class __BoxKey(XxxKeyMixin):
    A = "a"
    B = "b"
    C = "c"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


BOX_KEY = __BoxKey()
__all__ = ["BOX_KEY", "Box"]


class Box(BaseModel):
    a: pydantic.NonNegativeFloat = 0.0
    b: pydantic.NonNegativeFloat = 0.0
    c: pydantic.NonNegativeFloat = 0.0
    alpha: Annotated[float, pydantic.Field(ge=0, le=180)] = 90.0
    beta: Annotated[float, pydantic.Field(ge=0, le=180)] = 90.0
    gamma: Annotated[float, pydantic.Field(ge=0, le=180)] = 90.0

    @pydantic.model_validator(mode="after")
    def __check_keys(self) -> Self:
        fields = self.__pydantic_fields__.keys()
        assert set(fields) >= set(BOX_KEY._DICT.values()), (
            "Invalid fields or ENERGETICS_KEY."
        )
        assert set(fields) >= set(self._convert().keys()), (
            "Invalid _convert dictionary."
        )
        return self

    @override
    def _string(self) -> str:
        return "PBC" if self.is_periodic else "NOPBC"

    @cached_property
    def CELL(self) -> NDArray[Shape["3,3"], float]:  # type: ignore
        """The numpy.ndarray of `ase.Cell` after minkowski-reduce."""
        par = [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]
        return Cell.new(par).minkowski_reduce()[0].array  # type: ignore

    @pydantic.computed_field
    @cached_property
    def is_periodic(self) -> bool:
        v = np.array([self.a, self.b, self.c])
        return bool(np.all(np.abs(v) < 1e-7))

    @pydantic.computed_field
    @cached_property
    def is_orthorhombic(self) -> bool:
        return cellutils.is_orthorhombic(self.CELL)

    @classmethod
    def from_ase(cls, ase_object: Cell | Atoms) -> Self:
        if not isinstance(ase_object, Cell):
            ase_object = ase_object.cell
        a, b, c, alpha, beta, gamma = ase_object.cellpar()
        return cls(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
