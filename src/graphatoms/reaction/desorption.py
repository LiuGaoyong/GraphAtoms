from typing import Self, override

import numpy as np
from ase import Atoms
from pydantic import model_validator

from graphatoms.reaction._event import EventBase
from graphatoms.system.system import System


class Desorption(EventBase):
    @model_validator(mode="after")
    def __check_something(self) -> Self:
        assert self.G is not None, "The gas must be not None."
        assert self.T is None, "The transition state must be None."
        assert len(self.R) == len(self.P) + len(self.G), (
            "The reactant state must be the sum of the "  #
            "product state and the gas molecule."
        )
        return self

    @override
    def apply(
        self,
        atoms: System | Atoms,
        *args,
        matched_indxs: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple[Atoms, float]:
        atoms, rmsd = super().apply(atoms, *args, matched_indxs, **kwargs)
        del atoms[np.arange(len(self.R)) >= len(self.P)]
        return atoms, rmsd
