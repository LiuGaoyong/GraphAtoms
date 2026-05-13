from ase import Atoms
from typing_extensions import Self

from ._base import RTGP


class CreateMiaxin(RTGP):
    @classmethod
    def from_ase_trajectory(cls, traj: list[Atoms]) -> Self:
        raise NotImplementedError
        pass

    @classmethod
    def from_try(cls) -> Self | None: ...
