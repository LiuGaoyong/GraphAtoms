from typing_extensions import Self

from ._base import RTGP


class IoMixin(RTGP):
    @classmethod
    def from_ase_trajectory(cls) -> Self:
        raise NotImplementedError
        pass
