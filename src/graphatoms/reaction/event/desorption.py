from typing import Self

from pydantic import model_validator

from ..base.rtgp import RTGP


class Desorption(RTGP):
    @model_validator(mode="after")
    def __check_something(self) -> Self:
        assert self.G is not None, "The gas must be not None."
        assert self.T is None, "The transition state must be None."
        assert len(self.R) == len(self.P) + len(self.G), (
            "The reactant state must be the sum of the "  #
            "product state and the gas molecule."
        )
        return self
