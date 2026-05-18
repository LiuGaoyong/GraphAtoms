from functools import cached_property
from typing import Annotated

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from GraphAtoms.dataclasses import NDArray, OurFrozenModel, numpy_validator


class BondsAttr(OurFrozenModel):
    pair: Annotated[NDArray, numpy_validator("int32", (-1, 2))] | None = None
    distance: Annotated[NDArray, numpy_validator("float32")] | None = None
    order: Annotated[NDArray, numpy_validator("float16")] | None = None

    @model_validator(mode="after")
    def __check_bonds(self) -> Self:
        if self.nbonds != 0:
            assert isinstance(self.pair, np.ndarray), (
                "The `pair` should be a numpy array if nbonds > 0."
            )
            assert self.pair.ndim == 2 and self.pair.shape[1] == 2, (
                "The `pair` should be a 2D array with shape (nbonds, 2)."
                f"But we got {self.pair.shape} instead."
            )
            if np.any(self.pair[:, 0] == self.pair[:, 1]):
                raise ValueError(
                    "The `pair` should not contain self-loop bonds. It "
                    "is typically caused by the structure is periodic "
                    "and contains too less atoms (bulk? surface? etc). "
                )
            for k in BondsAttr.__pydantic_fields__.keys():
                v = getattr(self, k, None)
                if v is not None:
                    assert v.shape[0] == self.nbonds, (
                        f"Invalid shape for `{k}`: Len({k})="
                        f"{len(v)} but nbonds={self.nbonds}."
                    )
        return self

    @cached_property
    def nbonds(self) -> int:
        if self.pair is None:
            return 0
        return self.pair.shape[0]

    @cached_property
    def source(self) -> NDArray:
        return self.P[:, 0]

    @cached_property
    def target(self) -> NDArray:
        return self.P[:, 1]

    @cached_property
    def P(self) -> NDArray:
        if self.pair is None:
            return np.array([[]], dtype="int32").reshape(-1, 2)
        return self.pair

    @cached_property
    def D(self) -> NDArray:
        if self.distance is None:
            return np.array([[]], dtype="float32").reshape(-1)
        return self.distance
