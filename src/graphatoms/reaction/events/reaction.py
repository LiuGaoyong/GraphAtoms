from typing import Self

from pydantic import model_validator

from graphatoms.reaction.base.event import EventBase


class _Reaction(EventBase):
    @model_validator(mode="after")
    def __check_something(self) -> Self:
        n = int(max(len(self.R), len(self.P)))
        assert self.T is not None, "The transition state must be not None."
        assert n == len(self.T), (
            "The number of atoms must be equal. But got "  #
            f"T={len(self.T)}, R={len(self.R)} "  #
            f"and P={len(self.P)}."
        )
        return self


class ReactionLH(_Reaction):
    """The reaction by Langmuir-Hinshelwood mechanism."""

    @model_validator(mode="after")
    def __check_something(self) -> Self:

        assert self.G is None, "The gas must be None."
        assert len(self.R) == len(self.P), (
            "The number of atoms must be equal. But got "
            f"R={len(self.R)} and P={len(self.P)}."
        )
        return self


class ReactionER(_Reaction):
    """The reaction by Eley-Rideal mechanism."""

    @model_validator(mode="after")
    def __check_something(self) -> Self:
        raise NotImplementedError()
        assert self.G is not None, "The gas must be not None."
        assert (
            len(self.P) == len(self.R) + len(self.G)  #
            or len(self.R) == len(self.P) + len(self.G)
        )
        return self
