from abc import abstractmethod
from functools import cached_property
from typing import Annotated, override

import numpy as np
from ase.symbols import Symbols
from pydantic import model_validator
from typing_extensions import Self

from GraphAtoms.dataclasses import NDArray, OurFrozenModel, numpy_validator


class NumbersMixin(OurFrozenModel):
    numbers: Annotated[NDArray, numpy_validator(int)]

    @cached_property
    def formula(self) -> str:
        return self.symbols.get_chemical_formula("metal")

    @cached_property
    def symbols(self) -> Symbols:
        return Symbols(self.numbers)

    @cached_property
    def is_nonmetal(self) -> bool:
        #          He, Ne, Ar, Kr, Xe, Rn
        gas_elem = [2, 10, 18, 36, 54, 86]
        #            H, C, N, O, F, P,  S,  Cl, Br, I
        gas_elem += [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        return bool(all(np.isin(self.numbers, gas_elem)))


class MoveFixTag(OurFrozenModel):
    move_fix_tag: Annotated[NDArray, numpy_validator("int8")] | None = None

    @model_validator(mode="after")
    def __check_atoms(self) -> Self:
        if self.move_fix_tag is not None:
            assert self.isfix.sum() != 0, "`isfix` sum == 0"
            assert self.iscore.sum() != 0, "`iscore` sum == 0"
            assert self.isfix.sum != self.natoms, "`ismoved` sum == 0"
        return self

    @cached_property
    @abstractmethod
    def natoms(self) -> int: ...

    @property
    def nfix(self) -> int:
        return int(self.isfix.sum())

    @property
    def isfix(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag < 0  # type: ignore

    @property
    def ncore(self) -> int:
        return int(self.iscore.sum())

    @property
    def iscore(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == 0

    @property
    def nmoved(self) -> int:
        return self.natoms - self.nfix

    @property
    def isfirstmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == 1

    @property
    def islastmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == np.max(self.move_fix_tag)


class AtomsAttr(NumbersMixin, MoveFixTag):
    positions: Annotated[NDArray, numpy_validator(float, (-1, 3))]
    is_outer: Annotated[NDArray, numpy_validator(bool)] | None = None
    coordination: Annotated[NDArray, numpy_validator("uint8")] | None = None
    hashes: list[str] | None = None

    @model_validator(mode="after")
    def __check_atoms(self) -> Self:
        assert self.positions.shape == (self.natoms, 3), self.positions.shape
        for k in AtomsAttr.__pydantic_fields__.keys():
            v = getattr(self, k, None)
            if v is not None:
                assert len(v) == self.natoms, (
                    f"Invalid shape for `{k}`: Len({k})="
                    f"{len(v)} but natoms={self.natoms}."
                )
        return self

    def __len__(self) -> int:
        return self.numbers.shape[0]

    @cached_property
    @override
    def natoms(self) -> int:
        return len(self.numbers)

    @property
    def Z(self) -> NDArray:
        return self.numbers

    @property
    def R(self) -> NDArray:
        return self.positions

    @property
    @abstractmethod
    def CN(self) -> NDArray: ...
