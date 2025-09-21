from collections.abc import Sized
from functools import cached_property
from typing import Annotated, override

import numpy as np
import pydantic
from ase.symbols import Symbols
from pandas import DataFrame
from typing_extensions import Self

from GraphAtoms.common import NpzPklBaseModel, XxxKeyMixin
from GraphAtoms.utils.ndarray import NDArray, numpy_validator


class __AtomKey(XxxKeyMixin):
    NUMBER = "numbers"
    POSITION = "positions"
    COORDINATION = "coordination"
    MOVE_FIX_TAG = "move_fix_tag"
    IS_OUTER = "is_outer"


ATOM_KEY = __AtomKey()
__all__ = ["ATOM_KEY", "AtomsMixin"]


class AtomsMixin(NpzPklBaseModel, Sized):
    numbers: Annotated[NDArray, numpy_validator("uint8")]
    positions: Annotated[NDArray, numpy_validator(float, (-1, 3))]
    is_outer: Annotated[NDArray, numpy_validator(bool)] | None = None
    move_fix_tag: Annotated[NDArray, numpy_validator("int8")] | None = None
    coordination: Annotated[NDArray, numpy_validator("uint8")] | None = None

    @pydantic.model_validator(mode="after")
    def __check_keys_and_shape(self) -> Self:
        assert self.numbers.shape == (self.natoms,), self.numbers.shape
        assert self.positions.shape == (self.natoms, 3), self.positions.shape
        for k in ("is_outer", "coordination", "move_fix_tag"):
            v = getattr(self, k, None)
            if v is not None:
                assert v.shape == (self.natoms,), f"Invalid shape for `{k}`."
                if k == "move_fix_tag":
                    assert self.isfix.sum() != 0, "`isfix` sum == 0"
                    assert self.iscore.sum() != 0, "`iscore` sum == 0"
                    assert self.isfix.sum != len(self), "`ismoved` sum == 0"
        return self

    @override
    def _string(self) -> str:
        result = self.symbols.get_chemical_formula("metal")
        return f"{result},{'SUB' if self.is_sub else 'TOT'}"

    @override
    def __len__(self) -> int:
        return self.numbers.shape[0]

    @override
    def __hash__(self) -> int:
        return super().__hash__()

    @property
    def natoms(self) -> int:
        return self.numbers.shape[0]

    @property
    def nfix(self) -> int:
        return int(self.isfix.sum())

    @property
    def isfix(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("Cannot get `isfix` for `move_fix_tag` is None.")
            return np.empty_like(self.numbers, dtype=bool)
        return self.move_fix_tag < 0  # type: ignore

    @property
    def ncore(self) -> int:
        return int(self.iscore.sum())

    @property
    def iscore(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("Cannot get `isfix` for `move_fix_tag` is None.")
            return np.empty_like(self.numbers, dtype=bool)
        return self.move_fix_tag == 0

    @property
    def nmoved(self) -> int:
        return self.natoms - self.nfix - self.ncore

    @property
    def isfirstmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("Cannot get `isfix` for `move_fix_tag` is None.")
            return np.empty_like(self.numbers, dtype=bool)
        return self.move_fix_tag == 1

    @property
    def islastmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("Cannot get `isfix` for `move_fix_tag` is None.")
            return np.empty_like(self.numbers, dtype=bool)
        return self.move_fix_tag == np.max(self.move_fix_tag)

    @classmethod
    def DF_ATOMS_PARSER(cls, df: DataFrame) -> dict[str, np.ndarray]:
        assert len(df.columns) >= 4, df.columns
        assert ATOM_KEY.NUMBER in df.columns, df.columns
        R_KEYS = [f"{ATOM_KEY.POSITION}_{k}" for k in "xyz"]
        assert all(k in df.columns for k in R_KEYS), df.columns
        dct = {ATOM_KEY.NUMBER: df[ATOM_KEY.NUMBER].to_numpy()}
        dct[ATOM_KEY.POSITION] = df[R_KEYS].to_numpy()  # type: ignore
        for k in set(df.columns[4:]) & set(ATOM_KEY._DICT.values()):
            dct[k] = df[k].to_numpy()
        return dct

    @property
    def DF_ATOMS(self) -> DataFrame:
        df = DataFrame({ATOM_KEY.NUMBER: self.numbers})
        for i, k in enumerate("xyz"):
            k = f"{ATOM_KEY.POSITION}_{k}"
            df[k] = self.positions[:, i]
        for k in set(ATOM_KEY._DICT.values()):
            if k not in (ATOM_KEY.POSITION, ATOM_KEY.NUMBER):
                v = getattr(self, k, None)
                if v is not None:
                    df[k] = v
        return df

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.positions, dtype=float)

    @property
    def Z(self) -> np.ndarray:
        return np.asarray(self.numbers, dtype=int)

    @property
    def COLOR(self) -> list[str] | np.ndarray:
        return self.__COLOR

    @cached_property
    def __COLOR(self) -> list[str] | np.ndarray:
        cn = np.char.mod("%d-", self.CN)
        z = np.char.mod("%d-", self.Z)
        return np.char.add(z, cn)

    @property
    def CN(self) -> np.ndarray:
        if self.coordination is not None:
            return self.coordination
        else:
            raise NotImplementedError(
                "Use matrix-related method to solve this issue."
            )

    @property
    def is_sub(self) -> bool:
        return self.coordination is not None

    @property
    def is_nonmetal(self) -> bool:
        gas_elem = [2, 10, 18, 36, 54, 86]
        gas_elem += [1, 6, 7, 8, 9, 15, 16, 17]
        return bool(np.all(np.isin(self.numbers, gas_elem)))

    @property
    def symbols(self) -> Symbols:
        return Symbols(numbers=self.numbers)
