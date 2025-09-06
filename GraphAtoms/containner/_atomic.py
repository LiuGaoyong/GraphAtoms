from collections.abc import Sized
from functools import cached_property
from typing import Annotated, override

import numpy as np
import pydantic
from ase import Atoms
from ase.cell import Cell
from ase.symbols import Symbols
from numpy.typing import ArrayLike as NpyArrayLike
from pandas import DataFrame
from scipy.sparse import coo_matrix, csr_array, sparray, spmatrix
from scipy.sparse import triu as sp_triu
from typing_extensions import Any, Self

from GraphAtoms.common.abc import NpzPklBaseModel
from GraphAtoms.common.error import NotSupportNonOrthorhombicLattice
from GraphAtoms.common.geometry import get_adjacency_sparse_matrix
from GraphAtoms.common.ndarray import NDArray, Shape
from GraphAtoms.containner._bonded import BOND_KEY
from GraphAtoms.containner._mRDKit import get_adjacency_by_rdkit

ArrayLike = NpyArrayLike | sparray | spmatrix


class __GraphKey:
    PRESSURE = "pressure"
    FMAX = "fmax_nonconstraint"
    FMAXC = "fmax_constraint"
    FREQS = "frequencies"
    ENERGY = "energy"
    HASH = "hash"
    BOX = "box"

    @property
    def _DICT(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in dir(self) if k[0] != "_"}


class __AtomKey:
    NUMBER = "numbers"
    POSITION = "positions"
    COORDINATION = "coordination"
    MOVE_FIX_TAG = "move_fix_tag"
    IS_OUTER = "is_outer"

    @property
    def _DICT(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in dir(self) if k[0] != "_"}


TOTAL_KEY, ATOM_KEY = __GraphKey(), __AtomKey()


class GraphAttrMixin(NpzPklBaseModel):
    pressure: pydantic.PositiveFloat | None = None
    box: NDArray[Shape["6"], float] | None = None  # type: ignore
    frequencies: NDArray[Shape["*"], float] | None = None  # type: ignore
    fmax_nonconstraint: float | None = None
    fmax_constraint: float | None = None
    energy: float | None = None
    hash: str | None = None

    @classmethod
    @override
    def _convert(cls) -> dict[str, tuple[tuple, str]]:
        result: dict[str, tuple[tuple, str]] = super()._convert()
        result["frequencies"] = ((None,), "float64")
        result["box"] = ((6,), "float64")
        return result

    @pydantic.model_validator(mode="after")
    def __check_keys(self) -> Self:
        fields = self.__pydantic_fields__.keys()
        assert set(fields) >= set(TOTAL_KEY._DICT.values())
        return self

    @cached_property
    def CELL(self) -> Cell:
        return Cell.new(self.box)

    @property
    def PBC(self) -> bool:
        return self.box is not None

    @override
    def _string(self) -> str:
        return "PBC" if self.PBC else "NOPBC"


class AtomicAttrMixin(NpzPklBaseModel, Sized):
    numbers: NDArray[Shape["*"], np.uint8]  # type: ignore
    positions: NDArray[Shape["*,3"], float]  # type: ignore
    is_outer: NDArray[Shape["*"], bool] | None = None  # type: ignore
    move_fix_tag: NDArray[Shape["*"], np.int8] | None = None  # type: ignore
    coordination: NDArray[Shape["*"], np.uint8] | None = None  # type: ignore

    @classmethod
    @override
    def _convert(cls) -> dict[str, tuple[tuple, str]]:
        result: dict[str, tuple[tuple, str]] = super()._convert()
        result["positions"] = ((None, 3), "float64")
        result["coordination"] = ((None,), "uint8")
        result["move_fix_tag"] = ((None,), "int8")
        result["numbers"] = ((None,), "uint8")
        result["is_outer"] = ((None,), "bool")
        return result

    @pydantic.model_validator(mode="after")
    def __check_keys_and_shape(self) -> Self:
        assert set(self.__pydantic_fields__) >= set(ATOM_KEY._DICT.values())
        assert self.numbers.shape == (self.natoms,), self.numbers.shape
        assert self.positions.shape == (self.natoms, 3), self.positions.shape
        if self.is_outer is not None:
            assert self.is_outer.shape == (self.natoms,), "vw345y"
        if self.coordination is not None:
            assert self.coordination.shape == (self.natoms,), "ve45yr56"
        if self.move_fix_tag is not None:
            assert self.move_fix_tag.shape == (self.natoms,), (
                "Invalid shape for `move_fix_tag`."
            )
            assert self.isfix.sum() != 0, "`isfix` sum == 0"
            assert self.iscore.sum() != 0, "`iscore` sum == 0"
            assert self.isfix.sum != len(self), "`ismoved` sum == 0"
        return self

    @override
    def __len__(self) -> int:
        return self.numbers.shape[0]

    @property
    def natoms(self) -> int:
        return self.numbers.shape[0]

    @property
    def isfix(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("Cannot get `isfix` for `move_fix_tag` is None.")
            return np.empty_like(self.numbers, dtype=bool)
        return self.move_fix_tag < 0  # type: ignore

    @property
    def iscore(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("Cannot get `isfix` for `move_fix_tag` is None.")
            return np.empty_like(self.numbers, dtype=bool)
        return self.move_fix_tag == 0

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
        dct[ATOM_KEY.POSITION] = df[R_KEYS].to_numpy()
        for k in set(df.columns[4:]) & set(ATOM_KEY._DICT.values()):
            dct[k] = df[k].to_numpy()
        return dct

    @property
    def DF_ATOMS(self) -> DataFrame:
        df = DataFrame({ATOM_KEY.NUMBER: self.numbers})
        for i, k in enumerate("xyz"):
            k = f"{ATOM_KEY.POSITION}_{k}"
            df[k] = self.positions[:, i]
        if self.is_outer is not None:
            df[ATOM_KEY.IS_OUTER] = self.is_outer
        if self.coordination is not None:
            df[ATOM_KEY.COORDINATION] = self.coordination
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

    @override
    def _string(self) -> str:
        result = self.symbols.get_chemical_formula("metal")
        return f"{result},{'SUB' if self.is_sub else 'TOT'}"


class AtomicContainner(AtomicAttrMixin, GraphAttrMixin):
    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif not GraphAttrMixin.__eq__(self, other):
            return False
        elif not AtomicAttrMixin.__eq__(self, other):
            return False
        else:
            return True

    @override
    def _string(self) -> str:
        result = AtomicAttrMixin._string(self)
        return f"{result},{GraphAttrMixin._string(self)}"

    @pydantic.validate_call
    def infer_connectivity(
        self,
        plus_factor: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.5,
        multiply_factor: Annotated[float, pydantic.Field(ge=0.8, le=1.5)] = 1.0,
    ) -> csr_array:
        m = get_adjacency_sparse_matrix(
            cov_multiply_factor=multiply_factor,
            cov_plus_factor=plus_factor,
            geometry=self.positions,
            numbers=self.numbers,
            cell=self.CELL,
        ).astype(bool, copy=False)
        return csr_array(sp_triu(m, k=1))

    @pydantic.validate_call(config={"arbitrary_types_allowed": True})
    def infer_bond_order(
        self,
        arr: ArrayLike,
        charge: pydantic.NonNegativeInt = 0,
    ) -> csr_array:
        m_conn = csr_array(arr, shape=(self.natoms, self.natoms), dtype=bool)
        m_conn_coo = coo_matrix(sp_triu(m_conn + m_conn.T, k=1))
        m = get_adjacency_by_rdkit(
            source=m_conn_coo.row,
            target=m_conn_coo.col,
            numbers=self.numbers,
            geometry=self.positions,
            infer_order=True,
            charge=int(charge),
        ).astype(float, copy=False)
        return csr_array(sp_triu(m, k=1))

    def infer_bond(
        self,
        plus_factor: float = 0.5,
        multiply_factor: float = 1,
        infer_order: bool = False,
        charge: int = 0,
    ) -> csr_array:
        conn = self.infer_connectivity(plus_factor, multiply_factor)
        return self.infer_bond_order(conn, charge) if infer_order else conn

    def infer_bond_as_dict(
        self,
        plus_factor: float = 0.5,
        multiply_factor: float = 1,
        infer_order: bool = False,
        charge: int = 0,
    ) -> dict[str, Any]:
        return self._infer_bond(
            dct=self.model_dump(exclude_none=True),
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            infer_order=infer_order,
            charge=charge,
        )

    @classmethod
    def _infer_bond(
        cls,
        dct: dict[str, Any],
        plus_factor: float = 0.5,
        multiply_factor: float = 1,
        infer_order: bool = False,
        charge: int = 0,
    ) -> dict[str, Any]:
        obj = AtomicContainner.model_validate(dct)
        if BOND_KEY.CONNECTIVITY not in dct:
            m = obj.infer_connectivity(plus_factor, multiply_factor)
            dct[BOND_KEY.CONNECTIVITY] = np.column_stack(coo_matrix(m).coords)
            if not infer_order:
                return dct
        if BOND_KEY.ORDER not in dct and infer_order:
            # infer bond order by rdkit if needed
            conn = np.asarray(dct[BOND_KEY.CONNECTIVITY], dtype=int)
            assert conn.ndim == 2 and conn.shape[1] == 2, conn.shape
            v, (row, col) = np.ones(conn.shape[0], dtype=bool), conn.T
            m = csr_array((v, (row, col)), shape=(obj.natoms, obj.natoms))
            o = coo_matrix(obj.infer_bond_order(m, charge))
            dct[BOND_KEY.CONNECTIVITY] = np.column_stack(o.coords)
            dct[BOND_KEY.ORDER] = o.data
            return dct
        else:
            return dct

    #########################################################################
    #                          ASE Atoms Interface                          #
    #########################################################################
    @staticmethod
    def __ase_exclude_keys_set() -> set[str]:
        return {ATOM_KEY.NUMBER, ATOM_KEY.POSITION, TOTAL_KEY.BOX}

    def as_ase(self) -> Atoms:
        return Atoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.CELL,
            pbc=self.PBC,
            info=self.model_dump(
                mode="python",
                exclude_none=True,
                exclude=self.__ase_exclude_keys_set(),
            ),
        )

    @classmethod
    def from_ase(cls, atoms: Atoms) -> Self:
        if not atoms.cell.orthorhombic:
            raise NotSupportNonOrthorhombicLattice()
        dct: dict[str, Any] = atoms.info
        dct[ATOM_KEY.NUMBER] = atoms.numbers
        dct[ATOM_KEY.POSITION] = atoms.positions
        if np.sum(atoms.cell.array.any(1) & atoms.pbc) > 0:
            cell = atoms.cell.complete().minkowski_reduce()[0]
            dct[TOTAL_KEY.BOX] = cell.cellpar()
        return cls.model_validate(dct)
