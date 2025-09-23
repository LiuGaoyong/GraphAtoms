from abc import ABC, abstractmethod
from functools import cached_property
from typing import Annotated, Any

import numpy as np
import pydantic
from ase import Atoms
from numpy import ndarray, where
from pandas import DataFrame
from scipy import sparse as sp
from typing_extensions import Self, override

from GraphAtoms.common import NpzPklBaseModel, XxxKeyMixin
from GraphAtoms.containner._atomic import AtomsWithBoxEng
from GraphAtoms.utils.geometry import get_adjacency_sparse_matrix
from GraphAtoms.utils.ndarray import NDArray, numpy_validator
from GraphAtoms.utils.rdtool import get_adjacency_by_rdkit


class __BondKey(XxxKeyMixin):
    SHIFT_X = "shift_x"
    SHIFT_Y = "shift_y"
    SHIFT_Z = "shift_z"
    SOURCE = "source"
    TARGET = "target"
    ORDER = "order"


BOND_KEY = __BondKey()
ArrayLike = sp.spmatrix | sp.sparray | ndarray
__all__ = ["BOND_KEY", "Bonds"]


class Bonds(NpzPklBaseModel):
    source: Annotated[NDArray, numpy_validator("int32")]
    target: Annotated[NDArray, numpy_validator("int32")]
    shift_x: Annotated[NDArray, numpy_validator("int8")] | None = None
    shift_y: Annotated[NDArray, numpy_validator("int8")] | None = None
    shift_z: Annotated[NDArray, numpy_validator("int8")] | None = None
    order: Annotated[NDArray, numpy_validator("float16")] | None = None

    @pydantic.model_validator(mode="after")
    def __check_keys(self) -> Self:
        assert self.nbonds != 0, "No bonds"
        for k in BOND_KEY._DICT.values():
            v = getattr(self, k, None)
            if v is not None:
                assert isinstance(v, np.ndarray), f"Invalid {k}."
                assert v.shape == (self.nbonds,), f"Invalid shape for {k}."
        return self

    @override
    def _string(self) -> str:
        return f"{self.nbonds}Bonds"

    @property
    @abstractmethod
    def natoms(self) -> int: ...

    @property
    def nbonds(self) -> int:
        return self.source.shape[0]

    @classmethod
    def DF_BONDS_PARSER(cls, df: DataFrame) -> dict[str, np.ndarray]:
        assert len(df.columns) >= 2, df.columns
        dct = {
            BOND_KEY.SOURCE: df[df.columns[0]].to_numpy(),
            BOND_KEY.TARGET: df[df.columns[1]].to_numpy(),
        }
        for k in set(df.columns[2:]) & set(BOND_KEY._DICT.values()):
            if not df[k].isnull().all():
                dct[k] = df[k].to_numpy()
        return dct

    @property
    def DF_BONDS(self) -> DataFrame:
        inc: set[str] = set(BOND_KEY._DICT.values())
        return DataFrame(self.model_dump(include=inc, exclude_none=True))

    @property
    def MATRIX(self) -> sp.csr_array:
        return self.__MATRIX

    @cached_property
    def __MATRIX(self) -> sp.csr_array:
        if self.order is None:
            order = np.ones(self.nbonds, bool)
        else:
            # scipy.sparse does not support dtype float16(self.order.dtype)
            # so we use single float numbers(float32) in the return statement
            order = np.asarray(self.order, dtype="f4")
        return sp.csr_array(
            (order, (self.source, self.target)),
            shape=(self.natoms, self.natoms),
        )

    @cached_property
    def CN_MATRIX(self) -> np.ndarray:
        m = self.__MATRIX.astype(bool)
        m = sp.csr_array((m + m.T).astype(int))
        return np.asarray(m.sum(axis=1)).astype(int)


class BondsInitializer:
    @classmethod
    @pydantic.validate_call(config={"arbitrary_types_allowed": True})
    def infer_connectivity(
        cls,
        obj: AtomsWithBoxEng,
        plus_factor: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.5,
        multiply_factor: Annotated[float, pydantic.Field(ge=0.8, le=1.5)] = 1.0,
    ) -> sp.csr_array:
        m = get_adjacency_sparse_matrix(
            cov_multiply_factor=multiply_factor,
            cov_plus_factor=plus_factor,
            geometry=obj.positions,
            numbers=obj.numbers,
            cell=obj.ase_cell,
        ).astype(bool, copy=False)
        return sp.csr_array(sp.triu(m, k=1))

    @classmethod
    @pydantic.validate_call(config={"arbitrary_types_allowed": True})
    def infer_bond(
        cls,
        obj: AtomsWithBoxEng | Atoms,
        plus_factor: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.5,
        multiply_factor: Annotated[float, pydantic.Field(ge=0.8, le=1.5)] = 1.0,
        charge: pydantic.NonNegativeInt = 0,
        infer_order: bool = False,
    ) -> sp.csr_array:
        if not isinstance(obj, AtomsWithBoxEng):
            obj = AtomsWithBoxEng.from_ase(obj)
        result = cls.infer_connectivity(obj, plus_factor, multiply_factor)
        if infer_order:
            m_conn = sp.triu(result + result.T, k=1)
            m_conn_coo = sp.coo_matrix(m_conn)
            m = get_adjacency_by_rdkit(
                numbers=obj.numbers,
                geometry=obj.positions,
                source=m_conn_coo.row,
                target=m_conn_coo.col,
                infer_order=True,
                charge=int(charge),
            ).astype(float, copy=False)
            result = sp.csr_array(sp.triu(m, k=1))
        return result

    @classmethod
    @pydantic.validate_call(config={"arbitrary_types_allowed": True})
    def infer_bond_as_dict(
        cls,
        obj: AtomsWithBoxEng | Atoms,
        plus_factor: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.5,
        multiply_factor: Annotated[float, pydantic.Field(ge=0.8, le=1.5)] = 1.0,
        charge: pydantic.NonNegativeInt = 0,
        infer_order: bool = False,
    ) -> dict[str, Any]:
        m = cls.infer_bond(
            obj,
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            infer_order=infer_order,
            charge=charge,
        ).tocoo()
        result = {
            BOND_KEY.SOURCE: m.row,
            BOND_KEY.TARGET: m.col,
        }
        if infer_order:
            result[BOND_KEY.ORDER] = m.data
        return result


class GraphMixinConnectedComponents(ABC):
    """The function mixin class by `scipy.sparse.csgraph`."""

    @property
    @abstractmethod
    def MATRIX(self) -> sp.csr_array: ...

    @cached_property
    def connected_components_label(self) -> ndarray:
        return sp.csgraph.connected_components(
            csgraph=self.MATRIX,
            directed=False,
            return_labels=True,
        )[1]

    @property
    def connected_components_number(self) -> int:
        return max(self.connected_components_label) + 1

    @cached_property
    def connected_components(self) -> list[ndarray]:
        labels = self.connected_components_label
        n: int = self.connected_components_number
        result = [where(labels == i)[0] for i in range(n)]
        return sorted(result, reverse=True, key=lambda x: len(x))

    @cached_property
    def connected_components_biggest(self) -> ndarray:
        cc: list[ndarray] = self.connected_components
        ccl: list[int] = [len(i) for i in cc]
        i_biggest = ccl.index(max(ccl))
        return cc[i_biggest]

    @property
    def is_connected(self) -> bool:
        return self.connected_components_number == 1

    @property
    def is_disconnected(self) -> bool:
        return self.connected_components_number != 1


class BondsWithComp(Bonds, BondsInitializer, GraphMixinConnectedComponents):
    pass
