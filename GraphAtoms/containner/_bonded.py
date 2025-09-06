from abc import abstractmethod
from functools import cached_property

from numpy import asarray, float16, int32, ndarray, ones
from pandas import DataFrame
from pydantic import model_validator
from scipy.sparse import csr_array
from typing_extensions import Self, override

from GraphAtoms.common.abc import NpzPklBaseModel
from GraphAtoms.common.ndarray import NDArray, Shape


class __BondKey:
    CONNECTIVITY = "conn"
    ORDER = "order"

    @property
    def _DICT(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in dir(self) if k[0] != "_"}


BOND_KEY = __BondKey()


class BondAttrMixin(NpzPklBaseModel):
    conn: NDArray[Shape["*,2"], int32]  # type: ignore
    order: NDArray[Shape["*"], float16] | None = None  # type: ignore

    @classmethod
    @override
    def _convert(cls) -> dict[str, tuple[tuple, str]]:
        result: dict[str, tuple[tuple, str]] = super()._convert()
        result["conn"] = ((None, 2), "int32")
        result["order"] = ((None,), "float16")
        return result

    @model_validator(mode="after")
    def __check_keys(self) -> Self:
        assert self.nbonds != 0, "No bonds"
        fields = self.__pydantic_fields__.keys()
        assert set(fields) >= set(BOND_KEY._DICT.values())
        return self

    @property
    @abstractmethod
    def natoms(self) -> int: ...

    @property
    def nbonds(self) -> int:
        return self.conn.shape[0]

    @classmethod
    def DF_BONDS_PARSER(cls, df: DataFrame) -> dict[str, ndarray]:
        assert len(df.columns) >= 2, df.columns
        dct = {BOND_KEY.CONNECTIVITY: df[df.columns[:2]].to_numpy()}
        for k in set(df.columns[2:]) & set(BOND_KEY._DICT.values()):
            dct[k] = df[k].to_numpy()
        return dct

    @property
    def DF_BONDS(self) -> DataFrame:
        df = DataFrame(self.conn, columns=["i", "j"])
        if self.order is not None:
            df[BOND_KEY.ORDER] = self.order
        return df

    @property
    def MATRIX(self) -> csr_array:
        return self.__MATRIX

    @cached_property
    def __MATRIX(self) -> csr_array:
        shp, nb = (self.natoms, self.natoms), self.nbonds
        if self.order is None:
            return csr_array((ones(nb, bool), self.conn.T), shape=shp)
        else:
            # scipy.sparse does not support dtype float16(self.order.dtype)
            # so we use single float numbers(float32) in the return statement
            return csr_array((self.order, self.conn.T), shape=shp, dtype="f4")

    @cached_property
    def CN_MATRIX(self) -> ndarray:
        m = self.__MATRIX.astype(bool)
        m = csr_array((m + m.T).astype(int))
        return asarray(m.sum(axis=1)).astype(int)

    @override
    def _string(self) -> str:
        return f"{self.nbonds}Bonds"
