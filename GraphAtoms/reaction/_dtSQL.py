from abc import abstractmethod
from typing import override

from numpy import frombuffer, ndarray
from pyarrow import Schema
from pydantic import model_validator, validate_call
from pydantic_to_pyarrow import get_pyarrow_schema
from sqlmodel import Field, SQLModel
from typing_extensions import Any, Self

from GraphAtoms.common import XxxKeyMixin
from GraphAtoms.containner import Cluster, Gas, Graph, System


class __SQLKey(XxxKeyMixin):
    ID = "id"
    GRAPH_HASH = "graph_hash"
    DATA_HASH = "data_hash"
    FORMULA = "formula"
    NATOMS = "natoms"
    NBONDS = "nbonds"


SQL_KEY = __SQLKey()
__all__ = ["SQL_KEY"]


class _ABCdtSQL(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    graph_hash: str = Field(default="", index=True)
    data_hash: str = Field(default="", index=True)
    formula: str = Field(index=True)
    natoms: int = Field(index=True)
    nbonds: int = Field(index=True)

    @model_validator(mode="before")
    @classmethod
    def __validate(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert set(cls.__pydantic_fields__.keys()) <= set(
            cls._dataclass().__pydantic_fields__.keys()
        ) | set(SQL_KEY._DICT.values()), (
            f"Invalid fields. Please check the "
            f"`__pydantic_fields__` of `{cls._dataclass().__name__}`."
        )
        assert isinstance(get_pyarrow_schema(cls), Schema), (
            f"Cannot convert {cls.__name__} to pyarrow."
        )  # assert this class can be converted to pyarrow
        return values

    @classmethod
    @abstractmethod
    def _dataclass(cls) -> type[Graph]:
        """The base data class for validation."""
        raise NotImplementedError

    @abstractmethod
    def convert_to(self) -> Graph:
        """Convert to the base data class."""
        dct = self.model_dump(
            exclude_none=True,
            exclude=set(SQL_KEY._DICT.values()),
        )
        cls: type[Graph] = self._dataclass()
        convert_dct = cls._convert()
        for k, v in dct.items():
            if k not in convert_dct:
                raise ValueError(f"Cannot convert field({k}) to numpy.")
            v = frombuffer(v, convert_dct[k][1])
            dct[k] = v.reshape(convert_dct[k][0])
        return self._dataclass().model_validate(dct)

    @classmethod
    @abstractmethod
    def convert_from(cls, data: Graph) -> Self:
        """Convert from the base data class."""
        dct: dict[str, Any] = {
            k: (v.tobytes() if isinstance(v, ndarray) else v)
            for k, v in data.model_dump(exclude_none=True).items()
            if k in cls.__pydantic_fields__.keys()
        } | {
            "graph_hash": data.hash,
            "data_hash": data._data_hash,
            "formula": data.symbols.get_chemical_formula("metal"),
            "natoms": data.natoms,
            "nbonds": data.nbonds,
        }
        return cls.model_validate(dct)


class _SQLABC(_ABCdtSQL):
    # Atoms
    numbers: bytes
    positions: bytes

    # Energetics
    frequencies: bytes | None = Field(default=None)
    energy: float | None = Field(default=None, index=True)
    fmax_nonconstraint: float | None = Field(default=None, index=True, ge=0)
    fmax_constraint: float | None = Field(default=None, index=True, ge=0)

    # Bonds
    order: bytes | None = Field(default=None)
    source: bytes
    target: bytes


class SystemSQL(_SQLABC):
    is_outer: bytes | None = Field(default=None)
    a: float = Field(default=0, index=True, ge=0)
    b: float = Field(default=0, index=True, ge=0)
    c: float = Field(default=0, index=True, ge=0)
    alpha: float = Field(default=90, index=True, ge=0, le=180)
    beta: float = Field(default=90, index=True, ge=0, le=180)
    gamma: float = Field(default=90, index=True, ge=0, le=180)
    shift_x: bytes | None = Field(default=None)
    shift_y: bytes | None = Field(default=None)
    shift_z: bytes | None = Field(default=None)

    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return System

    @override
    @validate_call
    def convert_to(self) -> System:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @validate_call
    def convert_from(cls, data: System) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore


class ClusterSQL(_SQLABC):
    move_fix_tag: bytes | None = Field(default=None)
    coordination: bytes | None = Field(default=None)

    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return Cluster

    @override
    @validate_call
    def convert_to(self) -> Cluster:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @validate_call
    def convert_from(cls, data: Cluster) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore


class GraphSQL(SystemSQL, ClusterSQL):
    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return Graph

    @override
    @validate_call
    def convert_to(self) -> Graph:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @validate_call
    def convert_from(cls, data: Graph) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore


class GasSQL(_SQLABC):
    sticking: float = Field(default=1, index=True, ge=0, le=1e2)
    pressure: float = Field(default=101325, index=True, ge=0)

    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return Gas

    @override
    @validate_call
    def convert_to(self) -> Gas:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @validate_call
    def convert_from(cls, data: Gas) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore
