from abc import abstractmethod
from collections.abc import Callable
from typing import Literal, override

from numpy import ndarray
from pyarrow import Schema
from pydantic import model_validator, validate_call
from pydantic.main import IncEx
from sqlmodel import Column, Field, SQLModel
from sqlmodel._compat import SQLModelConfig
from typing_extensions import Any, Self

from GraphAtoms.common import XxxKeyMixin
from GraphAtoms.containner import Cluster, Gas, Graph, System
from GraphAtoms.utils._pyarrow import get_pyarrow_schema


class __SQLKey(XxxKeyMixin):
    ID = "id"
    GRAPH_HASH = "graph_hash"
    DATA_HASH = "data_hash"
    FORMULA = "formula"
    NATOMS = "natoms"
    NBONDS = "nbonds"


SQL_KEY = __SQLKey()
__all__ = ["SQL_KEY"]


class _ConvertMixin(SQLModel):
    @classmethod
    def _dataclass(cls) -> type[Graph]:
        """The base data class for validation."""
        return Graph

    def convert_to(self) -> Graph:
        """Convert to the base data class."""
        dct = self.model_dump(
            exclude_none=True,
            exclude=set(SQL_KEY._DICT.values()),
        )
        cls: type[Graph] = self._dataclass()
        return cls.model_validate(dct)

    @classmethod
    def convert_from(cls, data: Graph) -> Self:
        """Convert from the base data class."""
        dct: dict[str, Any] = {
            k: (v.tobytes() if isinstance(v, ndarray) else v)
            for k, v in data.model_dump(exclude_none=True).items()
            if k in cls.__pydantic_fields__.keys()
        } | {
            "graph_hash": data.hash,
            "data_hash": data.get_data_hash(),
            "formula": data.symbols.get_chemical_formula("metal"),
            "natoms": data.natoms,
            "nbonds": data.nbonds,
        }
        return cls.model_validate(dct)


class _BaseModel(SQLModel):
    model_config = SQLModelConfig(
        ser_json_bytes="base64",  # type: ignore
        val_json_bytes="base64",  # type: ignore
    )

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
        assert isinstance(cls.get_pyarrow_schema(), Schema), (
            f"Cannot convert {cls.__name__} to pyarrow."
        )  # assert this class can be converted to pyarrow
        return values

    @classmethod
    @abstractmethod
    def _dataclass(cls) -> type[Graph]:
        """The base data class for validation."""
        raise NotImplementedError

    @classmethod
    def get_pyarrow_schema(cls) -> Schema:
        """Get the pyarrow schema of this class."""
        return get_pyarrow_schema(cls)

    def __get_include_and_exclude(
        self,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
    ) -> tuple[IncEx | None, IncEx | None]:
        is_column_set, is_column_unset = set(), set()
        for k in self.__pydantic_fields__.keys():
            v = getattr(self, k)
            if isinstance(v, Column):
                is_column_set.add(k)
            else:
                is_column_unset.add(k)
        if exclude is None:
            exclude = is_column_set
        elif include is None:
            include = is_column_unset
        else:
            raise KeyError("Cannot set both `include` and `exclude` as None.")
        return include, exclude

    @override
    def model_dump(
        self,
        *,
        mode: Literal["json", "python"] | str = "python",
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        include, exclude = self.__get_include_and_exclude(
            include=include, exclude=exclude
        )
        return super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> str:
        include, exclude = self.__get_include_and_exclude(
            include=include, exclude=exclude
        )
        return super().model_dump_json(
            indent=indent,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )


class SQLABC(_BaseModel):
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


class _BaseSystemSQL(SQLABC):
    is_outer: bytes | None = Field(default=None)
    shift_x: bytes | None = Field(default=None)
    shift_y: bytes | None = Field(default=None)
    shift_z: bytes | None = Field(default=None)
    box: bytes | None = Field(default=None)


class _BaseClusterSQL(SQLABC):
    move_fix_tag: bytes | None = Field(default=None)
    coordination: bytes | None = Field(default=None)


class GraphSQL(_ConvertMixin, _BaseSystemSQL, _BaseClusterSQL, table=True):
    pass


class SystemSQL(_BaseSystemSQL, _ConvertMixin, table=True):
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


class ClusterSQL(_BaseClusterSQL, _ConvertMixin, table=True):
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


class GasSQL(SQLABC, _ConvertMixin, table=True):
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
