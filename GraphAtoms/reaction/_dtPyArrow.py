from typing import override

import numpy as np
import pydantic
import sqlmodel
from pyarrow import Schema
from pydantic_to_pyarrow import get_pyarrow_schema
from typing_extensions import Any, Self

from GraphAtoms.containner import Cluster, Gas, Graph, System


class ABCdt(sqlmodel.SQLModel):
    @classmethod
    def _dataclass(cls) -> type[Graph]:
        """The base data class for validation."""
        raise NotImplementedError

    @classmethod
    def _validate(cls) -> None:
        assert set(cls.__pydantic_fields__.keys()) <= set(
            cls._dataclass().__pydantic_fields__.keys()
        ), (
            f"Invalid fields. Please check the "
            f"`__pydantic_fields__` of `{cls._dataclass().__name__}`."
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def __validate(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert isinstance(get_pyarrow_schema(cls), Schema), (
            f"Cannot convert {cls.__name__} to pyarrow."
        )
        return values

    def _convert_to(self) -> dict[str, Any]:
        dct = self.model_dump(exclude_none=True)
        cls: type[Graph] = self._dataclass()
        convert_dct = cls._convert()
        for k, v in dct.items():
            if k not in convert_dct:
                raise ValueError(f"Cannot convert field({k}) to numpy.")
            v = np.frombuffer(v, convert_dct[k][1])
            dct[k] = v.reshape(convert_dct[k][0])
        return dct

    def convert_to(self) -> Graph:
        """Convert to the base data class."""
        return self._dataclass().model_validate(self._convert_to())

    @classmethod
    def _convert_from(cls, data: Graph) -> dict[str, Any]:
        return {
            k: (v.tobytes() if isinstance(v, np.ndarray) else v)
            for k, v in data.model_dump(exclude_none=True).items()
            if k in cls.__pydantic_fields__.keys()
        }

    @classmethod
    def convert_from(cls, data: Graph) -> Self:
        """Convert from the base data class."""
        dct: dict = cls._convert_from(data)
        return cls.model_validate(dct)


class _ItemABC(ABCdt):
    # Atoms
    numbers: bytes
    positions: bytes

    # Energetics
    energy: float | None = sqlmodel.Field(default=None, index=True)
    frequencies: bytes | None = sqlmodel.Field(default=None)
    fmax_nonconstraint: float | None = sqlmodel.Field(
        default=None,
        index=True,
        ge=0,
    )
    fmax_constraint: float | None = sqlmodel.Field(
        default=None,
        index=True,
        ge=0,
    )

    # Bonds
    order: bytes | None = sqlmodel.Field(default=None)
    source: bytes
    target: bytes


class SystemItem(_ItemABC):
    is_outer: bytes | None = sqlmodel.Field(default=None)
    a: float = sqlmodel.Field(default=0, index=True, ge=0)
    b: float = sqlmodel.Field(default=0, index=True, ge=0)
    c: float = sqlmodel.Field(default=0, index=True, ge=0)
    alpha: float = sqlmodel.Field(default=90, index=True, ge=0, le=180)
    beta: float = sqlmodel.Field(default=90, index=True, ge=0, le=180)
    gamma: float = sqlmodel.Field(default=90, index=True, ge=0, le=180)
    shift_x: bytes | None = sqlmodel.Field(default=None)
    shift_y: bytes | None = sqlmodel.Field(default=None)
    shift_z: bytes | None = sqlmodel.Field(default=None)

    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return System

    @override
    @pydantic.validate_call
    def convert_to(self) -> System:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @pydantic.validate_call
    def convert_from(cls, data: System) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore


class ClusterItem(_ItemABC):
    move_fix_tag: bytes | None = sqlmodel.Field(default=None)
    coordination: bytes | None = sqlmodel.Field(default=None)

    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return Cluster

    @override
    @pydantic.validate_call
    def convert_to(self) -> Cluster:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @pydantic.validate_call
    def convert_from(cls, data: Cluster) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore


class GraphItem(SystemItem, ClusterItem):
    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return Graph

    @override
    @pydantic.validate_call
    def convert_to(self) -> Graph:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @pydantic.validate_call
    def convert_from(cls, data: Graph) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore


class GasItem(_ItemABC):
    sticking: float = sqlmodel.Field(default=1, index=True, ge=0, le=1e2)
    pressure: float = sqlmodel.Field(default=101325, index=True, ge=0)

    @classmethod
    @override
    def _dataclass(cls) -> type[Graph]:
        return Gas

    @override
    @pydantic.validate_call
    def convert_to(self) -> Gas:  # type: ignore
        return super().convert_to()  # type: ignore

    @classmethod
    @override
    @pydantic.validate_call
    def convert_from(cls, data: Gas) -> Self:  # type: ignore
        return super().convert_from(data)  # type: ignore


if __name__ == "__main__":
    for cls in [GraphItem, SystemItem, ClusterItem, GasItem]:
        print(cls.__name__, cls._dataclass(), cls.__pydantic_fields__.keys())
        print(get_pyarrow_schema(cls))
        print("-" * 32)
    gas = Gas.from_molecule("CO")
    print(gas)
    print(GasItem.convert_from(gas))
