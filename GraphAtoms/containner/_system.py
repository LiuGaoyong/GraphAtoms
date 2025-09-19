from numpy import frombuffer, ndarray
from pyarrow import Schema
from pydantic import model_validator, validate_call
from pydantic_to_pyarrow import get_pyarrow_schema
from sqlmodel import Field, SQLModel
from typing_extensions import Any, Self, override

from GraphAtoms.containner._atomic import ATOM_KEY
from GraphAtoms.containner._graph import Graph


class System(Graph):
    """The whole system."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        msg = "The key of `{:s}` should be None for System."
        for k in (ATOM_KEY.MOVE_FIX_TAG, ATOM_KEY.COORDINATION):
            assert getattr(self, k) is None, msg.format(k)
        return self


class _PyArrowItemABC(SQLModel):  # Atoms
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

    @classmethod
    def _dataclass(cls) -> type[Graph]:
        """The base data class for validation."""
        return Graph

    @model_validator(mode="before")
    @classmethod
    def __validate(cls, values: dict[str, Any]) -> dict[str, Any]:
        assert set(cls.__pydantic_fields__.keys()) <= set(
            cls._dataclass().__pydantic_fields__.keys()
        ), (
            f"Invalid fields. Please check the "
            f"`__pydantic_fields__` of `{cls._dataclass().__name__}`."
        )
        assert isinstance(get_pyarrow_schema(cls), Schema), (
            f"Cannot convert {cls.__name__} to pyarrow."
        )
        return values

    def convert_to(self) -> Graph:
        """Convert to the base data class."""
        dct = self.model_dump(exclude_none=True)
        cls: type[Graph] = self._dataclass()
        convert_dct = cls._convert()
        for k, v in dct.items():
            if k not in convert_dct:
                raise ValueError(f"Cannot convert field({k}) to numpy.")
            v = frombuffer(v, convert_dct[k][1])
            dct[k] = v.reshape(convert_dct[k][0])
        return self._dataclass().model_validate(dct)

    @classmethod
    def convert_from(cls, data: Graph) -> Self:
        """Convert from the base data class."""
        dct: dict = {
            k: (v.tobytes() if isinstance(v, ndarray) else v)
            for k, v in data.model_dump(exclude_none=True).items()
            if k in cls.__pydantic_fields__.keys()
        }
        return cls.model_validate(dct)


class SystemItem(_PyArrowItemABC):
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
    def _dataclass(cls) -> type[System]:
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
