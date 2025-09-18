from typing import override

import numpy as np
import pydantic
from sqlmodel import Field
from typing_extensions import Any, Self

from GraphAtoms.containner import Cluster, Gas, Graph, System
from GraphAtoms.reaction._dtPyArrow import (
    ABCdt,
    ClusterItem,
    GasItem,
    GraphItem,
    SystemItem,
)


class _SQLABC(ABCdt, table=True):
    id: int | None = Field(default=None, primary_key=True)
    graph_hash: str = Field(default="", index=True)
    data_hash: str = Field(default="", index=True)
    formula: str = Field(index=True)
    natoms: int = Field(index=True)
    nbonds: int = Field(index=True)

    @classmethod
    @override
    def _validate(cls) -> None: ...

    @override
    def _convert_to(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in ABCdt._convert_to(self).items()
            if k
            not in (
                {"id", "graph_hash", "data_hash"}
                | {"formula", "natoms", "nbonds"}
            )
        }

    @classmethod
    @override
    def _convert_from(cls, data: Graph) -> dict[str, Any]:
        return {
            k: (v.tobytes() if isinstance(v, np.ndarray) else v)
            for k, v in data.model_dump(exclude_none=True).items()
            if k in cls.__pydantic_fields__.keys()
        } | {
            "graph_hash": data.hash,
            "data_hash": data._data_hash,
            "formula": data.symbols.get_chemical_formula("metal"),
            "natoms": data.natoms,
            "nbonds": data.nbonds,
        }


class SystemSQL(_SQLABC, SystemItem):
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


class ClusterSQL(_SQLABC, ClusterItem):
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


class GraphSQL(_SQLABC, GraphItem):
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


class GasSQL(_SQLABC, GasItem, table=False):
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

    @classmethod
    @override
    def _convert_from(cls, data: Graph) -> dict[str, Any]:
        result = {
            k: (v.tobytes() if isinstance(v, np.ndarray) else v)
            for k, v in data.model_dump(exclude_none=True).items()
            if k in cls.__pydantic_fields__.keys()
        } | {
            "graph_hash": data.hash,
            "data_hash": data._data_hash,
            "formula": data.symbols.get_chemical_formula("metal"),
            "natoms": data.natoms,
            "nbonds": data.nbonds,
        }
        for k, v in result.items():
            print(k, v)
        cls.model_validate(result)
        assert False

        return result


if __name__ == "__main__":
    from pydantic_to_pyarrow import get_pyarrow_schema

    for cls in [GraphSQL, SystemSQL, ClusterSQL, GasSQL]:
        print(cls.__name__, cls._dataclass(), cls.__pydantic_fields__.keys())
        print(get_pyarrow_schema(cls))
        print("-" * 32)
    gas = Gas.from_molecule("CO")
    print(gas)
    print(GasSQL.convert_from(gas))
