# ruff: noqa: D100, D101, D102, D103, D104
import pyarrow as pa
import pytest

from GraphAtoms.containner import Cluster, Gas, Graph, System
from GraphAtoms.reaction._dtSQL import (
    SQLABC,
    ClusterSQL,
    GasSQL,
    GraphSQL,
    SystemSQL,
)


class Test_PyArrowCompability:
    @staticmethod
    def get_all_item_classes() -> list[type[SQLABC]]:
        return [GasSQL, GraphSQL, SystemSQL, ClusterSQL]

    def get_obj(self, system: System, cls: type[SQLABC]) -> Graph:
        if cls is ClusterSQL:
            obj = Cluster.select_by_hop(
                system,
                system.get_hop_distance(0),
            )
        elif cls is GraphSQL:
            obj = Graph.from_ase(system.to_ase())
        elif cls is SystemSQL:
            obj = system
        elif cls is GasSQL:
            obj = Gas.from_molecule("CO")
        else:
            raise ValueError(f"Unknown class: {cls}")
        return obj

    @pytest.mark.parametrize("cls_id", sorted(range(4)))
    def test_XxxItem_pyarrow_compability(self, cls_id: int) -> None:
        cls: type[SQLABC] = self.get_all_item_classes()[cls_id]
        print(cls.get_pyarrow_schema(), "-" * 32, sep="\n")

    @pytest.mark.parametrize("cls_id", sorted(range(4)))
    def test_Xxx_as_PyArrow_Table(self, system: System, cls_id: int) -> None:
        cls: type[SQLABC] = self.get_all_item_classes()[cls_id]
        obj = self.get_obj(system, cls)
        obj_sql = cls.convert_from(obj)  # type: ignore
        data = obj_sql.model_dump(exclude_none=True)
        schema: pa.Schema = cls.get_pyarrow_schema()
        print(pa.Table.from_pylist([data] * 5, schema=schema))
        obj2 = obj_sql.convert_to()  # type: ignore
        assert isinstance(obj2, type(obj))
        assert obj == obj2


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-s"])
