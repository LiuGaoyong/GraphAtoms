# ruff: noqa: D100, D101, D102, D103, D104
import random
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pytest
from ase import Atoms
from ase.build import molecule
from ase.cluster import Octahedron

from GraphAtoms.system import Cluster, Gas, System
from GraphAtoms.system import SysGraph as Graph


@pytest.fixture(scope="module")
def system() -> System:
    result = System.from_ase(Octahedron("Cu", 3))
    assert result.pair is not None
    return result


def _equal_test(obj1: Graph, obj2: Graph) -> None:
    if obj1 != obj2:
        for k in obj1.__pydantic_fields__:
            v1 = getattr(obj1, k)
            v2 = getattr(obj2, k)
            if type(v1) != type(v2):  # noqa: E721
                print(f"Type mismatch for {k}: {type(v1)} != {type(v2)}")
            else:
                if isinstance(v1, np.ndarray):
                    if v1.dtype in ("float16", "float32", "float64"):
                        assert np.allclose(v1, v2, rtol=1e-5, atol=1e-8), (
                            f"Value mismatch for {k}: \n{v1}\n != \n{v2}"
                        )
                    else:
                        assert np.array_equal(v1, v2), (
                            f"Value mismatch for {k}: \n{v1}\n != \n{v2}"
                        )
                else:
                    assert v1 == v2, f"Value mismatch for {k}: {v1} != {v2}"


class Test_ContainerBasic:
    @pytest.mark.parametrize("cls", [Cluster, Gas, Graph, System])
    def test_cls_frozen(self, cls: type[Graph]) -> None:
        assert cls.model_config.get("frozen", False)

    def test_graph_basic(self) -> None:
        obj_order = Gas.from_molecule("CH4", pressure=101325)
        print("*" * 32, "Test PyGData from obj_order")
        pygdata = obj_order.to_pygdata()
        print(pygdata, pygdata.num_edges, pygdata.num_nodes)
        assert pygdata.num_nodes == obj_order.natoms
        assert pygdata.num_edges == obj_order.nbonds
        for k in pygdata.node_attrs():
            v = pygdata[k]
            print("NODE", k, type(v))
        for k in pygdata.edge_attrs():
            v = pygdata[k]
            print("EDGE", k, type(v))
        print("*" * 32, "Test PyGData equality from obj_order")
        new_obj_order = Graph.from_pygdata(pygdata)
        print(repr(new_obj_order), "\n", repr(obj_order))
        _equal_test(new_obj_order, obj_order)

    def test_bondgraph(self) -> None:
        atoms = molecule("C6H6")
        lst = list(range(len(atoms)))
        gas = Gas.from_ase(Atoms([atoms[i] for i in lst]), pressure=101325)
        data0 = (gas.hash, gas.smiles)
        random.shuffle(lst)
        gas = Gas.from_ase(Atoms([atoms[i] for i in lst]), pressure=101325)
        print(gas.hash)
        data1 = (gas.hash, gas.smiles)
        assert data0 == data1, f"{data0} != {data1}"


class Test_Container:
    def test_get_hop_distance(self, system: System) -> None:
        print(system.get_hop_distance(0))  # type: ignore
        k = np.zeros(system.natoms, bool)
        k[0] = True
        print(system.get_hop_distance(k))

    def test_select_cluster(self, system: System) -> None:
        print(system.P)
        sub = Cluster.select_by_hop(
            system,
            system.get_hop_distance(0),  # type: ignore
            max_moved_hop=0,
            env_hop=1,
        )
        print(sub.is_outer)
        print(
            "-" * 32,
            Cluster.model_json_schema(),
            "-" * 32,
            sub,
            repr(sub),
            "-" * 32,
            sep="\n",
        )
        print("=" * 32)
        sub2 = Cluster.select_by_distance(
            system,
            np.asarray([0]),
            env_distance=3.2,
            max_moved_distance=0.0,
        )
        print(
            sub,
            sub2,
            "-" * 32,
            sub.move_fix_tag,
            sub2.move_fix_tag,
            sep="\n",
        )

    def test_len(self, system: System) -> None:
        assert len(system) == 19

    def test_repr(self, system: System) -> None:
        print(str(system), repr(system), sep="\n")

    def test_eq(self, system: System) -> None:
        assert system.__eq__(system), "System equality test fail!!!"

    def test_hash(self, system: System) -> None:
        print(system.__hash__)
        lst = [hash(i) for i in [system] * 5]
        assert len(set(lst)) == 1, "Hash value conflict!!!"

    @pytest.mark.parametrize(
        "fmt",
        ["ASE", "PyGData", "IGraph", "RustworkX", "NetworkX"],
    )
    def test_convert(self, system: System, fmt: str) -> None:
        obj = system
        print("-" * 64)
        _obj = obj.convert_to(fmt.lower())  # type: ignore
        if fmt.lower() == "ase":
            assert isinstance(_obj, Atoms), "ASE object expected!!!"
            print(_obj.info.keys())
        new_obj = obj.convert_from(_obj, fmt.lower())
        _equal_test(new_obj, obj)
        print(f"Convert to/from {fmt} OK!!!")

    @pytest.mark.parametrize("fmt", ["json", "pkl", "npz"])
    def test_io(self, system: System, fmt: str) -> None:
        obj = system
        print("-" * 64)
        with TemporaryDirectory() as path:
            fname = obj.write(Path(path) / f"system.{fmt}")
            new_obj = System.read(fname=fname)
        _equal_test(new_obj, obj)
        print(f"IO write/read {fmt} OK!!!")

    def test_getitem(self, system: System) -> None:
        print(repr(system.get_induced_subgraph([0, 1, 2, 3, 4])))

    # def test_update_geometry(self, system: System) -> None:
    #     new_g = np.asarray(system.positions, copy=True) + 1
    #     system.replace_geometry(new_geometry=new_g, isfix=[2, 3])

    def test_get_weisfeiler_lehman_hash(self, system: System) -> None:
        print(system.get_weisfeiler_lehman_hashes())

    # def test_print_property_is_cached_or_not(self, system: System) -> None:
    #     for k in sorted(
    #         k
    #         for k in (
    #             set(dir(System))
    #             - set(System.__pydantic_fields__)
    #             - {"iscore", "ncore", "isfix", "nfix"}
    #             - {"islastmoved", "isfirstmoved", "nmoved"}
    #         )
    #         if (
    #             not k.startswith("_")
    #             and not k.startswith("model_")
    #             and not callable(getattr(System, k))
    #         )
    #     ):
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             v1, v2 = getattr(system, k), getattr(system, k)
    #         if callable(v1):
    #             continue
    #         print(f"{k:<35s}: {str(id(v1) == id(v2)):5s} {id(v1)}={id(v2)}.")

    @pytest.mark.parametrize(
        "k",
        sorted(
            k
            for k in (
                set(dir(System))
                - set(System.__pydantic_fields__)
                - {"iscore", "ncore", "isfix", "nfix"}
                - {"islastmoved", "isfirstmoved", "nmoved"}
            )
            if (
                not k.startswith("_")
                and not k.startswith("model_")
                and not callable(getattr(System, k))
            )
        ),
    )
    def test_property_is_cached(self, system: System, k: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v1, v2 = getattr(system, k), getattr(system, k)
            assert id(v1) == id(v2), f"Hash of property changed: {k}!!!"

    @pytest.mark.parametrize("algo", ["vf2", "lad"])
    def test_match_cluster(self, system: System, algo: str) -> None:
        if system.nbonds == 0:
            return
        clst = Cluster.select_by_hop(
            system,
            system.get_hop_distance(0),  # type: ignore
        )
        matching = System.match(
            pattern=clst,  # type: ignore
            pattern4match=system,
            algorithm=algo,  # type: ignore
            return_match_target=True,
        )
        assert isinstance(matching, np.ndarray)
        assert matching.shape == (48, len(system))
        matching0 = np.asarray(
            [
                np.vectorize(lambda x: np.argwhere(matched_indxs == x).item())(
                    np.arange(len(clst))
                )
                for matched_indxs in matching
            ]
        )
        matching1 = System.match(
            pattern=clst,  # type: ignore
            pattern4match=system,
            algorithm=algo,  # type: ignore
            return_match_target=False,
        )
        assert isinstance(matching1, np.ndarray)
        np.testing.assert_array_equal(matching0, matching1)


class Test_PyArrowCompability:
    @staticmethod
    def get_all_item_classes() -> dict[str, type[Graph]]:
        return {
            "Gas": Gas,
            "Graph": Graph,
            "System": System,
            "Cluster": Cluster,
        }

    @pytest.mark.parametrize("cls_name", ["Gas", "Graph", "System", "Cluster"])
    def test_XxxItem_pyarrow_compability(self, cls_name: str) -> None:
        cls: type[Graph] = self.get_all_item_classes()[cls_name]
        print(cls.get_pyarrow_schema(), "-" * 32, sep="\n")

    @pytest.mark.parametrize("cls_name", ["Gas", "Graph", "System", "Cluster"])
    def test_Xxx_as_PyArrow_Table(self, system: System, cls_name: str) -> None:
        cls: type[Graph] = self.get_all_item_classes()[cls_name]
        if cls is Cluster:
            obj = Cluster.select_by_hop(
                system,
                system.get_hop_distance(0),  # type: ignore
                max_moved_hop=0,
                env_hop=1,
            )
        elif cls is Graph:
            obj = Graph.from_ase(system.to_ase())
        elif cls is System:
            obj = system
        elif cls is Gas:
            obj = Gas.from_molecule("CO", pressure=101325)
        else:
            raise ValueError(f"Unknown class: {cls}")

        print(
            pa.Table.from_pylist(
                [
                    obj.to_dict(
                        numpy_ndarray_compatible=False,
                        exclude_none=True,
                    )
                ]
                * 5,
                schema=cls.get_pyarrow_schema(),
            )
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-s"])
