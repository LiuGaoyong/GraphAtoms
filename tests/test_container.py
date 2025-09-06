# ruff: noqa: D100, D101, D102, D103, D104
from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from ase.build import molecule
from ase.cluster import Octahedron

from GraphAtoms.common.abc import BaseModel
from GraphAtoms.containner import AtomicContainner, Cluster, Gas, System
from GraphAtoms.containner._graph import GRAPH_KEY, GraphContainner

THIS_DIR = Path(__file__).parent


def pytest_AtomicContainner() -> None:
    pprint(AtomicContainner.model_json_schema())
    atoms = molecule("H2O")
    obj = AtomicContainner.from_ase(atoms)
    print(obj, obj.box)
    print(repr(obj))
    new_atoms = obj.as_ase()
    print(new_atoms)
    new_obj = AtomicContainner.from_ase(new_atoms)
    print(repr(new_obj), "\n", repr(obj))
    assert new_obj == obj


@pytest.mark.parametrize("system", [System.from_ase(Octahedron("Cu", 8))])
class Test_Container:
    def test_len(self, system: System) -> None:
        assert len(system) == 344

    def test_repr(self, system: System) -> None:
        print(str(system), repr(system), sep="\n")

    def test_eq(self, system: System) -> None:
        assert system.__eq__(system), "System equality test fail!!!"

    def test_hash(self, system) -> None:
        lst = [hash(i) for i in [system] * 5]
        assert len(set(lst)) == 1, "Hash value conflict!!!"

    @pytest.mark.parametrize(
        "mode",
        [
            "ASE",
            "PyGData",
            "IGraph",
            "RustworkX",
            "NetworkX",
            # "NetworKit",
        ],
    )
    def test_convert(self, system: System, mode: str) -> None:
        obj = system
        print("-" * 64)
        _obj = obj.convert_as(mode.lower())  # type: ignore
        new_obj = obj.convert_from(
            _obj,
            mode.lower(),  # type: ignore
            infer_conn=False,
            infer_order=False,
        )
        assert new_obj == obj, f"\nobj={repr(obj)}\nnew_obj={repr(new_obj)}"
        print(f"Convert to/from {mode} OK!!!")

    # "npz": P1: cannot for str; P2: not for nest dict ...
    @pytest.mark.parametrize("fmt", ["json", "pkl"])
    def test_io(self, system: System, fmt: str) -> None:
        obj = system
        print("-" * 64)
        with TemporaryDirectory() as path:
            fname = obj.write(Path(path) / f"system.{fmt}")
            new_obj = System.read(fname=fname)
        assert new_obj == obj, f"\nobj={repr(obj)}\nnew_obj={repr(new_obj)}"
        print(f"IO write/read {fmt} OK!!!")

    def test_getitem(self, system: System) -> None:
        print(repr(system.get_induced_subgraph([0, 1, 2, 3, 4])))

    # def test_update_geometry(self, system: System) -> None:
    #     new_g = np.asarray(system.positions, copy=True) + 1
    #     system.replace_geometry(new_geometry=new_g, isfix=[2, 3])

    def test_get_weisfeiler_lehman_hash(self, system: System) -> None:
        print(system.get_weisfeiler_lehman_hashes())

    def test_print_property_is_cached_or_not(self, system: System) -> None:
        for k in sorted(
            set(dir(system))
            - set(dir(BaseModel))
            - set(system.__pydantic_fields__)
            - {"THERMO"}
        ):
            if (
                not k.startswith("_")
                and (k not in ["isfix", "isfirstmoved"])
                and (k not in ["iscore", "islastmoved"])
                and not callable(getattr(system, k))
            ):
                v1, v2 = getattr(system, k), getattr(system, k)
                print(
                    f"{k:<35s}: {str(id(v1) == id(v2)):5s} {id(v1)}={id(v2)}."
                )

    @pytest.mark.parametrize(
        "k",
        sorted(
            k
            for k in set(dir(System))
            - set(dir(BaseModel))
            - set(System.__pydantic_fields__)
            - {"isfix", "iscore", "islastmoved", "isfirstmoved"}
            - {"DF_ATOMS", "DF_BONDS", "THERMO", "THERMO_ATOMS", "Z"}
            - {"connected_components_number", "natoms", "nbonds", "symbols"}
            if not k.startswith("_") and not callable(getattr(System, k))
        ),
    )
    def test_property_is_cached(self, system: System, k: str) -> None:
        v1, v2 = getattr(system, k), getattr(system, k)
        assert id(v1) == id(v2), f"Hash of property changed: {k}!!!"

    @pytest.mark.parametrize("algo", ["vf2", "lad"])
    def test_match_cluster(self, system: System, algo: str) -> None:
        if system.nbonds == 0:
            return
        clst = Cluster.select_by_hop(system, system.get_hop_distance(0))
        matching = System.match(
            pattern=clst,
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
            pattern=clst,
            pattern4match=system,
            algorithm=algo,  # type: ignore
            return_match_target=False,
        )
        assert isinstance(matching1, np.ndarray)
        np.testing.assert_array_equal(matching0, matching1)


def test_graph_basic() -> None:
    """Test the system with bonds.

    Run 100 time:
        Timing:                        incl.     excl.
    -----------------------------------------------------
    Infer Bond Order:              0.152     0.152  21.1% |-------|
    Infer Connectivity:            0.106     0.106  14.7% |-----|
    OurContainer => PyG Data:      0.007     0.007   1.0% |
    ase.Atoms => OurContainer:     0.004     0.004   0.6% |
    Other:                         0.451     0.451  62.6% |------------...|
    -----------------------------------------------------
    Total:                                   0.720 100.0%
    """
    obj_smp = GraphContainner.from_ase(molecule("CH4"))
    obj_conn = GraphContainner.from_ase(
        molecule("CH4"),
        infer_conn=True,
        infer_order=False,
    )
    obj_order = GraphContainner.from_ase(
        molecule("CH4"),
        infer_conn=True,
        infer_order=True,
    )
    for obj in (obj_smp, obj_conn, obj_order):
        print("#" * 32)
        print(obj, obj.box)
        print(repr(obj))
        new_atoms = obj.as_ase()
        print(new_atoms)
        print(obj.MATRIX)

    print("*" * 32, "Test PyGData from obj_order")
    pygdata = obj_order.as_pygdata()
    print(pygdata, pygdata.num_edges, pygdata.num_nodes)
    assert pygdata.num_nodes == obj_order.natoms
    assert pygdata.num_edges == obj_order.nbonds, obj_order.MATRIX.toarray()
    for k in pygdata.node_attrs():
        v = pygdata[k]
        print("NODE", k, type(v))
    for k in pygdata.edge_attrs():
        v = pygdata[k]
        print("EDGE", k, type(v))

    print("*" * 32, "Test PyGData equality from obj_order")
    new_obj_order = GraphContainner.from_pygdata(pygdata)
    print(repr(new_obj_order), "\n", repr(obj_order))
    if new_obj_order != obj_order:
        for k in set(GRAPH_KEY._DICT):
            print(k, getattr(obj_order, k), getattr(new_obj_order, k))
        raise ValueError  # Shouldn't raise ValueError


@pytest.mark.parametrize("obj", [GraphContainner.from_ase(Octahedron("Au", 8))])
@pytest.mark.parametrize(
    "mode",
    [
        "ASE",
        "PyGData",
        "IGraph",
        "RustworkX",
        "NetworkX",
        # "NetworKit",
    ],
)
def test_graph_convert(obj: GraphContainner, mode: str) -> None:
    print("-" * 64)
    _obj = obj.convert_as(mode.lower())  # type: ignore
    new_obj = obj.convert_from(
        _obj,
        mode.lower(),  # type: ignore
        infer_conn=False,
        infer_order=False,
    )
    assert new_obj == obj, f"\nobj={repr(obj)}\nnew_obj={repr(new_obj)}"
    print(mode, "OK!!!")


def test_gas_thermo() -> None:
    gas = Gas.from_molecule(
        "CO",
        energy=0.138,  # GFNFF by XTB@6.7.1
        frequencies=[0, 0, 0, 12.6, 12.6, 2206.3],
        pressure=101325.0,
    )
    print(gas.get_free_energy(200, verbose=True))


def test_select_cluster() -> None:
    obj = System.from_ase(Octahedron("Au", 8))
    sub = Cluster.select_by_hop(obj, obj.get_hop_distance(0))
    print(
        "-" * 32,
        Cluster.model_json_schema(),
        "-" * 32,
        sub,
        repr(sub),
        "-" * 32,
        sep="\n",
    )
    sub2 = Cluster.select_by_distance(obj, np.asarray([0]))
    print(
        sub,
        sub2,
        "-" * 32,
        sub.move_fix_tag,
        sub2.move_fix_tag,
        sep="\n",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-s", "--maxfail=1"])
