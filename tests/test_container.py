# ruff: noqa: D100, D101, D102, D103, D104
import warnings
from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pytest
from ase import Atoms
from ase.build import molecule
from ase.thermochemistry import HarmonicThermo

from GraphAtoms.common import BaseModel
from GraphAtoms.containner import AtomsWithBoxEng as AtomicContainner
from GraphAtoms.containner import Cluster, Gas, Graph, System
from GraphAtoms.containner import Graph as GraphContainner
from GraphAtoms.containner._aOther import OtherMixin


class Test_ContainerBasic:
    def test_AtomicContainner(self) -> None:
        pprint(AtomicContainner.model_json_schema())
        atoms = molecule("H2O")
        obj = AtomicContainner.from_ase(atoms)
        print(obj.ase_cell)
        print(obj, obj.ase_cell)
        print(repr(obj))
        new_atoms: Atoms = obj.to_ase()
        print(new_atoms)
        new_obj = AtomicContainner.from_ase(new_atoms)
        print(repr(new_obj), "\n", repr(obj))
        assert new_obj == obj

    def test_graph_basic(self) -> None:
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
            print(obj, obj.ase_cell)
            print(repr(obj))
            new_atoms = obj.to_ase()
            print(new_atoms)
            print(obj.MATRIX)

        print("*" * 32, "Test PyGData from obj_order")
        pygdata = obj_order.to_pygdata()
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
            for k in obj_order.__pydantic_fields__:
                print(k, getattr(obj_order, k), getattr(new_obj_order, k))
            raise ValueError  # Shouldn't raise ValueError


class Test_Container:
    def test_select_cluster(self, system: System) -> None:
        sub = Cluster.select_by_hop(system, system.get_hop_distance(0))
        print(
            "-" * 32,
            Cluster.model_json_schema(),
            "-" * 32,
            sub,
            repr(sub),
            "-" * 32,
            sep="\n",
        )
        sub2 = Cluster.select_by_distance(system, np.asarray([0]))
        print(
            sub,
            sub2,
            "-" * 32,
            sub.move_fix_tag,
            sub2.move_fix_tag,
            sep="\n",
        )

    def test_len(self, system: System) -> None:
        assert len(system) == 344

    def test_repr(self, system: System) -> None:
        print(str(system), repr(system), sep="\n")

    def test_eq(self, system: System) -> None:
        assert system.__eq__(system), "System equality test fail!!!"

    def test_hash(self, system: System) -> None:
        print(system.__hash__)
        lst = [hash(i) for i in [system] * 5]
        assert len(set(lst)) == 1, "Hash value conflict!!!"

    @pytest.mark.parametrize(
        "mode",
        ["ASE", "PyGData", "IGraph", "RustworkX", "NetworkX"],
    )
    def test_convert(self, system: System, mode: str) -> None:
        obj = system
        print("-" * 64)
        _obj = obj.convert_to(mode.lower())  # type: ignore
        if mode.lower() == "ase":
            assert isinstance(_obj, Atoms), "ASE object expected!!!"
            print(_obj.info.keys())
        new_obj = obj.convert_from(
            _obj,
            mode.lower(),  # type: ignore
            infer_conn=False,
            infer_order=False,
        )
        assert new_obj == obj, f"\nobj={repr(obj)}\nnew_obj={repr(new_obj)}"
        print(f"Convert to/from {mode} OK!!!")

    # "npz": P1: cannot for str; P2: not for nest dict ...
    @pytest.mark.parametrize("fmt", ["json", "pkl", "npz"])
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
                and (k not in ["ncore", "nfix", "nmoved"])
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
            - {"a", "b", "c", "alpha", "beta", "gamma"}
            - {"ncore", "nfix", "nmoved", "iscore", "isfix"}
            - {"islastmoved", "isfirstmoved", "vib_energies"}
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


class Test_Thermo:
    def test_gas_thermo(self) -> None:
        from ase.thermochemistry import IdealGasThermo
        from ase.units import invcm

        e, T, p = 0.138, 200, 101325.0
        gas = Gas.from_molecule(
            "CO",
            energy=e,  # GFNFF by XTB@6.7.1
            frequencies=[0, 0, 0, 12.6, 12.6, 2206.3],
            pressure=p,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            thermo = IdealGasThermo(
                vib_energies=gas.frequencies * invcm,  # type: ignore
                geometry="linear",
                potentialenergy=0.138,
                atoms=gas.to_ase(),
                spin=0,
                symmetrynumber=1,
            )
        print(thermo.vib_energies)
        print(gas.vib_energies)

        v0 = gas.get_vibrational_energy_contribution(T)
        v00 = thermo._vibrational_energy_contribution(T)
        assert np.isclose(v0, v00)

        v0 = gas.get_enthalpy(T)
        v00 = thermo.get_enthalpy(T)
        assert np.isclose(v0, v00)

        v1 = gas.get_entropy(T)
        v11 = thermo.get_entropy(T, p)
        assert np.isclose(v1, v11)

        v1 = gas.get_free_energy(T)
        v11 = thermo.get_gibbs_energy(T, p)
        assert np.isclose(v1, v11)

    def test_harmonic_thermo(self) -> None:
        e, T = 0.138, 200
        gas = OtherMixin(frequencies=[0, 0, 0, 12.6, 12.6, 2206.3], energy=e)  # type: ignore
        thermo = HarmonicThermo(
            vib_energies=gas.vib_energies, potentialenergy=e
        )  # type: ignore
        print(thermo.vib_energies)
        print(gas.vib_energies)

        v0 = gas.get_vibrational_energy_contribution(T)
        v00 = thermo._vibrational_energy_contribution(T)
        assert np.isclose(v0, v00)

        v0 = gas.get_enthalpy(T)
        v00 = thermo.get_internal_energy(T)
        assert np.isclose(v0, v00)

        v1 = gas.get_entropy(T)
        v11 = thermo.get_entropy(T)
        assert np.isclose(v1, v11)

        v1 = gas.get_free_energy(T)
        v11 = thermo.get_helmholtz_energy(T)
        assert np.isclose(v1, v11)


class Test_PyArrowCompability:
    @staticmethod
    def get_all_item_classes() -> list[type[Graph]]:
        return [Gas, System, Cluster, Graph]

    @pytest.mark.parametrize("cls_id", sorted(range(4)))
    def test_XxxItem_pyarrow_compability(self, cls_id: int) -> None:
        cls: type[Graph] = self.get_all_item_classes()[cls_id]
        print(cls.get_pyarrow_schema(), "-" * 32, sep="\n")

    @pytest.mark.parametrize("cls_id", sorted(range(4)))
    def test_Xxx_as_PyArrow_Table(self, system: System, cls_id: int) -> None:
        cls: type[Graph] = self.get_all_item_classes()[cls_id]
        if cls is Cluster:
            obj = Cluster.select_by_hop(
                system,
                system.get_hop_distance(0),
            )
        elif cls is Graph:
            obj = Graph.from_ase(system.to_ase())
        elif cls is System:
            obj = system
        elif cls is Gas:
            obj = Gas.from_molecule("CO")
        else:
            raise ValueError(f"Unknown class: {cls}")

        print(
            pa.Table.from_pylist(
                [
                    obj.model_dump(
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
