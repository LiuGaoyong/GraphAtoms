import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from omegaconf import DictConfig, OmegaConf

from GraphAtoms.geometry._bond_list import bond_list
from GraphAtoms.geometry._neighbor_list import _test_neighbor_list_for_atoms


@pytest.mark.parametrize(
    "a",
    [
        molecule("H2O"),
        bulk("Cu", "fcc"),
        fcc111("Cu", (1, 1, 2), vacuum=8, periodic=True),
    ],
)
def test_neghbor_list(a: Atoms) -> None:
    _test_neighbor_list_for_atoms(a, c=3, idx=[0])


@pytest.mark.parametrize(
    "method, name",
    [
        ("raw", ""),
        ("pymatgen", "JmolNN"),
        ("pymatgen", "MinimumDistanceNN"),
        ("pymatgen", "VoronoiNN"),
        ("pymatgen", "IsayevNN"),
        ("pymatgen", "OpenBabelNN"),
        ("pymatgen", "CovalentBondNN"),
        ("pymatgen", "MinimumOKeeffeNN"),
        ("pymatgen", "MinimumVIRENN"),
        ("pymatgen", "BrunnerNNReciprocal"),
        ("pymatgen", "BrunnerNNRelative"),
        ("pymatgen", "BrunnerNNReal"),
        ("pymatgen", "EconNN"),
    ],
)
def test_bond_list(method: str, name: str) -> None:
    if method == "pymatgen":
        cfg = {"_target_": f"pymatgen.core.local_env.{name:s}"}
    elif method == "raw":
        cfg = {"plus_factor": 0.3, "multiply_factor": 1.0}
    else:
        raise ValueError(method)
    print()
    pbc = bulk("Cu", "fcc", cubic=True)
    mol, pbc0 = molecule("H2O"), bulk("Cu")
    nn = str(cfg.get("_target_", "")).split(".")[-1]
    for atoms in (mol, pbc0, pbc):
        if not atoms.pbc.any():
            is_periodic = False
            if nn in (
                "VoronoiNN",
                "IsayevNN",
                "CrystalNN",
                "MinimumVIRENN",
                "BrunnerNNReal",
                "BrunnerNNRelative",
                "BrunnerNNReciprocal",
            ):
                continue
        else:
            is_periodic = True
            if nn in (
                "CovalentBondNN",
                "OpenBabelNN",
            ):
                continue
        print("====================================================")
        dct = {"method": method, "is_periodic": is_periodic, "cfg": cfg}
        print(OmegaConf.to_yaml(DictConfig(dct)))
        print(bond_list(atoms, method, DictConfig(cfg)))
        print("====================================================")
