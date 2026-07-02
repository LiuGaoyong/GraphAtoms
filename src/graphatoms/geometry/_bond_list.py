import warnings
from typing import Literal

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii as COV_R
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import sparse as sp

from graphatoms.utils.parser import DictConfig, hydra_parse
from graphatoms.utils.rdutils import get_adjacency_by_rdkit

from ._neighbor_list import neighbor_list

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
        from pymatgen.analysis.local_env import NearNeighbors  # type: ignore
except ImportError:
    # moved to the pymatgen.core.local_env. This stub will be removed v2027.1
    from pymatgen.core.graphs import MoleculeGraph, StructureGraph
    from pymatgen.core.local_env import NearNeighbors  # type: ignore

__all__ = ["bond_list"]


def bond_list(
    atoms: Atoms,
    method: str | Literal["raw", "pymatgen"] = "raw",
    cfg: DictConfig = DictConfig({}),
    infer_order: bool = False,
) -> sp.spmatrix | sp.sparray:
    """Compute bond list (list of bonded atom index pairs) for an Atoms object.

    Args:
        atoms: ASE Atoms object.
        method: "raw" (covalent radii) or "pymatgen" (nearest neighbor strategy).
        cfg: Hydration config dict for pymatgen strategy.
        infer_order: whether infer bond order.

    Returns:
        Nx2 array of bonded atom index pairs.
    """  # noqa: E501
    is_nonperiodic = np.sum(atoms.cell.array.any(1) & atoms.pbc) == 0
    if len(atoms) == 0:
        return np.empty(shape=(0, 2), dtype=int)
    elif len(atoms) == 1 and is_nonperiodic:
        return np.empty(shape=(0, 2), dtype=int)

    if method == "raw":
        plus_factor = cfg.get("plus_factor", 0.3)
        multiply_factor = cfg.get("multiply_factor", 1.0)
        max_d: float = 2 * np.max(COV_R[atoms.numbers]) + 5
        i, j, d = neighbor_list("ijd", atoms, max_d, False, bothways=False)
        deq = COV_R[atoms.numbers[i]] + COV_R[atoms.numbers[j]]
        deq = deq * multiply_factor + plus_factor
        mask = np.logical_and(1e-3 < d, d < deq)
        edges = np.column_stack([i[mask], j[mask]])
    elif method == "pymatgen":
        nn = str(cfg.get("_target_", "")).split(".")[-1]
        if (
            is_nonperiodic
            and nn
            in (
                "VoronoiNN",
                "IsayevNN",
                "CrystalNN",
                "MinimumVIRENN",
                "BrunnerNNReal",
                "BrunnerNNRelative",
                "BrunnerNNReciprocal",
            )
        ) or (
            not is_nonperiodic
            and nn
            in (
                "CovalentBondNN",
                "OpenBabelNN",
            )
        ):
            raise RuntimeError(
                f"The pymatgen of `{nn}` is not supported for "
                f"{'non' if is_nonperiodic else ''}periodic system."
            )
        if nn == "CutOffDictNN":
            raise RuntimeError(
                "The pymatgen.core.local_env.CutOffDictNN is not supported."
            )

        obj: Molecule | Structure = AseAtomsAdaptor.get_structure(
            atoms=atoms,  # type: ignore
            cls=(Molecule if is_nonperiodic else Structure),
        )
        Graph = MoleculeGraph if is_nonperiodic else StructureGraph
        strategy: NearNeighbors = hydra_parse(cfg, NearNeighbors)
        if hasattr(Graph, "from_local_env_strategy"):
            g = Graph.from_local_env_strategy(obj, strategy)
        else:
            g = Graph.with_local_env_strategy(obj, strategy)
        assert isinstance(g.graph, nx.graph.Graph)
        edges = list(g.graph.edges(data=False))
        edges = np.asarray(edges, dtype=int)
    else:
        raise ValueError(
            f"The method of `{method}` is not supported."  #
            "Please use `raw` or `pymatgen`."
        )
    assert edges.ndim == 2 and edges.shape[1] == 2, (  # type: ignore
        f"The edges must be Nx2 array. But {edges.shape} got."
    )
    matrix = sp.csr_matrix(
        (np.ones_like(edges[:, 0]), edges.T),
        shape=(len(atoms), len(atoms)),
        dtype=bool,
    )
    matrix = sp.csr_matrix(matrix, dtype=int)
    if infer_order:
        # TODO: use rdkit to get bond order !!!
        edges = sp.coo_array(matrix).coords
        raise NotImplementedError
        get_adjacency_by_rdkit(
            numbers=np.array(
                [
                    0
                    if z not in (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 35, 53)
                    else z
                    for z in atoms.numbers
                ]
            ),
            geometry=atoms.positions,
            source=edges[0],
            target=edges[1],
            infer_order=False,
        )

    return matrix


def test_bond_order():
    from ase.build import molecule
    from ase.cluster import Octahedron

    atoms = molecule("CH3CH2OH")
    atoms = Octahedron("Cu", 3, 1)
    m = bond_list(atoms=atoms)

    m = sp.coo_array(m)
    edges, data = m.coords, m.data
    print(edges[0], edges[1], data)
    m2 = get_adjacency_by_rdkit(
        numbers=np.array(
            [
                1
                if z not in (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 35, 53)
                else z
                for z in atoms.numbers
            ]
        ),
        geometry=atoms.positions,
        source=edges[0],
        target=edges[1],
        order=data,
        infer_order=True,
        charge=-59,
    )
    print(m2)
