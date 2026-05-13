import warnings
from typing import Literal

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii as COV_R
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ..utils.parser import DictConfig, hydra_parse
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


# TODO: add more parameters for `bond_list`` function
# def bond_list(
#     quantities: str,
#     a: Atoms,
#     bothways: bool = True,
#     indices: ArrayLike | None = None,
#     method: str | Literal["raw", "pymatgen"] = "raw",
#     cfg: DictConfig = DictConfig({}),
# ) -> np.ndarray:


def bond_list(
    atoms: Atoms,
    method: str | Literal["raw", "pymatgen"] = "raw",
    cfg: DictConfig = DictConfig({}),
    **kwargs,
) -> np.ndarray:
    """Compute bond list (list of bonded atom index pairs) for an Atoms object.

    Args:
        atoms: ASE Atoms object.
        method: "raw" (covalent radii) or "pymatgen" (nearest neighbor strategy).
        cfg: Hydration config dict for pymatgen strategy.
        **kwargs: Additional arguments passed to neighbor list.

    Returns:
        Nx2 array of bonded atom index pairs.
    """
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
        result = np.column_stack([i[mask], j[mask]])
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
        result = np.asarray(edges, dtype=int)
    else:
        raise ValueError(
            f"The method of `{method}` is not supported."  #
            "Please use `raw` or `pymatgen`."
        )

    assert result.ndim == 2 and result.shape[1] == 2, (
        f"The result must be Nx2 array. But {result.shape} got."
    )
    return result
