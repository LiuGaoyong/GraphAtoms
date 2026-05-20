"""Distance calculation factory for atomic configurations."""

import numpy as np
from ase.cell import Cell
from ase.data import covalent_radii as COV_R
from scipy import sparse as sp

from ._distance_pairs import distance_pairs
from .sample import fibonacci_lattice as inverse_3d_sphere_surface_sampling


def check_atom_is_inner(  # noqa: D417
    index: int,
    numbers: np.ndarray,
    geometry: np.ndarray,
    adjacency_array: sp.csr_matrix | sp.spmatrix | sp.sparray,
    cell: Cell | None = None,
) -> bool:
    """Check whether the given atom is inner by distance.

    Args:
        index (int): the index of the atom.
        numbers (np.ndarray): the atomic number.
        geometry (np.ndarray): the atomic geometry.
        adjacency_matrix (sp.csr_matrix): the adjacency matrix.
        cell (Cell, optional): the periodic cell. Defaults to None.

    Returns:
        bool: whether the given atom is inner.
    """
    geometry = np.asarray(geometry, dtype=float)
    # get neighbors of the index atom
    nbr = sp.coo_array(adjacency_array)
    assert nbr.shape == (1, len(numbers)), nbr.shape
    idxs_neighbor = np.setdiff1d(nbr.coords[1], [index])
    if len(idxs_neighbor) == 0:
        return False
    else:
        # create sphere surface points sampling
        mesh = inverse_3d_sphere_surface_sampling(1000)
        mesh = p1 = geometry[index] + mesh * COV_R[numbers[index]]
        # get distance between mesh and neighbor
        p2 = np.atleast_2d(geometry[idxs_neighbor])
        src, tgt, d = distance_pairs(
            quantities="ijd",
            p1=p1,
            p2=p2,
            cutoff=(
                np.max(COV_R[numbers[idxs_neighbor]])
                + float(COV_R[numbers[index]])
            ),
            cell=cell,
        )
        # get mesh mask array
        mask = d >= COV_R[tgt]
        d, src, tgt = d[mask], src[mask], tgt[mask]
        mask = sp.csr_matrix(
            (np.ones_like(d, dtype=bool), (src, tgt)),
            shape=(p1.shape[0], p2.shape[0]),
            dtype=bool,
        )
        assert mask.shape == (len(mesh), len(idxs_neighbor))
        mask_array = mask.sum(axis=1, dtype=bool)
        # set is_inner
        return bool(np.sum(mask_array) == len(mesh))


#######################################################################
#                                   Test
#######################################################################
def _test_inner(atoms) -> None:
    from ._bond_list import Atoms, bond_list

    assert isinstance(atoms, Atoms)
    m: sp.csr_matrix = bond_list(atoms)
    print(len(atoms))
    for i in range(len(atoms)):
        is_inner = check_atom_is_inner(
            index=i,
            numbers=atoms.numbers,
            geometry=atoms.positions,
            adjacency_array=m[i],
        )
        print(f"{i:3d}: {is_inner}")
        assert is_inner is bool(i == 9)


def test_inner() -> None:  # noqa: D103
    from ase.cluster import Octahedron

    _test_inner(Octahedron("Cu", 3, 1))
