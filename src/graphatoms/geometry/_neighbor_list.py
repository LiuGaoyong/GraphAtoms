from typing import Literal

import numpy as np
from ase.atoms import Atoms
from ase.neighborlist import build_neighbor_list as __build_neighbor_list_ase
from ase.neighborlist import neighbor_list as _ase_neighbor_list_3loop
from numpy.typing import ArrayLike

from ._distance_pairs import distance_pairs

__all__ = ["neighbor_list"]


#############################################################
# ASE
def _ase_neighbor_list_kdtree(
    quantities: str,
    a: Atoms,
    cutoff: float,
    indices: np.ndarray | None = None,
    self_interaction: bool = False,
    bothways: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, ...]:
    if indices is None:
        indices = np.arange(len(a))
    nl = __build_neighbor_list_ase(
        atoms=a,
        cutoffs=[float(cutoff / 2)] * len(a),
        self_interaction=self_interaction,
        bothways=bothways,
        sorted=sorted,
        skin=0,
    )
    i = np.array([], dtype=int)
    j = np.array([], dtype=int)
    S = np.zeros(shape=(0, 3), dtype=int)
    for idx in indices:
        jj, SS = nl.get_neighbors(idx)
        j = np.append(j, jj)
        S = np.vstack([S, SS])
        i = np.append(i, np.full(len(jj), idx))
    dct: dict[str, np.ndarray] = {
        "i": i,
        "j": j,
        "S": S,
    }
    if "d" in quantities.lower():
        v = a.positions[j] - a.positions[i]
        dct["D"] = D = v + S @ a.cell
        if "d" in quantities:
            dct["d"] = np.linalg.norm(D, axis=1)
    return tuple(dct[q] for q in quantities)


#############################################################
# matscipy
def _matscipy_neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
    self_interaction: bool = False,
    bothways: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, ...]:
    try:
        pass  # type: ignore
    except Exception:
        raise Exception(
            "Please install vesin to speed up "  #
            "the calculation of neighbor list.",
        )
    raise NotImplementedError


#############################################################
# VESIN
def _vesin_neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
    self_interaction: bool = False,
    max_nbins: int = 0,
) -> tuple[np.ndarray, ...]:
    try:
        from vesin import ase_neighbor_list
    except Exception:
        raise Exception(
            "Please install vesin to speed up "  #
            "the calculation of neighbor list.",
        )
    return tuple(
        ase_neighbor_list(
            quantities=quantities,
            a=a,
            cutoff=cutoff,
            self_interaction=self_interaction,
            max_nbins=max_nbins,
        )
    )


#############################################################
# interface
def neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
    self_interaction: bool = False,
    max_nbins: int = int(1e6),
    bothways: bool = True,
    indices: ArrayLike | None = None,
    backend: str
    | Literal[
        "sklearn", "pmg", "pymatgen", "ase_3loop", "ase_kdtree", "vesin"
    ] = "pmg",
) -> tuple[np.ndarray, ...]:
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters
    ----------
    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are:

        * 'i' : first atom index
        * 'j' : second atom index
        * 'd' : absolute distance
        * 'D' : distance vector
        * 'S' : shift vector (number of cell boundaries crossed by the bond
            between atom i and j). With the shift vector S, the
            distances D between atoms can be computed from:
            D = a.positions[j] - a.positions[i] + S @ a.cell
    a: :class:`ase.Atoms`
        Atomic configuration.
    cutoff: float or dict
        This is a global cutoff for all elements.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.
    indices: np.ndarray | None
        Indices of atoms to use in the neighbor list.
        If None, all atoms are.
    backend: str
        The backend for computation.

    Returns:
    -------
    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.
    """
    assert isinstance(a, Atoms), (
        f"The `a` must be ase.Atoms, but"  #
        f" {a.__class__.__name__} got"
    )
    assert all(i in "ijdDS" for i in quantities), (
        "The `quantities` must be an elements of "  #
        f"'ijdDS', but got {quantities},"
    )
    if len(a) <= 1 and a.cell.rank == 0:
        return tuple(np.array([]) for _ in quantities)

    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        indices = np.unique(indices.flatten())
        if len(indices) == 0:
            indices = None
        else:
            assert 0 <= indices.min() <= indices.max() < len(a), (
                "The indices must be in range [0, len(a))."
            )
            if len(indices) == len(a):
                indices = None
    if indices is not None:
        assert bothways is True, (
            "Only support `bothways=True` "  #
            "for `indices` not None."
        )

    msg = "The {} is not support for non-empty indices."
    if backend == "ase_3loop":
        assert indices is None, msg.format(backend)
        return _ase_neighbor_list_3loop(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            self_interaction=self_interaction,
            max_nbins=int(max_nbins),
        )  # type: ignore
    elif backend == "vesin":
        assert indices is None, msg.format(backend)
        return _vesin_neighbor_list(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            self_interaction=self_interaction,
            max_nbins=int(max_nbins),
        )  # type: ignore
    elif backend == "matscipy":
        raise NotImplementedError
        return _matscipy_neighbor_list(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            self_interaction=self_interaction,
        )
    elif backend == "ase_kdtree":
        return _ase_neighbor_list_kdtree(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            indices=indices,
            self_interaction=self_interaction,
            bothways=bothways,
        )
    elif backend in ("sklearn", "pmg", "pymatgen"):
        if indices is None:
            indices = np.arange(len(a))
        return distance_pairs(
            quantities=quantities,
            p1=a.positions,
            p2=a.positions[indices],
            cell=a.cell,
            cutoff=cutoff,
            backend=backend,
        )
    else:
        raise RuntimeError(f"Unknown neighbor list backend: {backend}")


#######################################################################
#                                   Test
#######################################################################


def _test_neighbor_list_for_atoms(a: Atoms, c: float, idx=[0, 1]) -> None:

    print()
    print(a)
    print("=============================================================")
    print("Full neighbor list")
    print(*_ase_neighbor_list_3loop("ijd", a, c), sep="\n")
    print(*neighbor_list("ijd", a, c), sep="\n")
    print("=============================================================")
    print("Half neighbor list")
    print(*neighbor_list("ijd", a, c, bothways=False), sep="\n")
    print(*_ase_neighbor_list_kdtree("ijd", a, c, bothways=False), sep="\n")
    print("=============================================================")
    print(f"Neighbor list with indices: {idx}")
    print(*neighbor_list("ijd", a, c, indices=idx), sep="\n")


def test_neighbor_list_for_atoms() -> None:
    from ase.build import fcc111

    a = fcc111(
        "Cu",
        [2, 2, 1],
        vacuum=10,
        orthogonal=True,
        periodic=True,
    )
    c = 3.0
    _test_neighbor_list_for_atoms(a, c)


if __name__ == "__main__":
    test_neighbor_list_for_atoms()
