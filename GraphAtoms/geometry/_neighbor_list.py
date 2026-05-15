import warnings

import numpy as np
from ase.atoms import Atoms
from numpy.typing import ArrayLike

try:
    NEIGHBORLIST_BACKEND = "vesin"
    from vesin import NeighborList as NeighborListVesin
    from vesin import ase_neighbor_list as __all_neighbor_list
except ImportError:
    warnings.warn(
        "Please install vesin/matscipy to speed up "  #
        "the calculation of neighbor list.",
        category=UserWarning,
    )
    NEIGHBORLIST_BACKEND = "ase"
    from ase.neighborlist import neighbor_list as __all_neighbor_list
from ase.neighborlist import build_neighbor_list as __build_neighbor_list_ase

__all__ = ["all_neighbor_list", "neighbor_list"]


def all_neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
    self_interaction: bool = False,
    max_nbins: int = int(1e6),
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

    Returns:
    -------
    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.

    Examples:
    --------
    >>> import numpy as np
    >>> from ase.build import bulk, molecule

    1. Coordination counting

    >>> atoms = molecule('isobutane')
    >>> i = neighbor_list('i', atoms, 1.85)
    >>> coord = np.bincount(i, minlength=len(atoms))

    2. Coordination counting with different cutoffs for each pair of species

    >>> cutoff = {("H", "H"): 1.1, ('C', 'H'): 1.3, ('C', 'C'): 1.85}
    >>> i = neighbor_list('i', atoms, cutoff)
    >>> coord = np.bincount(i, minlength=len(atoms))

    3. Pair distribution function

    >>> atoms = bulk('Cu', cubic=True) * 3
    >>> atoms.rattle(0.5, rng=np.random.default_rng(42))
    >>> cutoff = 5.0
    >>> d = neighbor_list('d', atoms, cutoff)
    >>> hist, bin_edges = np.histogram(d, bins=100, range=(0.0, cutoff))
    >>> hist = hist / len(atoms)  # per atom
    >>> rho_mean = len(atoms) / atoms.cell.volume
    >>> dv = 4.0 * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3) / 3.0
    >>> rho = hist / dv
    >>> pdf = rho / rho_mean

    4. Forces of a pair potential

    >>> natoms = len(atoms)
    >>> i, j, d, D = neighbor_list('ijdD', atoms, 5.0)
    >>> # Lennard-Jones potential
    >>> eps = 1.0
    >>> sgm = 1.0
    >>> epairs = 4.0 * eps * ((sgm / d) ** 12 - (sgm / d) ** 6)
    >>> energy = 0.5 * epairs.sum()  # correct double-counting
    >>> dd = -4.0 * eps * (12 * (sgm / d) ** 13 - 6 * (sgm / d) ** 7) / sgm
    >>> dd = (dd * (D.T / d)).T
    >>> fx = -1.0 * np.bincount(i, weights=dd[:, 0], minlength=natoms)
    >>> fy = -1.0 * np.bincount(i, weights=dd[:, 1], minlength=natoms)
    >>> fz = -1.0 * np.bincount(i, weights=dd[:, 2], minlength=natoms)

    5. Force-constant matrix of a pair potential

    >>> i, j, d, D = neighbor_list('ijdD', atoms, 5.0)
    >>> epairs = 1.0 / d  # Coulomb potential
    >>> forces = (D.T / d**3).T  # (npairs, 3)
    >>> # second derivative
    >>> d2 = 3.0 * D[:, :, None] * D[:, None, :] / d[:, None, None] ** 5
    >>> for k in range(3):
    ...     d2[:, k, k] -= 1.0 / d**3
    >>> # force-constant matrix
    >>> fc = np.zeros((natoms, 3, natoms, 3))
    >>> for ia in range(natoms):
    ...     for ja in range(natoms):
    ...         fc[ia, :, ja, :] -= d2[(i == ia) & (j == ja), :, :].sum(axis=0)
    >>> for ia in range(natoms):
    ...     fc[ia, :, ia, :] -= fc[ia].sum(axis=1)

    """  # noqa: E501
    assert isinstance(a, Atoms), (
        f"The `a` must be ase.Atoms, but"  #
        f" {a.__class__.__name__} got"
    )
    assert all(i in "ijdDS" for i in quantities), (
        "The `quantities` must be an elements of "  #
        f"'ijdDS', but got {quantities},"
    )
    if len(a) == 1:
        return __neighbor_list_ase(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            self_interaction=self_interaction,
        )

    return __all_neighbor_list(
        quantities=quantities,
        a=a,
        cutoff=float(cutoff),
        self_interaction=self_interaction,
        max_nbins=int(max_nbins),
    )  # type: ignore


def neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
    self_interaction: bool = False,
    max_nbins: int = int(1e6),
    bothways: bool = True,
    indices: ArrayLike | None = None,
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
    # Check parameters finished.

    if indices is None and len(a) != 1:
        if bothways:
            return __all_neighbor_list(
                quantities=quantities,
                a=a,
                cutoff=float(cutoff),
                self_interaction=self_interaction,
                max_nbins=int(max_nbins),
            )  # type: ignore
        elif NEIGHBORLIST_BACKEND == "vesin":
            calculator = NeighborListVesin(  # type: ignore
                cutoff=float(cutoff),
                full_list=False,
                sorted=True,
            )
            return calculator.compute(
                points=a.positions,
                box=a.cell[:],  # type: ignore
                periodic=a.pbc,
                quantities=quantities,
                copy=True,
            )
        elif NEIGHBORLIST_BACKEND == "ase":
            indices = np.arange(len(a))
        else:
            raise RuntimeError(
                "Unknown neighbor list "  #
                f"backend: {NEIGHBORLIST_BACKEND}"
            )
    return __neighbor_list_ase(
        quantities=quantities,
        a=a,
        cutoff=cutoff,
        indices=indices,
        self_interaction=self_interaction,
        bothways=bothways,
    )


def __neighbor_list_ase(
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


#######################################################################
#                                   Test
#######################################################################


def _test_neighbor_list_for_atoms(a: Atoms, c: float, idx=[0, 1]) -> None:
    print()
    print(a)
    print("=============================================================")
    print("Full neighbor list")
    print(*neighbor_list("ijd", a, c), sep="\n")
    print(*__neighbor_list_ase("ijd", a, c), sep="\n")
    print("=============================================================")
    print("Half neighbor list")
    print(*neighbor_list("ijd", a, c, bothways=False), sep="\n")
    print(*__neighbor_list_ase("ijd", a, c, bothways=False), sep="\n")
    print("=============================================================")
    print(f"Neighbor list with indices: {idx}")
    print(*neighbor_list("ijd", a, c, indices=idx), sep="\n")
