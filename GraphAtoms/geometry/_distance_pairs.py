"""Calculate pairs."""

import itertools
from typing import Literal

import numpy as np
from ase.cell import Cell
from ase.geometry import minkowski_reduce, wrap_positions
from scipy import sparse as sp
from scipy.linalg import norm, pinv
from sklearn.neighbors import NearestNeighbors


#############################################################
# scikit-learn
def _sklearn_get_nn(X) -> NearestNeighbors:
    nn = NearestNeighbors()
    nn.fit(X=X, y=None)
    return nn


def _sklearn_get_sparse_matrix(
    nn: NearestNeighbors,
    points: np.ndarray,
    radius: float,
    return_distance: bool = False,
) -> sp.csr_matrix:
    """Get sparse matrix by distance.

    Args:
        nn (NearestNeighbors): The sklearn NearestNeighbors object.
        points (np.ndarray): The query point or points.
        radius (float): Radius of neighborhoods.
        return_distance (bool, optional): Defaults to False.

    Returns:
        sp.csr_matrix: The shape is (len(points), len(nn.X)).
    """
    return nn.radius_neighbors_graph(
        X=points,
        radius=radius,
        mode="distance" if return_distance else "connectivity",
    )


def _sklearn_get_distance_pairs(
    p1: np.ndarray,
    p2: np.ndarray | None = None,
    cell: np.ndarray | Cell | None = None,
    max_distance: float = float("inf"),
    return_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(cell, Cell):
        cell = Cell.new(cell=cell)
    p1 = np.asarray(p1, dtype=float)
    if p2 is None:
        p2 = p1
    else:
        p2 = np.asarray(p2, dtype=float)
    assert p1.ndim == p2.ndim == 2
    assert p1.shape[1] == p2.shape[1] == 3
    l1, l2 = p1.shape[0], p2.shape[0]
    if p1.shape[0] >= p2.shape[0]:
        swap = False
    else:
        swap, p1, p2 = True, p2, p1
    assert p1.shape[0] >= p2.shape[0]
    max_distance = float(max_distance)
    assert max_distance > 0

    distance = _sklearn_get_sparse_matrix(
        nn=_sklearn_get_nn(X=p1),
        points=p2,
        radius=max_distance,
        return_distance=bool(return_distance),
    )  # shape = (len(p2), len(p1))
    result: dict[tuple[int, int, int], sp.csr_matrix] = {
        (0, 0, 0): distance.T if swap else distance
    }
    if cell.rank != 0:
        cell, _ = minkowski_reduce(cell)
        assert isinstance(p2, np.ndarray)
        p2 = wrap_positions(p2, cell, eps=0)
        p1 = wrap_positions(p1, cell, eps=0)
        assert max_distance != float("inf")
        nn = _sklearn_get_nn(X=p1)
        cell_pinv_T = np.transpose(pinv(cell))
        n = [int(max_distance * norm(i)) + 1 for i in cell_pinv_T]
        nrange3 = [range(-i, i + 1) for i in n]
        for n1, n2, n3 in itertools.product(*nrange3):
            # if n1 <= 0 and (n2 < 0 or n2 == 0 and n3 < 0):
            #     continue
            shift_idxs = np.array([n1, n2, n3])
            distance = _sklearn_get_sparse_matrix(
                nn=nn,
                radius=max_distance,
                return_distance=bool(return_distance),
                points=(p2 - shift_idxs @ cell),  # type: ignore
            )
            result[(n1, n2, n3)] = distance.T if swap else distance

    shift, dist = np.empty((0, 3), int), np.array([], float)
    source, target = np.array([], int), np.array([], int)
    for shift_idxs, matrix in result.items():
        assert matrix.shape == (l2, l1)
        distance = sp.coo_matrix(matrix)
        if len(distance.data) != 0:
            source = np.append(source, distance.col)
            target = np.append(target, distance.row)
            if return_distance:
                dist = np.append(dist, distance.data)
            shift_idxs_lst = [tuple(shift_idxs)] * distance.nnz
            shift = np.vstack([shift, np.asarray(shift_idxs_lst, int)])
    assert shift.ndim == 2 and shift.shape[1] == 3
    assert source.max() < l1 and target.max() < l2, (
        f"{source.max()} {l1}, {target.max()} {l2}"
    )
    return source, target, shift, dist


#############################################################
# pymatgen
def _pymatgen_get_distance_pairs(
    p1: np.ndarray,
    p2: np.ndarray | None = None,
    cell: np.ndarray | Cell | None = None,
    max_distance: float = float("inf"),
    return_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(cell, Cell):
        cell = Cell.new(cell=cell)
    p1 = np.asarray(p1, dtype=float)
    if p2 is None:
        p2 = p1
    else:
        p2 = np.asarray(p2, dtype=float)
    assert p1.ndim == p2.ndim == 2
    assert p1.shape[1] == p2.shape[1] == 3
    l1, l2 = p1.shape[0], p2.shape[0]
    max_distance = float(max_distance)
    assert max_distance > 0

    if cell.rank == 0:
        return _sklearn_get_distance_pairs(
            p1=p1,
            p2=p2,
            cell=cell,
            max_distance=max_distance,
            return_distance=return_distance,
        )
    else:
        try:
            from pymatgen.optimization.neighbors import (
                find_points_in_spheres,
            )

            target, source, shift, distance = find_points_in_spheres(
                np.ascontiguousarray(p1, dtype=float),
                np.ascontiguousarray(p2, dtype=float),
                r=max_distance,
                pbc=np.ascontiguousarray(cell.array.any(1), dtype=np.int64),
                lattice=np.ascontiguousarray(cell.array, dtype=float),
                tol=1e-8,
            )
            shift = np.asarray(shift, int)
            target = np.asarray(target, int)
            source = np.asarray(source, int)
            distance = np.asarray(distance, float)
            assert distance.shape == (len(source),), "Impossible"
            assert source.shape == (len(source),), "Impossible"
            assert target.shape == (len(target),), "Impossible"
            assert shift.shape == (len(source), 3), "Impossible"
        except ImportError:
            raise ImportError(
                "Cannot import `find_points_in_spheres` from "
                "`pymatgen.optimization.neighbors` module. Please "
                "upgrade your pymatgen to 2022.0.10 or later."
            )
        assert source.max() < l1 and target.max() < l2
        self_pair = (source == target) & (distance <= 1e-5)
        cond = np.logical_not(self_pair)  # exclude self pair
        return (
            source[cond],
            target[cond],
            shift[cond] if cell.rank != 0 else None,  # type: ignore
            distance[cond] if return_distance else None,
        )


#############################################################
# interface


def distance_pairs(
    quantities: str,
    p1: np.ndarray,
    p2: np.ndarray | None = None,
    cell: np.ndarray | Cell | None = None,
    cutoff: float = 6.0,
    backend: str | Literal["sklearn", "pmg", "pymatgen"] = "pmg",
) -> tuple[np.ndarray, ...]:
    """Compute distance pairs for two points set.

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
                D = p2[j] - p1[i] + S @ a.cell
    p1: np.ndarray
        the first points set.
    p2: np.ndarray | None
        the second points set. if none, p2 is p1.
    cell: np.ndarray | Cell | None
        the periodic lattice.
    cutoff: float
        This is a global cutoff for all points pair.
    backend: str
        The backend for computation.

    Returns:
    -------
    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.
    """
    p1 = np.asarray(p1, dtype=float)
    if p2 is None:
        p2 = p1
    else:
        p2 = np.asarray(p2, dtype=float)
    assert p1.ndim == p2.ndim == 2
    assert p1.shape[1] == p2.shape[1] == 3
    cell = Cell.new(cell)
    if cell.rank != 0:
        assert np.isfinite(cutoff), (
            "The `cutoff` must is finite number for periodic cell."
        )

    if backend.lower() == "sklearn":
        i, j, S, d = _sklearn_get_distance_pairs(
            p1=p1,
            p2=p2,
            cell=cell,
            max_distance=cutoff,
            return_distance=bool("d" in quantities),
        )
    elif backend.lower() in ("pmg", "pymatgen"):
        j, i, S, d = _pymatgen_get_distance_pairs(
            p1=p2,
            p2=p1,
            cell=cell,
            max_distance=cutoff,
            return_distance=bool("d" in quantities),
        )
        # Swap i-j & p1-p2 for that the result match:
        #       D = p2[j] - p1[i] + S @ a.cell
    else:
        raise NotImplementedError(
            f"Invalid backend='{backend}', only 'sklearn',"
            " 'pmg' and 'pymatgen' are supported."
        )
    dct = dict(zip("ijSd", [i, j, S, d]))
    if "D" in quantities:
        dct["D"] = p2[j] - p1[i] + S @ cell
    return tuple(dct[q] for q in quantities)


#######################################################################
#                                   Test
#######################################################################
def test_distance_pairs_atoms(c: float = 5) -> None:
    from ase.build import fcc111

    atoms = fcc111(
        "Cu",
        [6, 8, 1],
        vacuum=10,
        orthogonal=True,
        periodic=True,
    )
    p2 = atoms.positions.copy()
    cell = atoms.cell.array.copy()

    p1 = p2[:2, :]
    i, j, S, D, d = distance_pairs("ijSDd", p1, p2, cell, 3, "pmg")
    for k in range(len(S)):
        print(p1[i[k]], p2[j[k]], sep="\n")
        print(S[k], D[k], d[k])
        print()

    np.testing.assert_array_almost_equal(d, np.linalg.norm(D, axis=1))


if __name__ == "__main__":
    test_distance_pairs_atoms()
