"""The calculation of distance."""

"""Distance calculation factory for atomic configurations."""

import itertools
from abc import ABC, abstractmethod
from typing import Literal, override

import numpy as np
import sparse
from ase.cell import Cell
from ase.data import covalent_radii as COV_R
from ase.geometry import minkowski_reduce, wrap_positions
from scipy import sparse as sp
from scipy.linalg import norm, pinv
from sklearn.neighbors import NearestNeighbors

from .sample import inverse_3d_sphere_surface_sampling


class DistanceFactoryBase(ABC):
    """Base class for distance-related calculations.

    Provides factory methods for computing neighbor lists, distance matrices,
    adjacency matrices, and inner atom detection.
    """

    def available(self) -> bool:
        return True

    @classmethod
    @abstractmethod
    def get_neighbor_list(
        cls,
        p1: np.ndarray,
        p2: np.ndarray | None = None,
        cell: np.ndarray | Cell | None = None,
        max_distance: float = float("inf"),
        return_distance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Get Distance Matrix.

        Args:
            p1 (np.ndarray): The first group of positions.
            p2 (np.ndarray | None, optional): The second group
                of positions. Defaults to None. It means p2 is p1.
            cell (np.ndarray, optional): The periodic cell. Defaults to None.
            max_distance (float, optional): Defaults to float("inf").
            return_distance (bool, optional): Defaults to True.

        Returns:
            - source: The 1D int array of index in `p1`.
            - target: The 1D int array of index in `p2`.
            - shift: 2D Nx3 int array. if cell is None, it is None.
            - distance: 1D float array. if not return_distance, it is None.

        The all of returns must has same N i.e. length.
        """

    @classmethod
    def get_distance_sparse_matrix(
        cls,
        p1: np.ndarray,
        p2: np.ndarray | None = None,
        cell: np.ndarray | Cell | None = None,
        max_distance: float = float("inf"),
        return_distance: bool = True,
    ) -> sp.csr_matrix:
        """Get Distance Matrix.

        Args:
            p1 (np.ndarray): The first group of positions.
            p2 (np.ndarray | None, optional): The second group
                of positions. Defaults to None. It means p2 is p1.
            cell (np.ndarray, optional): The periodic cell. Defaults to None.
            max_distance (float, optional): Defaults to float("inf").
            return_distance (bool, optional): Defaults to True.

        Returns:
            coo_array: The Distance Matrix with shape (len(p1), len(p2)).

        Note: if cell is not None, the MIC distance will
            be included in the result sparse matrix.
        """
        p1 = np.asarray(p1, dtype=float)
        if p2 is None:
            p2 = p1
        else:
            p2 = np.asarray(p2, dtype=float)
        assert p1.ndim == p2.ndim == 2
        assert p1.shape[1] == p2.shape[1] == 3
        source, target, _, distance = cls.get_neighbor_list(
            p1=p1,
            p2=p2,
            cell=cell,
            max_distance=max_distance,
            return_distance=True,
        )
        shape = (p1.shape[0], p2.shape[0])
        assert isinstance(distance, np.ndarray)
        dtype = float if return_distance else bool
        data = np.column_stack([distance, source, target])
        data = data[(source != target) | (distance > 1e-5)]
        data = data[np.lexsort(data.T)]  # lexsort , distance last & ij first
        distance, (source, target) = data[:, 0], np.asarray(data[:, 1:].T, int)
        _, idx = np.unique(source * shape[0] + target, return_index=True)
        dist, i, j = distance[idx], source[idx], target[idx]
        return sp.csr_matrix((dist, (i, j)), shape, dtype)

    @staticmethod
    def __get_bool_batch(
        numbers: np.ndarray,
        batch: np.ndarray | list[int | bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return numbers (int) & batch (bool) array."""
        numbers = np.asarray(numbers, dtype=int)
        if batch is None:
            batch = np.ones_like(numbers, dtype=bool)
        else:
            batch = np.asarray(batch)
            if batch.dtype not in (bool, np.bool_):
                arr = np.zeros_like(numbers, dtype=bool)
                arr[batch] = True
                batch = arr
            else:
                assert batch.shape == numbers.shape
        return numbers, batch

    @classmethod
    def get_distance_reduce_array(
        cls,
        p1: np.ndarray,
        p2: np.ndarray | None = None,
        cell: np.ndarray | Cell | None = None,
        max_distance: float = float("inf"),
        reduce_axis: Literal[0, 1] = 0,
    ) -> np.ndarray:
        d_sp = cls.get_distance_sparse_matrix(p1, p2, cell, max_distance)
        d_coo = sp.coo_matrix(d_sp, copy=False)
        d_coo2 = sparse.COO(
            coords=d_coo.coords,
            fill_value=np.inf,
            shape=d_coo.shape,
            data=np.where(d_coo.data == 0, np.inf, d_coo.data),
        )
        return d_coo2.min(axis=reduce_axis).todense()

    @classmethod
    def get_adjacency_sparse_matrix(
        cls,
        numbers: np.ndarray,
        geometry: np.ndarray,
        batch: np.ndarray | list[int | bool] | None = None,
        batch_other: np.ndarray | list[int | bool] | None = None,
        cov_multiply_factor: float = 1.0,
        cov_plus_factor: float = 0.5,
        cell: Cell | None = None,
    ) -> sp.csr_matrix:
        """Get Adjacency Matrix by distance.

        Args:
            numbers (np.ndarray): the atomic number.
            geometry (np.ndarray): the atomic geometry.
            batch (np.ndarray | list[int | bool] | None):
                the batch atoms of calculation. Default to None.
            batch_other (np.ndarray | list[int | bool] | None):
                the other batch atoms of calculation. Default to None.
            cell (Cell, optional): the periodic cell. Defaults to None.
            cov_multiply_factor (float): The multiply factor(1.0).
            cov_plus_factor (float): The plus factor(0.5).

        Note:
            connected = bool(
                distance < (ri + rj)
                         * cov_multiply_factor      # Default to 1.0
                         + cov_plus_factor means    # Default to 0.5
            )

        Returns:
            sp.csr_matrix: The Adjacency Matrix.
        """
        _, batch_other = cls.__get_bool_batch(numbers, batch_other)
        numbers, batch = cls.__get_bool_batch(numbers, batch)
        geometry = np.asarray(geometry, dtype=float)
        max_d = 2 * max(COV_R[numbers])
        max_d *= cov_multiply_factor
        max_d += cov_plus_factor
        n = len(numbers)

        distance = cls.get_distance_sparse_matrix(
            p1=geometry[batch],
            p2=geometry[batch_other],
            max_distance=max_d,
            return_distance=True,
            cell=cell,
        ).tocoo()
        d, (src, tgt) = distance.data, distance.coords
        if distance.shape != (np.sum(batch), np.sum(batch_other)):
            if distance.shape == (n, n):
                tgt = np.arange(n)[batch_other][tgt]
                src = np.arange(n)[batch][src]
        # if error, check distance.shape, src & tgt

        deq = COV_R[numbers[src]] + COV_R[numbers[tgt]]
        deq = cov_multiply_factor * deq + cov_plus_factor
        mask = np.logical_and(1e-3 < d, d < deq)
        src, tgt = src[mask], tgt[mask]
        d = np.ones_like(src, dtype=bool)
        result = sp.csr_matrix(
            (d, (src, tgt)),
            shape=distance.shape,
            dtype=bool,
        )
        assert result.dtype in (bool, np.bool_)
        return result

    @classmethod
    def get_is_inner(
        cls,
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
            mesh = geometry[index] + mesh * COV_R[numbers[index]]
            # get distance between mesh and neighbor
            distance = cls.get_distance_sparse_matrix(
                p1=mesh,
                p2=np.atleast_2d(geometry[idxs_neighbor]),
                max_distance=(
                    np.max(COV_R[numbers[idxs_neighbor]])
                    + float(COV_R[numbers[index]])
                ),
                return_distance=True,
                cell=cell,
            ).tocoo()
            # get mesh mask array
            d, (src, tgt) = distance.data, distance.coords
            mask = d >= COV_R[tgt]
            d, src, tgt = d[mask], src[mask], tgt[mask]
            d, shape = np.ones_like(d, dtype=bool), distance.shape
            mask = sp.csr_matrix(
                (d, (src, tgt)),
                shape=shape,
                dtype=bool,
            )
            assert mask.shape == (len(mesh), len(idxs_neighbor))
            mask_array = mask.sum(axis=1, dtype=bool)
            # set is_inner
            return bool(np.sum(mask_array) == len(mesh))

    @classmethod
    def get_is_inner_array(
        cls,
        numbers: np.ndarray,
        geometry: np.ndarray,
        adjacency_matrix: sp.csr_matrix | sp.spmatrix | sp.sparray,
        batch: np.ndarray | list[int | bool] | None = None,
        cell: Cell | None = None,
    ) -> list[bool] | np.ndarray:
        """Check whether each atom is inner by distance.

        Args:
            numbers (np.ndarray): the atomic number.
            geometry (np.ndarray): the atomic geometry.
            adjacency_matrix (sp.csr_matrix): the adjacency matrix.
            batch (np.ndarray | list[int | bool], optional):
                the batch atoms of calculation. Default to None.
            cell (Cell, optional): the periodic cell. Defaults to None.

        Returns:
            list[bool]: whether each atom is inner.
        """
        numbers, batch = cls.__get_bool_batch(numbers, batch)
        adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        adjacency_matrix += adjacency_matrix.T
        # adjacency_matrix = adjacency_matrix.todia()  # type: ignore
        result = np.zeros_like(batch, dtype=bool)
        for index, in_batch in enumerate(batch):
            if in_batch:
                result[index] = cls.get_is_inner(
                    index=index,
                    numbers=numbers,
                    geometry=geometry,
                    adjacency_array=adjacency_matrix[index, :],  # type: ignore
                    cell=cell,
                )
        return result


class SKLearnDistanceFactory(DistanceFactoryBase):
    """Distance factory using sklearn's NearestNeighbors.

    Suitable for non-periodic or small periodic systems.
    """

    @staticmethod
    def __get_nn(X) -> NearestNeighbors:
        nn = NearestNeighbors()
        nn.fit(X=X, y=None)
        return nn

    @staticmethod
    def __get_sparse_matrix(
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

    @classmethod
    @override
    def get_neighbor_list(
        cls,
        p1: np.ndarray,
        p2: np.ndarray | None = None,
        cell: np.ndarray | Cell | None = None,
        max_distance: float = float("inf"),
        return_distance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
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

        distance = cls.__get_sparse_matrix(
            nn=cls.__get_nn(X=p1),
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
            nn = cls.__get_nn(X=p1)
            cell_pinv_T = np.transpose(pinv(cell))
            n = [int(max_distance * norm(i)) + 1 for i in cell_pinv_T]
            nrange3 = [range(-i, i + 1) for i in n]
            for n1, n2, n3 in itertools.product(*nrange3):
                # if n1 <= 0 and (n2 < 0 or n2 == 0 and n3 < 0):
                #     continue
                shift_idxs = np.array([n1, n2, n3])
                distance = cls.__get_sparse_matrix(
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


class PyMatGenDistanceFactory(DistanceFactoryBase):
    """Distance factory using pymatgen's find_points_in_spheres.

    Optimized for periodic systems with large cells.
    """

    @override
    def available(self) -> bool:
        """Check if pymatgen is available."""
        try:
            from pymatgen.optimization.neighbors import (
                find_points_in_spheres,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    @classmethod
    @override
    def get_neighbor_list(
        cls,
        p1: np.ndarray,
        p2: np.ndarray | None = None,
        cell: np.ndarray | Cell | None = None,
        max_distance: float = float("inf"),
        return_distance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
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
            obj = SKLearnDistanceFactory()
            source, target, shift, distance = obj.get_neighbor_list(
                p1,
                p2,
                max_distance=max_distance,
                return_distance=True,
            )
            assert isinstance(distance, np.ndarray)
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


#########################################################################
__all__ = [
    "DistanceFactoryBase",
    "PyMatGenDistanceFactory",
    "SKLearnDistanceFactory",
]


SUPPORT_DISTANCE_BACKEND = ["sklearn", "pymatgen"]


def get_distance_factory(
    backend: str | Literal["sklearn", "pymatgen"] = "sklearn",
) -> DistanceFactoryBase:
    """Instantiate a factory class for distance calculation."""
    assert backend in SUPPORT_DISTANCE_BACKEND, (
        f"The backend of {backend} is not supported. "
        f"Only {SUPPORT_DISTANCE_BACKEND} are available."
    )

    if backend == "sklearn":
        result = SKLearnDistanceFactory()
    elif backend == "pymatgen":
        result = PyMatGenDistanceFactory()
    else:
        raise ValueError(
            f"Unknown backend of {backend} for distace factory."
            f" Only {SUPPORT_DISTANCE_BACKEND} are available."
        )

    if result.available:
        return result
    else:
        return SKLearnDistanceFactory()
