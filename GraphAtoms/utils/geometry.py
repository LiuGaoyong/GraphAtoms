"""The calculation of distance."""

import itertools
from typing import Literal

import numpy as np
import sparse
from ase.cell import Cell
from ase.data import covalent_radii as COV_R
from ase.geometry import find_mic, minkowski_reduce, wrap_positions
from scipy import sparse as sp
from scipy.linalg import norm, pinv
from sklearn.neighbors import NearestNeighbors

from GraphAtoms.utils.rotation import kabsch, rotate

rigid_transform_3D = kabsch
__all__ = [
    "kabsch",
    "rotate",
    "rigid_transform_3D",
    "get_distance_sparse_matrix",
    "get_adjacency_sparse_matrix",
    "get_is_inner_array",
    "rigid_transform_3D",
    "distance_factory",
]


class _DistanceFactory:
    """The factory of distance-related calculation."""

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
        """
        if not isinstance(cell, Cell):
            cell = Cell.new(cell=cell)
        p1 = np.asarray(p1, dtype=float)
        if p2 is None:
            p2 = p1
        else:
            p2 = np.asarray(p2, dtype=float)
        assert p1.ndim == p2.ndim == 2
        assert p1.shape[1] == p2.shape[1] == 3
        if len(p1) >= len(p2):
            swap = False
        else:
            swap, p1, p2 = True, p2, p1
        l1, l2 = p1.shape[0], p2.shape[0]
        max_distance = float(max_distance)
        assert max_distance > 0

        distance = cls.__get_sparse_matrix(
            nn=cls.__get_nn(X=p1),
            points=p2,
            radius=max_distance,
            return_distance=bool(return_distance),
        )
        if cell.rank != 0:
            distance = distance.astype(bool)
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
                if n1 <= 0 and (n2 < 0 or n2 == 0 and n3 < 0):
                    continue
                shift_idxs = np.array([n1, n2, n3])
                distance += cls.__get_sparse_matrix(
                    nn=nn,
                    radius=max_distance,
                    return_distance=False,
                    points=(p2 - shift_idxs @ cell),  # type: ignore
                )
            distance += distance.T
            if return_distance:
                src, tgt = distance.tocoo().coords
                v: np.ndarray = p2[src] - p1[tgt]  # type: ignore
                _, d = find_mic(v, cell=cell, pbc=True)
                distance = sp.csr_matrix(
                    (d, (src, tgt)),
                    shape=distance.shape,
                )

        assert distance.shape == (l2, l1), f"{distance.shape} != {(l2, l1)}"
        return distance if swap else distance.T

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
        adjacency_matrix: sp.csr_matrix | sp.spmatrix | sp.sparray,
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
        adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        adjacency_matrix += adjacency_matrix.T
        # get neighbors of the index atom
        assert isinstance(adjacency_matrix, sp.csr_matrix)
        nbr = sp.coo_array(adjacency_matrix[index, :])
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
        result = np.zeros_like(batch, dtype=bool)
        for index, in_batch in enumerate(batch):
            if in_batch:
                result[index] = cls.get_is_inner(
                    index=index,
                    numbers=numbers,
                    geometry=geometry,
                    adjacency_matrix=adjacency_matrix,
                    cell=cell,
                )
        return result


distance_factory = _DistanceFactory
get_distance_sparse_matrix = distance_factory.get_distance_sparse_matrix
get_adjacency_sparse_matrix = distance_factory.get_adjacency_sparse_matrix
get_is_inner_array = distance_factory.get_is_inner_array


def inverse_3d_sphere_surface_sampling(n: int) -> np.ndarray:
    """The 3D Sphere Surface Sampling Based on Inverse Transform.

    Reference:
        https://en.wikipedia.org/wiki/Inverse_transform_sampling
        http://corysimon.github.io/articles/uniformdistn-on-sphere/
        https://niepp.github.io/2021/12/09/uniform-sampling-on-sphere.html

    Args:
        n (int): the number of points for sampling.

    Returns:
        np.ndarray: (2n, 3) shape float numpy.ndarray for half sphere
    """
    u = np.random.uniform(0, 1, size=n)
    v = np.random.uniform(0, 1, size=n)
    # inversion method
    phi = v * 2 * np.pi
    cos_theta = 1 - u
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    z0, z1 = cos_theta, -cos_theta
    half0 = np.column_stack([x, y, z0])
    half1 = np.column_stack([x, y, z1])
    return np.vstack([half0, half1])
