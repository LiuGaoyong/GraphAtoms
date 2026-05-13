from functools import cached_property
from typing import override

import numpy as np
from scipy import sparse as sp

from .._base import Base


class ScipyGraphMixin(Base):
    """Mixin for scipy-based graph operations.

    Provides sparse matrix representations, coordination numbers,
    connectivity analysis, and neighbor queries.
    """
    @cached_property
    def MATRIX(self) -> sp.csr_array:
        if self.order is None:
            order = np.ones(self.nbonds, bool)
        else:
            # scipy.sparse does not support dtype float16(self.order.dtype)
            # so we use single float numbers(float32) in the return statement
            order = np.asarray(self.order, dtype="f4")
        return sp.csr_array(
            (order, (self.source, self.target)),
            shape=(self.natoms, self.natoms),
        )

    @cached_property
    @override
    def _CN_MATRIX(self) -> np.ndarray:
        m = self.MATRIX.astype(bool)
        m = sp.csr_array((m + m.T).astype(int))
        return np.asarray(m.sum(axis=1)).astype(int)

    @property
    @override
    def CN(self) -> np.ndarray:
        if self.coordination is not None:
            return self.coordination
        else:
            return self._CN_MATRIX

    def get_neighbors(self, i: int) -> np.ndarray:
        idx1 = self.source[self.target == i]
        idx2 = self.target[self.source == i]
        return np.unique(np.append(idx1, idx2))

    def check_induced_graph(self) -> bool:
        if self.coordination is None:
            return False
        else:
            return bool(np.all(self._CN_MATRIX == self.coordination))

    @cached_property
    def __get_connected_components(self) -> tuple[int, np.ndarray]:
        return sp.csgraph.connected_components(
            csgraph=self.MATRIX,
            directed=False,
            return_labels=True,
        )

    @cached_property
    def connected_components_number(self) -> int:
        return self.__get_connected_components[0]

    @cached_property
    def connected_components(self) -> list[np.ndarray]:
        labels = self.__get_connected_components[1]
        n: int = self.connected_components_number
        result = [np.where(labels == i)[0] for i in range(n)]
        return sorted(result, reverse=True, key=lambda x: len(x))

    @cached_property
    def connected_components_biggest(self) -> np.ndarray:
        cc: list[np.ndarray] = self.connected_components
        ccl: list[int] = [len(i) for i in cc]
        i_biggest = ccl.index(max(ccl))
        return cc[i_biggest]

    @cached_property
    def is_connected(self) -> bool:
        return self.connected_components_number == 1
