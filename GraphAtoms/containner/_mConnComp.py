from abc import ABC, abstractmethod
from functools import cached_property

from numpy import ndarray, where
from scipy.sparse import csgraph, csr_array


class GraphMixinConnectedComponents(ABC):
    @property
    @abstractmethod
    def MATRIX(self) -> csr_array: ...

    @cached_property
    def connected_components_label(self) -> ndarray:
        return csgraph.connected_components(
            csgraph=self.MATRIX,
            directed=False,
            return_labels=True,
        )[1]

    @property
    def connected_components_number(self) -> int:
        return max(self.connected_components_label) + 1

    @cached_property
    def connected_components(self) -> list[ndarray]:
        labels = self.connected_components_label
        n: int = self.connected_components_number
        result = [where(labels == i)[0] for i in range(n)]
        return sorted(result, reverse=True, key=lambda x: len(x))

    @cached_property
    def connected_components_biggest(self) -> ndarray:
        cc: list[ndarray] = self.connected_components
        ccl: list[int] = [len(i) for i in cc]
        i_biggest = ccl.index(max(ccl))
        return cc[i_biggest]

    @property
    def is_connected(self) -> bool:
        return self.connected_components_number == 1

    @property
    def is_disconnected(self) -> bool:
        return self.connected_components_number != 1
