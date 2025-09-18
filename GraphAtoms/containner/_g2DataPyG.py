"""The io mixin class of GraphContainer for PyG Data."""

from abc import ABC, abstractmethod

from numpy import asarray, isin, ndarray
from numpy.typing import ArrayLike
from torch import from_numpy
from torch_geometric import utils as pygutils
from torch_geometric.data import Data as DataPyG
from typing_extensions import Self

from GraphAtoms.common._abc import BaseMixin


class GraphMixinPyG(ABC, BaseMixin):
    @abstractmethod
    def to_pygdata(self) -> DataPyG: ...
    @classmethod
    @abstractmethod
    def from_pygdata(cls, data: DataPyG) -> Self: ...

    def get_active_subgraph(self, k: ArrayLike) -> Self:
        """Return the edge subgraph only active nodes included."""
        data = self.to_pygdata()
        na, nb = data.num_nodes, data.num_edges
        if na is None or na == 0:
            # raise KeyError(f"Got zero node PyG Data object: {data}")
            return self
        elif nb is None or nb == 0:
            # raise KeyError(f"Got zero edge PyG Data object: {data}")
            return self
        elif data.edge_index is None:
            return self
        else:
            i, j = data.edge_index.numpy(force=True)
            subset = self.get_index(k=k, n=na)
            mask = from_numpy(isin(i, subset) & isin(j, subset))
            subpygdata = data.edge_subgraph(mask)  # shallow copy
            return self.from_pygdata(data=subpygdata)

    def get_edge_subgraph(self, k: ArrayLike) -> Self:
        data = self.to_pygdata()
        n = data.num_edges
        if n is None or n == 0:
            # raise KeyError(f"Got zero edge PyG Data object: {data}")
            return self
        else:
            subset = from_numpy(self.get_mask_or_index(k=k, n=n))
            subpygdata = data.edge_subgraph(subset)  # shallow copy
            return self.from_pygdata(data=subpygdata)

    def get_induced_subgraph(self, k: ArrayLike) -> Self:
        data = self.to_pygdata()
        n = data.num_nodes
        if n is None or n == 0:
            # raise KeyError(f"Got zero node PyG Data object: {data}")
            return self
        else:
            subset = from_numpy(self.get_mask_or_index(k=k, n=n))
            subpygdata = data.subgraph(subset)  # shallow copy
            return self.from_pygdata(data=subpygdata)

    def get_k_hop_subgraph(self, k: ArrayLike, num_hops: int = 1) -> Self:
        k_hop_neighbor = self.get_k_hop_neighbor(k, num_hops=num_hops)
        if k_hop_neighbor.size == 0:
            raise KeyError(
                "The number of nodes or edges is zero. Input: self="
                f"{repr(self)}, k={repr(k)}, num_hops={repr(num_hops)}."
            )
        else:
            return self.get_k_hop_subgraph(k_hop_neighbor)

    def get_k_hop_neighbor(self, k: ArrayLike, num_hops: int = 1) -> ndarray:
        data = self.to_pygdata()
        n = data.num_nodes
        if data.edge_index is None or n is None or n == 0:
            # raise KeyError("Unsupported: PyG.Data without edge.")
            # raise KeyError(f"Got zero node PyG Data object: {data}")
            return asarray([], dtype=int)
        else:
            subset, _, _, _ = pygutils.k_hop_subgraph(
                from_numpy(self.get_index(k=k, n=n)),
                num_hops=num_hops,
                edge_index=data.edge_index,
                relabel_nodes=False,
                num_nodes=n,
                directed=False,
            )
            return subset.numpy(force=True).astype(int)
