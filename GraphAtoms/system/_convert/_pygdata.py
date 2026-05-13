import warnings

import numpy as np
import torch
from numpy.typing import ArrayLike
from typing_extensions import Any, Self

from .._base import Base

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from torch_geometric.data import Data as DataPyG
    from torch_geometric.utils import k_hop_subgraph


class PyTorchGeometricConverter(Base):
    @classmethod
    def from_pygdata(cls, data: DataPyG, **kwargs) -> Self:
        assert data.pos is not None and "numbers" in data.keys()
        dct: dict[str, Any] = {
            "positions": data.pos.numpy(force=True),
            "pair": (
                data.edge_index.numpy(force=True).T
                if data.edge_index is not None
                else None
            ),
        } | {
            k: data[k].numpy(force=True)
            if isinstance(data[k], torch.Tensor)
            else data[k]
            for k in set(cls.__pydantic_fields__.keys()) & set(data.keys())
            if k not in ("positions", "pair")
        }
        return cls.from_dict(dct, **kwargs)

    def to_pygdata(self) -> DataPyG:
        #  UserWarning: The given NumPy array is not writable, and
        #   PyTorch does not support non-writable tensors. This means
        #   writing to this tensor will result in undefined behavior.
        #   You may want to copy the array to protect its data or make it
        #   writable before converting it to a tensor. This type of warning
        #   will be suppressed for the rest of this program. (Triggered
        #       internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            conn = np.column_stack([self.source, self.target])
            result = DataPyG(
                pos=torch.from_numpy(self.positions),
                edge_index=torch.from_numpy(conn.astype(int).T),
            )
            for k, v in self.to_dict(
                mode="python",
                exclude_none=True,
                exclude_computed_fields=True,
                exclude={"positions", "pair"},
                numpy_ndarray_compatible=True,
                numpy_convert_to_list=False,
            ).items():
                if isinstance(v, np.ndarray):
                    dtype_name: str = v.dtype.name
                    if dtype_name.startswith("uint"):
                        if dtype_name != "uint8":
                            d = dtype_name[1:]
                            v = v.astype(d)
                    result[k] = torch.from_numpy(v)
                elif np.isscalar(v) or isinstance(v, list | tuple):
                    result[k] = v
                else:
                    raise TypeError(f"{k}(type={type(v)}: {v}")
        result.validate(raise_on_error=True)
        return result

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
            mask = torch.from_numpy(np.isin(i, subset) & np.isin(j, subset))
            subpygdata = data.edge_subgraph(mask)  # shallow copy
            return self.from_pygdata(data=subpygdata)

    def get_edge_subgraph(self, k: ArrayLike) -> Self:
        data = self.to_pygdata()
        n = data.num_edges
        if n is None or n == 0:
            # raise KeyError(f"Got zero edge PyG Data object: {data}")
            return self
        else:
            subset = torch.from_numpy(self.get_mask_or_index(k=k, n=n))
            subpygdata = data.edge_subgraph(subset)  # shallow copy
            return self.from_pygdata(data=subpygdata)

    def get_induced_subgraph(self, k: ArrayLike) -> Self:
        data = self.to_pygdata()
        n = data.num_nodes
        if n is None or n == 0:
            # raise KeyError(f"Got zero node PyG Data object: {data}")
            return self
        else:
            subset = torch.from_numpy(self.get_mask_or_index(k=k, n=n))
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

    def get_k_hop_neighbor(self, k: ArrayLike, num_hops: int = 1) -> np.ndarray:
        data = self.to_pygdata()
        n = data.num_nodes
        if data.edge_index is None or n is None or n == 0:
            # raise KeyError("Unsupported: PyG.Data without edge.")
            # raise KeyError(f"Got zero node PyG Data object: {data}")
            return np.asarray([], dtype=int)
        else:
            subset, _, _, _ = k_hop_subgraph(
                torch.from_numpy(self.get_index(k=k, n=n)),
                num_hops=num_hops,
                edge_index=data.edge_index,
                relabel_nodes=False,
                num_nodes=n,
                directed=False,
            )
            return subset.numpy(force=True).astype(int)
