from typing import Annotated, Self

import numpy as np
import pydantic

from graphatoms.dataclasses._numpydantic import NDArray, numpy_validator
from graphatoms.dataclasses._pydanticMixin import (
    PydanticIoFactoryMixin as PydanticMixin,
)
from graphatoms.dataclasses._pydanticModel import (
    _IndexMaskMixin as IndexMaskMixin,
)


class Graph(PydanticMixin, IndexMaskMixin):
    """The undirected graph based on numpy array."""

    nvertex: int  # the number of vertices in this graph
    edges: Annotated[NDArray, numpy_validator("int32", (-1, 2))]
    vertex_attrs: dict[str, NDArray] = {}  # the vertex attributes
    graph_attrs: dict[str, NDArray] = {}  # the graph attributes
    edge_attrs: dict[str, NDArray] = {}  # the edge attributes

    @property
    def nedge(self) -> int:
        return self.edges.shape[0]

    @pydantic.model_validator(mode="after")
    def __check_for_undirected(self) -> Self:
        if self.nvertex == 0:
            msg = "The graph must be empty. But N({:s})={:d}."
            for k, n in [
                ("edges", self.edges.shape[0]),
                ("vertex_attrs", len(self.vertex_attrs)),
                ("graph_attrs", len(self.graph_attrs)),
                ("edge_attrs", len(self.edge_attrs)),
            ]:
                assert n == 0, msg.format(k, n)
        else:
            # 1. check edges
            less = f"less than N={self.nvertex}."
            msg = "The {:s} vertex index must be {:s}."
            assert np.min(self.edges) >= 0, msg.format("min", "non-negative")
            assert np.max(self.edges) < self.nvertex, msg.format("max", less)
            # 2. check vertex_attrs
            msg = "The {:s} vertex attribute's length"
            msg += f" must be {self.nvertex:d}."
            for k, v in self.vertex_attrs.items():
                assert v.shape[0] == self.nvertex, msg.format(k)
            # 3. check edge_attrs
            msg = "The {:s} edge attribute's length"
            msg += f" must be {self.edges.shape[0]:d}."
            for k, v in self.edge_attrs.items():
                assert v.shape[0] == self.edges.shape[0], msg.format(k)
            # 4-A. delete self-loop edges
            mask = self.edges[:, 0] != self.edges[:, 1]
            attrs = {k: v[mask] for k, v in self.edge_attrs.items()}
            # 4-B. deduplicate edges because of undirected graph representation
            self.edges, idxs = np.unique(self.edges[mask], True, axis=0)
            self.edge_attrs = {k: v[idxs] for k, v in attrs.items()}
            for k, v in self.edge_attrs.items():
                assert v.shape[0] == self.edges.shape[0], msg.format(k)
        return self

