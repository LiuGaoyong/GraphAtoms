import numpy as np
from igraph import Graph
from networkx import Graph as NXGraph
from pandas import DataFrame
from pandas import concat as pd_concat
from rustworkx import PyGraph as RXGraph
from typing_extensions import Any, Self

from .._base import Base
from .._base._atomsAttr import AtomsAttr
from .._base._bondsAttr import BondsAttr


class IGraphConverter(Base):
    @classmethod
    def from_igraph(cls, graph: Graph, **kwargs) -> Self:
        dct: dict[str, Any] = {
            k: graph[k]
            for k in cls.__pydantic_fields__
            if k in graph.attributes()
        }
        # for atom attribute
        df = graph.get_vertex_dataframe()
        assert len(df.columns) >= 4, df.columns
        assert "numbers" in df.columns, df.columns
        R_KEYS = [f"positions_{k}" for k in "xyz"]
        assert all(k in df.columns for k in R_KEYS), df.columns  # type: ignore
        dct["numbers"] = df["numbers"].to_numpy()
        dct["positions"] = df[R_KEYS].to_numpy()
        for k in set(df.columns[4:]) & {
            "is_outer",
            "coordination",
            "move_fix_tag",
        }:
            dct[k] = df[k].to_numpy()
        # for bond attribute
        df = graph.get_edge_dataframe()
        assert len(df.columns) >= 2, df.columns
        dct["pair"] = df[df.columns[:2]].to_numpy()
        for k in set(df.columns[2:]) & {"order", "distance"}:
            if not df[k].isnull().all():
                dct[k] = df[k].to_numpy()
        return cls.from_dict(dct, **kwargs)

    def to_igraph(self, *args, **kwargs) -> Graph:
        df_atoms = DataFrame({"numbers": self.numbers})
        for i, k in enumerate("xyz"):
            df_atoms[f"positions_{k}"] = self.positions[:, i]
        for k, v in [
            ("is_outer", self.is_outer),
            ("coordination", self.coordination),
            ("move_fix_tag", self.move_fix_tag),
        ]:
            if v is not None:
                df_atoms[k] = v
        df_bonds = DataFrame({"source": self.source})
        df_bonds["target"] = self.target
        if self.order is not None:
            df_bonds["order"] = self.order
        if self.distance is not None:
            df_bonds["distance"] = self.distance
        G = Graph.DataFrame(df_bonds, False, df_atoms, True)
        for k, v in self.to_dict(
            exclude=(
                AtomsAttr.__pydantic_fields__.keys()
                | BondsAttr.__pydantic_fields__.keys()
            ),
            exclude_none=True,
            exclude_computed_fields=True,
            numpy_ndarray_compatible=True,
            numpy_convert_to_list=False,
        ).items():
            G[k] = v
        return G


class NetworkXConverter(IGraphConverter):
    def to_networkx(self, *args, **kwargs) -> NXGraph:
        return self.to_igraph(*args, **kwargs).to_networkx()

    @classmethod
    def from_networkx(cls, graph: NXGraph, **kwargs) -> Self:
        return cls.from_igraph(graph=Graph.from_networkx(graph), **kwargs)


class RustworkXConverter(IGraphConverter):
    def to_rustworkx(self) -> RXGraph:
        graph = self.to_igraph()
        # Ref: https://github.com/igraph/python-igraph/blob/main/src/igraph/io/libraries.py#L1-L70
        G = RXGraph(
            multigraph=False,
            attrs={x: graph[x] for x in graph.attributes()},
        )
        G.add_nodes_from([v.attributes() for v in graph.vs])
        G.add_edges_from((e.source, e.target, e.attributes()) for e in graph.es)
        return G

    @classmethod
    def from_rustworkx(cls, graph: RXGraph, **kwargs) -> Self:
        df_nodes = DataFrame(graph.nodes())
        source, target = np.asarray(graph.edge_list(), dtype=int).T
        df_edges0 = DataFrame({"i": source, "j": target})
        df_edges1 = DataFrame(graph.edges())
        df_edges = pd_concat(
            [df_edges0, df_edges1],
            ignore_index=True,
            axis="columns",
        )
        df_edges.columns = list(df_edges0.columns) + list(df_edges1.columns)
        igraph = Graph.DataFrame(df_edges, False, df_nodes, True)
        for k, v in graph.attrs.items():  # attr is dict type
            if k in cls.__pydantic_fields__ and v is not None:
                igraph[k] = v
        return cls.from_igraph(graph=igraph, **kwargs)


# ============================================================================


class GraphConverter(NetworkXConverter, RustworkXConverter):
    pass
