from abc import ABC, abstractmethod
from functools import cached_property, partial, reduce
from typing import Literal

import numpy as np
from igraph import Graph
from numpy.typing import ArrayLike
from pandas import DataFrame
from pydantic import PositiveInt, validate_call
from scipy.sparse import csr_array
from typing_extensions import Self

from GraphAtoms.common._abc import BaseMixin
from GraphAtoms.containner._aMixin import ATOM_KEY
from GraphAtoms.utils.string import hash as hash_string


class _GraphMixinIGraph(ABC, BaseMixin):
    @abstractmethod
    def to_igraph(self) -> Graph: ...
    @classmethod
    @abstractmethod
    def from_igraph(cls, graph: Graph) -> Self: ...
    @property
    @abstractmethod
    def COLOR(self) -> list[str] | np.ndarray: ...
    @property
    @abstractmethod
    def Z(self) -> np.ndarray: ...
    @property
    @abstractmethod
    def MATRIX(self) -> csr_array: ...
    @property
    @abstractmethod
    def DF_BONDS(self) -> DataFrame: ...

    @property
    def __DF_ATOMS(self) -> DataFrame:
        df = DataFrame({ATOM_KEY.NUMBER: self.Z})
        df["color"] = np.vectorize(hash)(self.COLOR)
        return df

    @cached_property
    def IGRAPH(self) -> Graph:
        return Graph.DataFrame(
            edges=self.DF_BONDS,
            vertices=self.__DF_ATOMS,
            directed=False,
            use_vids=True,
        )

    @cached_property
    def nsymmetry(self) -> int:
        return self.IGRAPH.count_automorphisms_vf2(self.Z)

    @validate_call
    def get_weisfeiler_lehman_hashes(
        self,
        hash_depth: PositiveInt = 3,
        digest_size: PositiveInt = 6,
    ) -> list[str]:
        """Return hash value for each atom."""
        labels, g = self.COLOR, self.IGRAPH
        for _ in range(hash_depth):
            labels = GraphMixinIGraph.weisfeiler_lehman_step(
                igraph=g,
                igcolor=labels,
                digest_size=digest_size,
            )
        return [str(i) for i in labels]

    @staticmethod
    def weisfeiler_lehman_step(
        igraph: Graph,
        igcolor: list[str] | list[int] | np.ndarray,
        digest_size: int = 6,
    ) -> np.ndarray:
        """Return hash string for each vertex by weisfeiler lehman algorithm.

        Reference:
            https://arxiv.org/pdf/1707.05005.pdf
            http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
            https://www.cnblogs.com/cyz666/p/9167507.html

        Args:
            igraph (Graph | np.matrix | np.ndarray | sp.spmatrix): .
            igcolor (list[str] | list[ int] | np.ndarray | None, optional):
                The vertex color in interger or string type. Defaults to None.
            digest_size (int, optional): The size of output str. Default to 6.

        Returns:
            list[str]: the hash string for each vertex.
        """
        igcolor = [str(i) for i in igcolor]
        assert len(igcolor) == igraph.vcount()
        return np.asarray(
            [
                hash_string(
                    label
                    + "".join(
                        sorted([igcolor[j] for j in igraph.neighbors(i)])
                    ),
                    digest_size=digest_size,
                )
                for i, label in enumerate(igcolor)
            ]
        )

    @classmethod
    def chordless_cycles(
        cls,
        igraph: Graph,
        batch: ArrayLike | list[int | bool] | int | None = None,
        max_ncore: Literal[3, 4, 5, 6] = 3,
    ) -> np.ndarray:
        """Got chordless cycles mask array for specific core numbers.

        Args:
            igraph (Graph): The graph to calculate.
            batch (np.ndarray | list[int | bool] | None, optional):
                the batch atoms of calculation. Default to None.
            max_ncore (Literal[3, 4, 5, 6], optional): The maximum
                number of core atoms in a site. Defaults to 3.

        Returns:
            np.ndarray: (n_site, n_atoms) boolean matrix
        """
        _batch = np.arange(igraph.vcount())
        if batch is not None:
            batch = np.asarray(batch)
            if batch.dtype in (bool, np.bool_):
                batch = _batch[batch]
        elif np.isscalar(batch):
            batch = [batch]  # type: ignore
        else:
            batch = _batch
        batch = np.asarray(batch, dtype=int)
        assert np.all(np.isin(batch, _batch))
        assert max_ncore in (3, 4, 5, 6), (
            f"Invalid max_ncore value: {max_ncore}."
            " Only 3, 4, 5 and 6 are supported here."
        )
        nbr = igraph.neighborhood(batch, order=3, mindist=0)
        nbr = np.unique(reduce(np.append, nbr))
        g: Graph = cls._subgraph_edges(igraph, nbr)
        if not g.is_simple():
            g = g.simplify()
        subisomor = partial(g.get_subisomorphisms_lad, induced=True)
        result: dict[int, np.ndarray] = {}
        for x in subisomor(Graph(3, [(0, 1), (1, 2), (2, 0)])):
            i = hash(tuple(sorted(x)))
            if i not in result:
                result[i] = np.isin(_batch, x)
        for n, edges in [
            (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
            (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
            (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        ]:
            if max_ncore >= n:
                for x in subisomor(Graph(n, edges)):
                    i = hash(tuple(sorted(x)))
                    if i not in result:
                        bonds = igraph.es.select(_within=x)
                        if len(bonds) == n:
                            result[i] = np.isin(_batch, x)
        return np.asarray([i for i in result.values()])

    @classmethod
    def _subgraph_edges(
        cls,
        igraph: Graph,
        active: np.ndarray | list[int] | list[bool] | None = None,
    ) -> Graph:
        """Return the edge subgraph only active bonds included."""
        active, n = np.asarray(active), igraph.vcount()
        if active.dtype in (bool, np.bool_):
            active = np.arange(n)[active]
        active = np.asarray(active, dtype=int)
        assert np.all(np.isin(active, np.arange(n)))
        active_bonds = igraph.es.select(_within=active)
        return igraph.subgraph_edges(active_bonds, delete_vertices=False)


class GraphMixinIGraph(_GraphMixinIGraph):
    def get_hop_distance(self, k: ArrayLike) -> np.ndarray:
        idx = self.get_index(k, self.Z.shape[0])
        d: list = self.IGRAPH.distances(idx)
        return np.asarray(d).min(axis=0)

    @classmethod
    @validate_call(config={"arbitrary_types_allowed": True})
    def match(
        cls,
        pattern: _GraphMixinIGraph,
        pattern4match: _GraphMixinIGraph,  # big graph
        algorithm: Literal["lad", "vf2"] = "lad",
        return_match_target: bool = True,
        only_number_color: bool = False,
        only_count: bool = False,
    ) -> int | None | np.ndarray:
        """Match two pattern by subisomorphisms algorithm.

        Args:
            pattern (AtomsPattern): the target pattern.
            pattern4match (AtomsPattern): the pattern for matching.
            algorithm (Literal["lad", "vf2"], optional): the algorithm of use.
                VF2 is optimal when max(nodes) < 10^3 and attach more attr.
                LAD is optimal when max(nodes) > 10^4 and sparse graph.
                    Defaults to "lad".
            return_match_target (bool, optional): wether return matched
                indexes for target pattern. Defaults to True.
            only_number_color (bool, optional): wether only color
                by numbers. Defaults to False.
            only_count (bool, optional): wether only return matching number.
                Defaults to False.

        Returns:
            int | None | np.ndarray:
                If only_count is True, the result is the integer.
                Otherwise, the result is None when no matching found.
                    If matched, a (N_match, N_pattern)
                    shape numpy.ndarray will be outputed.
        """
        graph: Graph = pattern4match.IGRAPH
        graph_small: Graph = pattern.IGRAPH
        if only_number_color:
            color, color_small = pattern4match.Z, pattern.Z
        else:
            color = np.vectorize(hash)(pattern4match.COLOR)
            color_small = np.vectorize(hash)(pattern.COLOR)
        assert color.shape[0] >= color_small.shape[0], (
            f"Pattern(N={color_small.shape[0]}) is "
            f"too big !!! Target(N={color.shape[0]})."
        )

        if not np.all(np.isin(color_small, color)):
            out = []
        elif algorithm == "lad":
            out = graph.get_subisomorphisms_lad(
                pattern=graph_small,
                domains=[
                    np.argwhere(color == color_small[i]).flatten()
                    for i in range(len(color_small))
                ],
                induced=True,
                # time_limit=2,
            )
        elif algorithm == "vf2":
            out = graph.get_subisomorphisms_vf2(
                other=graph_small,
                color1=color,
                color2=color_small,
            )
        else:
            raise RuntimeError("Impossible !!!")

        if only_count:
            return len(out)
        elif len(out) == 0:
            return None
        elif return_match_target:
            result = -np.ones(shape=(len(out), len(color)), dtype=int)
            result[
                np.column_stack([np.arange(len(out))] * len(out[0])),
                np.asarray(out, dtype=int),
            ] = np.arange(len(color_small), dtype=int)
            return result
        else:
            return np.asarray(out, dtype=int)
