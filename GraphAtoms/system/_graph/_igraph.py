from functools import cached_property, partial, reduce
from typing import Literal, override

import numpy as np
import pydantic
from igraph import Graph
from numpy.typing import ArrayLike

from GraphAtoms.utils.bytestool import hash_string

from ._scipy import ScipyGraphMixin

DEFAULT_WH_HASH_DEPTH = 3
DEFAULT_WH_HASH_SIZE = 6


class IGraphMixin(ScipyGraphMixin):
    @cached_property
    def __IGRAPH(self) -> Graph:
        edges = np.column_stack([self.source, self.target])
        return Graph(n=self.natoms, edges=edges, directed=False)

    @cached_property
    def __COLOR(self) -> np.ndarray:
        z = np.char.mod("%d-", self.numbers)
        cn = np.char.mod("%d-", self.CN)
        return np.char.add(z, cn)

    def get_igcolor(self) -> np.ndarray:
        return self.__COLOR

    def get_igraph(self) -> Graph:
        """Return the igraph object which only edges included."""
        return self.__IGRAPH

    @cached_property
    def nsymmetry(self) -> int:
        return self.__IGRAPH.count_automorphisms_vf2(self.numbers)

    @cached_property
    @override
    def hash(self) -> str:
        return hash_string(
            ",".join(
                sorted(
                    self.hashes
                    if self.hashes is not None
                    else self.get_weisfeiler_lehman_hashes()
                )
            ),
            digest_size=DEFAULT_WH_HASH_SIZE,
        )

    @pydantic.validate_call
    def get_weisfeiler_lehman_hashes(
        self,
        hash_depth: pydantic.PositiveInt = DEFAULT_WH_HASH_DEPTH,
        digest_size: pydantic.PositiveInt = DEFAULT_WH_HASH_SIZE,
    ) -> list[str]:
        """Return hash value for each atom."""
        labels: np.ndarray = self.__COLOR
        for _ in range(hash_depth):
            labels = self.__weisfeiler_lehman_step(
                igcolor=labels,
                digest_size=digest_size,
            )
        result = [str(i) for i in labels]
        object.__setattr__(self, "hashes", result)
        assert self.hashes is not None
        return self.hashes

    def __weisfeiler_lehman_step(
        self,
        igcolor: list[str | int] | np.ndarray,
        digest_size: int = DEFAULT_WH_HASH_SIZE,
    ) -> np.ndarray:
        """Return hash string for each vertex by weisfeiler lehman algorithm.

        Reference:
            https://arxiv.org/pdf/1707.05005.pdf
            http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
            https://www.cnblogs.com/cyz666/p/9167507.html

        Args:
            igcolor (list[str] | list[ int] | np.ndarray | None, optional):
                The vertex color in interger or string type. Defaults to None.
            digest_size (int, optional): The size of output str. Default to 6.

        Returns:
            list[str]: the hash string for each vertex.
        """
        igcolor = np.asarray(igcolor, dtype=str)
        assert len(igcolor) == self.natoms
        return np.asarray(
            [
                hash_string(
                    label + "".join(np.sort(igcolor[self.get_neighbors(i)])),
                    digest_size=digest_size,
                )
                for i, label in enumerate(igcolor)
            ]
        )

    def get_hop_distance(self, k: list[int] | np.ndarray) -> np.ndarray:
        k = self.get_index(k, self.natoms)
        d: list = self.__IGRAPH.distances(k)
        return np.asarray(d).min(axis=0)

    @staticmethod
    def match(
        pattern: "IGraphMixin",
        pattern4match: "IGraphMixin",  # big graph
        algorithm: Literal["lad", "vf2"] = "lad",
        return_match_target: bool = True,
        only_number_color: bool = False,
        only_count: bool = False,
    ) -> int | None | np.ndarray:
        """Match two pattern by subisomorphisms algorithm.

        Args:
            pattern (AtomsPattern): the target pattern(small).
            pattern4match (AtomsPattern): the pattern(big) for matching.
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
        cls: type = IGraphMixin
        msg = "The `{:s}` must be a `IGraphMixin` instance."
        assert isinstance(pattern, cls), msg.format("pattern")
        assert isinstance(pattern4match, cls), msg.format("pattern4match")
        assert algorithm in ["lad", "vf2"], "algorithm must be lad or vf2."

        graph: Graph = getattr(pattern4match, f"_{cls.__name__}__IGRAPH")
        graph_small: Graph = getattr(pattern, f"_{cls.__name__}__IGRAPH")
        if only_number_color:
            color, color_small = pattern4match.numbers, pattern.numbers
        else:
            color = getattr(pattern4match, f"_{cls.__name__}__COLOR")
            color_small = getattr(pattern, f"_{cls.__name__}__COLOR")
            color_small = np.vectorize(hash)(color_small)
            color = np.vectorize(hash)(color)
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

    @staticmethod
    def chordless_cycles(
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
        g: Graph = _subgraph_edges(igraph, nbr)
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


def _subgraph_edges(
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
