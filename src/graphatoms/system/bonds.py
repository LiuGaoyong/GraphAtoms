from functools import cached_property, partial, reduce
from typing import Annotated, Literal, Self, override

import numpy as np
import pydantic
from igraph import Graph as IGraph
from numpy.typing import ArrayLike
from pandas import DataFrame
from rdkit.Chem import Mol as RDMol  # type: ignore
from scipy import sparse as sp

from graphatoms.dataclasses import NDArray, OurFrozenModel, numpy_validator
from graphatoms.system.atoms import Matter
from graphatoms.utils.bytestool import hash_string
from graphatoms.utils.rdutils import get_rdmol

DEFAULT_WH_HASH_DEPTH = 3
DEFAULT_WH_HASH_SIZE = 6

__all__ = ["BondGraph"]


class BondGraph(Matter, OurFrozenModel):
    coordination: Annotated[NDArray, numpy_validator("uint8")] | None = None
    pair: Annotated[NDArray, numpy_validator("int32", (-1, 2))] | None = None
    distance: Annotated[NDArray, numpy_validator("float32")] | None = None
    order: Annotated[NDArray, numpy_validator("float16")] | None = None
    hashes: list[str] | None = None
    # hash: str | None = None

    @pydantic.model_validator(mode="after")
    def __check_atoms_and_bonds(self) -> Self:
        self._BOND_ATTRS = ("pair", "distance", "order")
        for k in ("coordination", "hashes"):
            v = getattr(self, k, None)
            if v is not None:
                assert len(v) == self.natoms, (
                    f"Invalid shape for `{k}`: Len({k})="
                    f"{len(v)} but natoms={self.natoms}."
                )
        if self.nbonds != 0:
            assert isinstance(self.pair, np.ndarray), (
                "The `pair` should be a numpy array if nbonds > 0."
            )
            assert self.pair.ndim == 2 and self.pair.shape[1] == 2, (
                "The `pair` should be a 2D array with shape (nbonds, 2)."
                f"But we got {self.pair.shape} instead."
            )
            if np.any(self.pair[:, 0] == self.pair[:, 1]):
                raise ValueError(
                    "The `pair` should not contain self-loop bonds. It "
                    "is typically caused by the structure is periodic "
                    "and contains too less atoms (bulk? surface? etc). "
                )
            for k in BondGraph.__pydantic_fields__.keys():
                if k in Matter.__pydantic_fields__.keys():
                    continue
                elif k in ("coordination", "hashes"):
                    continue
                else:
                    v = getattr(self, k, None)
                    if v is not None:
                        assert v.shape[0] == self.nbonds, (
                            f"Invalid shape for `{k}`: Len({k})="
                            f"{len(v)} but nbonds={self.nbonds}."
                        )
            # assert self.__IGRAPH.is_simple(), "The graph should be simple."
        return self

    @override
    def _string(self) -> str:
        lst = [f"{self.nbonds}Bonds"]
        if self.coordination is not None:
            lst.append("Sub")
        return ",".join(lst)

    @cached_property
    def nbonds(self) -> int:
        if self.pair is None:
            return 0
        return self.pair.shape[0]

    @cached_property
    def source(self) -> NDArray:
        return self.P[:, 0]

    @cached_property
    def target(self) -> NDArray:
        return self.P[:, 1]

    @cached_property
    def P(self) -> NDArray:
        if self.pair is None:
            return np.array([[]], dtype="int32").reshape(-1, 2)
        return self.pair

    @cached_property
    def D(self) -> NDArray:
        if self.distance is None:
            return np.array([[]], dtype="float32").reshape(-1)
        return self.distance

    @property
    def CN(self) -> np.ndarray:
        if self.coordination is not None:
            return self.coordination
        else:
            return self.__CN_MATRIX

    ###################################################
    # Some functions by `igraph`
    @cached_property
    def __IGRAPH(self) -> IGraph:
        edges = np.column_stack([self.source, self.target])
        g = IGraph(n=self.natoms, edges=edges, directed=False)
        return g.as_undirected()  # make sure it is undirected

    @cached_property
    def __COLOR(self) -> np.ndarray:
        z = np.char.mod("%d-", self.numbers)
        cn = np.char.mod("%d-", self.CN)
        return np.char.add(z, cn)

    def get_igcolor(self) -> np.ndarray:
        return self.__COLOR

    def get_igraph(self) -> IGraph:
        """Return the igraph object which only edges included."""
        return self.__IGRAPH

    @cached_property
    def nsymmetry(self) -> int:
        return self.__IGRAPH.count_automorphisms_vf2(self.numbers)

    def get_hop_distance(self, k: list[int] | np.ndarray) -> np.ndarray:
        k = self.get_index(k, self.natoms)
        d: list = self.__IGRAPH.distances(k)
        return np.asarray(d).min(axis=0)

    def get_chordless_cycles(
        self,
        batch_nbr_order: int = 3,
        batch: ArrayLike | list[int | bool] | int | None = None,
        max_ncore: Literal[3, 4, 5, 6] = 3,
    ) -> np.ndarray:
        """Got chordless cycles mask array for specific core numbers.

        Args:
            batch_nbr_order (int, optional): the max number of neighbors
                of a site. Defaults to 3.
            batch (np.ndarray | list[int | bool] | None, optional):
                the batch atoms of calculation. Default to None.
            max_ncore (Literal[3, 4, 5, 6], optional): The maximum
                number of core atoms in a site. Defaults to 3.

        Returns:
            np.ndarray: (n_site, n_atoms) boolean matrix
        """
        return chordless_cycles(
            igraph=self.__IGRAPH,
            batch_nbr_order=batch_nbr_order,
            max_ncore=max_ncore,
            batch=batch,
        )

    def get_match_mode(
        self,
        pattern: Self,  # big graph
        algorithm: Literal["lad", "vf2"] = "lad",
        return_match_target: bool = True,
        only_number_color: bool = False,
        only_count: bool = False,
    ) -> int | None | np.ndarray:
        """Match two pattern by subisomorphisms algorithm.

        Args:
            pattern (AtomsPattern): the target pattern(small).
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
        assert not self.check_induced_graph(), (
            "Only non-CN instance of BondGraph can call this method."
        )
        assert pattern.check_induced_graph(), (
            "Only has-CN instance can be set as pattern."
        )
        return match(
            pattern=pattern,
            pattern4match=self,
            algorithm=algorithm,
            return_match_target=return_match_target,
            only_number_color=only_number_color,
            only_count=only_count,
        )

    ###################################################
    # Some functions by `scipy.sparse`
    @cached_property
    def __CN_MATRIX(self) -> np.ndarray:
        m = self.MATRIX.astype(bool)
        m = sp.csr_array((m + m.T).astype(int))
        return np.asarray(m.sum(axis=1)).astype(int)

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

    def check_induced_graph(self) -> bool:
        if self.coordination is None:
            return False
        else:
            v0 = self.__CN_MATRIX
            v1 = self.coordination
            return bool(np.any(v0 != v1))

    def _get_neighbors_numpy(self, i: int) -> np.ndarray:
        """Get the first neighbors of index `i`."""
        idx1 = self.source[self.target == i]
        idx2 = self.target[self.source == i]
        return np.unique(np.append(idx1, idx2))

    def _get_neighbor_igraph(self, i: int | list[int]) -> np.ndarray:
        if isinstance(i, int):
            i = [i]
        result: list[list[int]] = self.__IGRAPH.neighborhood(
            i,
            order=1,
            mode="all",
            mindist=0,
        )
        if len(result) == 0:
            return np.array([])
        else:
            return np.unique(np.concatenate(result))

    def get_neighbors(
        self,
        i: int | list[int] | np.ndarray,
        exclude_i: bool = False,
    ) -> np.ndarray:
        """Get the first neighbors of index `i`."""
        if isinstance(i, np.ndarray):
            i = i.astype(int).tolist()
        nbrs = self._get_neighbor_igraph(i)  # type: ignore
        if exclude_i:
            nbrs = np.setdiff1d(nbrs, np.array(i))
        return nbrs

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

    ###################################################
    # Some functions for `weisfeiler lehman algorithm`
    def _weisfeiler_lehman_step(
        self,
        digest_size: int = DEFAULT_WH_HASH_SIZE,
        atomcolor: list[str | int] | np.ndarray | None = None,
    ) -> np.ndarray | list[str]:
        """Return hash string for each vertex by weisfeiler lehman algorithm.

        Reference:
            https://arxiv.org/pdf/1707.05005.pdf
            http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
            https://www.cnblogs.com/cyz666/p/9167507.html

        Args:
            digest_size (int, optional): The size of output str. Default to 6.
            atomcolor (list[str] | list[ int] | np.ndarray | None, optional):
                The vertex color in interger or string type. Defaults to None.
                If it is None, the value will be set as atomic number.

        Returns:
            np.ndarray | list[str]: the hash string for each vertex.
        """
        if atomcolor is None:
            atomcolor = self.numbers
        igcolor = np.asarray(atomcolor, dtype=str)
        assert len(igcolor) == self.natoms
        return np.asarray(
            [
                hash_string(
                    label
                    + "".join(np.sort(igcolor[self._get_neighbors_numpy(i)])),
                    digest_size=digest_size,
                )
                for i, label in enumerate(igcolor)
            ]
        )

    @pydantic.validate_call
    def get_weisfeiler_lehman_hashes(
        self,
        hash_depth: pydantic.PositiveInt = DEFAULT_WH_HASH_DEPTH,
        digest_size: pydantic.PositiveInt = DEFAULT_WH_HASH_SIZE,
    ) -> list[str]:
        """Return hash value for each atom."""
        labels = self.__COLOR
        for _ in range(hash_depth):
            labels = self._weisfeiler_lehman_step(
                atomcolor=np.asarray(labels),
                digest_size=digest_size,
            )
        result = [str(i) for i in labels]
        object.__setattr__(self, "hashes", result)
        assert self.hashes is not None
        return self.hashes

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

    ###################################################
    # from/to RDKit
    def to_rdmol(self, **kw) -> RDMol:
        return get_rdmol(
            numbers=self.numbers,
            source=self.source,
            target=self.target,
            order=self.order if self.order is not None else None,
            infer_order=False,
            charge=0,
            **kw,
        )

    @classmethod
    def from_rdmol(cls, rdmol: RDMol, **kw) -> Self:
        raise NotImplementedError

    ###################################################
    # from/to igraph
    def to_igraph(self, **kw) -> IGraph:
        df_atoms = DataFrame({"numbers": self.numbers})
        df_bonds = DataFrame({"source": self.source})
        df_bonds["target"] = self.target
        if self.order is not None:
            df_bonds["order"] = self.order
        if self.distance is not None:
            df_bonds["distance"] = self.distance
        return IGraph.DataFrame(df_bonds, False, df_atoms, True)

    @classmethod
    def from_igraph(cls, graph: IGraph, **kw) -> Self:
        raise NotImplementedError

    ###################################################
    # from/to igraph
    # def to_networkx(self, **kw) -> RDMol:
    #     return ...
    # @classmethod
    # def from_networkx(cls, rdmol: RDMol) -> Self:
    #     raise NotImplementedError
    # ###################################################
    # # from/to igraph
    # def to_rustworkx(self, **kw) -> RDMol:
    #     return ...
    # @classmethod
    # def from_rustworkx(cls, rdmol: RDMol) -> Self:
    #     raise NotImplementedError
    ###################################################
    # from/to igraph
    def to_pygdata(self, **kw) -> RDMol:
        raise NotImplementedError

    @classmethod
    def from_pygdata(cls, rdmol: RDMol) -> Self:
        raise NotImplementedError


def chordless_cycles(
    igraph: IGraph,
    max_ncore: Literal[3, 4, 5, 6] = 3,
    batch: ArrayLike | list[int | bool] | int | None = None,
    batch_nbr_order: int = 3,
) -> np.ndarray:
    """Got chordless cycles mask array for specific core numbers.

    Args:
        igraph (igraph.Graph): The graph to calculate.
        max_ncore (Literal[3, 4, 5, 6], optional): The maximum
            number of core atoms in a site. Defaults to 3.
        batch (np.ndarray | list[int | bool] | None, optional):
            the batch atoms of calculation. Default to None.
        batch_nbr_order (int, optional): The order of neighborhood.
            Defaults to 3.

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
    nbr = igraph.neighborhood(batch, order=batch_nbr_order, mindist=0)
    nbr = np.unique(reduce(np.append, nbr))
    g: IGraph = _subgraph_edges(igraph, nbr)
    if not g.is_simple():
        g = g.simplify()
    subisomor = partial(g.get_subisomorphisms_lad, induced=True)
    result: dict[int, np.ndarray] = {}
    for x in subisomor(IGraph(3, [(0, 1), (1, 2), (2, 0)])):
        i = hash(tuple(sorted(x)))
        if i not in result:
            result[i] = np.isin(_batch, x)
    for n, edges in [
        (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
    ]:
        if max_ncore >= n:
            for x in subisomor(IGraph(n, edges)):
                i = hash(tuple(sorted(x)))
                if i not in result:
                    bonds = igraph.es.select(_within=x)
                    if len(bonds) == n:
                        result[i] = np.isin(_batch, x)
    return np.asarray([i for i in result.values()])


def _subgraph_edges(
    igraph: IGraph,
    active: np.ndarray | list[int] | list[bool] | None = None,
) -> IGraph:
    """Return the edge subgraph only active bonds included."""
    active, n = np.asarray(active), igraph.vcount()
    if active.dtype in (bool, np.bool_):
        active = np.arange(n)[active]
    active = np.asarray(active, dtype=int)
    assert np.all(np.isin(active, np.arange(n)))
    active_bonds = igraph.es.select(_within=active)
    return igraph.subgraph_edges(active_bonds, delete_vertices=False)


def match(
    pattern: BondGraph,
    pattern4match: BondGraph,  # big graph
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
    cls: type = BondGraph
    msg = "The `{:s}` must be a `BondGraph` instance."
    assert isinstance(pattern, cls), msg.format("pattern")
    assert isinstance(pattern4match, cls), msg.format("pattern4match")
    assert algorithm in ["lad", "vf2"], "algorithm must be lad or vf2."

    graph: IGraph = getattr(pattern4match, f"_{cls.__name__}__IGRAPH")
    graph_small: IGraph = getattr(pattern, f"_{cls.__name__}__IGRAPH")
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


#######################################################################
#                                   Test
#######################################################################


def test_BondGraph() -> None:
    assert len(BondGraph.__abstractmethods__) == 0, (
        BondGraph.__abstractmethods__,
        BondGraph.__name__,
    )
