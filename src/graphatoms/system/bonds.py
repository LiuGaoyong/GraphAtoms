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
from graphatoms.geometry.itlee import itlee_check_permutation
from graphatoms.system.atoms import Matter
from graphatoms.utils.bytestool import hash_string
from graphatoms.utils.rdutils import get_rdmol

__all__ = ["BondGraph"]


class BondGraph(Matter, OurFrozenModel):
    coordination: Annotated[NDArray, numpy_validator("uint8")] | None = None
    pair: Annotated[NDArray, numpy_validator("int32", (-1, 2))] | None = None
    hashes: list[str] | None = None

    @pydantic.model_validator(mode="after")
    def __check_atoms_and_bonds(self) -> Self:
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
            assert self.__IGRAPH.is_simple(), "The graph should be simple."
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

    @property
    def CN(self) -> np.ndarray:
        if self.coordination is not None:
            return self.coordination
        else:
            return self.CN_MATRIX

    ###################################################
    # Some functions by `igraph`
    @cached_property
    def __IGRAPH(self) -> IGraph:
        edges = np.column_stack([self.source, self.target])
        g = IGraph(n=self.natoms, edges=edges, directed=True)
        # use directed graph for compatibility of full neighbor list
        assert not g.has_multiple(), "This graph has multiple edges."
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

    def bond_difference(
        self, other: "BondGraph"
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Get the bond difference between two bond graphs.

        Args:
            other (BondGraph): the other bond graph.

        Returns two lists:
            first: the break bonds.
            second: the make bonds.
        """
        g1, g2 = self.__IGRAPH, other.get_igraph()
        break_bonds = [e.tuple for e in g1.difference(g2).es]
        make_bonds = [e.tuple for e in g2.difference(g1).es]
        return break_bonds, make_bonds

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
        pattern: "BondGraph",
        algorithm: Literal["lad", "vf2"] = "lad",
        return_match_target: bool = True,
        only_number_color: bool = False,
        only_count: bool = False,
        *,
        tolerance: float | None = None,
        other_tolerance: float | None = None,
        verify_exact: bool = True,
        relax_coordination: bool = True,
        allow_reflection: bool = False,
        return_metrics: bool = False,
        max_branches: int = 200_000,
    ) -> int | None | np.ndarray | list[tuple[np.ndarray, float]]:
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
            tolerance (float | None, optional): ITLEE L2 tolerance in
                Angstrom. ``None`` or ``0`` keeps the exact behaviour;
                negative values raise ``ValueError``.
            other_tolerance (float | None, optional): second tolerance;
                the effective value is ``min(tolerance, other_tolerance)``.
            verify_exact (bool, optional): whether to verify exact
                candidates with ITLEE before falling back to the
                element-only colour pass.
            relax_coordination (bool, optional): retry with element-only
                colours when the CN-coloured pass has no valid candidate.
            allow_reflection (bool, optional): allow improper rotations.
            return_metrics (bool, optional): return ``(mapping, rmsd)``
                pairs instead of the legacy numpy match-mode array.  The
                mapping in each pair is a ``(n_pattern,)`` array of target
                indices.
            max_branches (int, optional): ITLEE search branch cap.

        Returns:
            int | None | np.ndarray | list[tuple[np.ndarray, float]]:
                If only_count is True, the result is the integer.
                Otherwise, the result is None when no matching found.
                    If matched, a (N_match, N_pattern)
                    shape numpy.ndarray will be outputed.
                If ``return_metrics`` is True, a list of
                ``(mapping, rmsd)`` tuples is returned instead.
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
            tolerance=tolerance,
            other_tolerance=other_tolerance,
            verify_exact=verify_exact,
            relax_coordination=relax_coordination,
            allow_reflection=allow_reflection,
            return_metrics=return_metrics,
            max_branches=max_branches,
        )

    ###################################################
    # Some functions by `scipy.sparse`
    @cached_property
    def CN_MATRIX(self) -> np.ndarray:
        m = self.MATRIX.astype(bool)
        m = sp.csr_array((m + m.T).astype(int))
        return np.asarray(m.sum(axis=1)).astype(int)

    @cached_property
    def MATRIX(self) -> sp.csr_array:
        order = np.ones(self.nbonds, bool)
        # if ........:
        #     # scipy.sparse does not support dtype float16(self.order.dtype)
        #     # so we use single float numbers(float32) in the return statement
        #     order = np.asarray(self.order, dtype="f4")
        return sp.csr_array(
            (order, (self.source, self.target)),
            shape=(self.natoms, self.natoms),
        )

    def check_induced_graph(self) -> bool:
        if self.coordination is None:
            return False
        else:
            v0 = self.CN_MATRIX
            v1 = self.coordination
            return bool(np.any(v0 != v1))

    def _get_neighbors_numpy(self, i: int) -> np.ndarray:
        """Get the first neighbors of index `i`."""
        idx1 = self.source[self.target == i]
        idx2 = self.target[self.source == i]
        return np.unique(np.append(idx1, idx2))

    def _get_neighbor_igraph(
        self,
        i: int | list[int] | np.ndarray,
        *args,
        hop: int = 1,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(i, int):
            i = [i]
        result: list[list[int]] = self.__IGRAPH.neighborhood(
            i,
            order=int(hop),
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
        nbrs = self._get_neighbor_igraph(i, hop=1)  # type: ignore
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
        """Return the connected components sorted by size in descending order.

        Each connected component is represented as a NumPy array containing the
        indices where the corresponding component label appears. The labels are
        taken from the internal connected-components result and range from ``0``
        to ``self.connected_components_number - 1``.

        Returns:
            list[np.ndarray]: A list of unique index arrays, one for each
                connected component, sorted from largest to smallest.
        """
        labels = self.__get_connected_components[1]
        n: int = self.connected_components_number
        result = [np.unique(np.where(labels == i)) for i in range(n)]
        return sorted(result, reverse=True, key=lambda x: len(x))

    @cached_property
    def connected_components_biggest(self) -> np.ndarray:
        return self.connected_components[0]

    @cached_property
    def is_connected(self) -> bool:
        return self.connected_components_number == 1

    ###################################################
    # Some functions for `weisfeiler lehman algorithm`
    def _weisfeiler_lehman_step(
        self,
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
                )
                for i, label in enumerate(igcolor)
            ]
        )

    @pydantic.validate_call
    def get_weisfeiler_lehman_hashes(
        self,
        hash_depth: pydantic.PositiveInt = 3,
    ) -> list[str]:
        """Return hash value for each atom."""
        labels = self.__COLOR
        for _ in range(hash_depth):
            labels = self._weisfeiler_lehman_step(
                atomcolor=np.asarray(labels),
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
            )
        )

    ###################################################
    # from/to RDKit
    def to_rdmol(self, **kw) -> RDMol:
        return get_rdmol(
            numbers=self.numbers,
            source=self.source,
            target=self.target,
            # order=self.order if self.order is not None else None,
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
        # if self.order is not None:
        #     df_bonds["order"] = self.order
        # if self.distance is not None:
        #     df_bonds["distance"] = self.distance
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


def _get_subisomorphisms(
    graph: IGraph,
    graph_small: IGraph,
    color: np.ndarray,
    color_small: np.ndarray,
    algorithm: Literal["lad", "vf2"],
) -> list[list[int]]:
    """Return subgraph isomorphism mappings for the given colours."""
    if not np.all(np.isin(color_small, color)):
        return []
    if algorithm == "lad":
        return graph.get_subisomorphisms_lad(
            pattern=graph_small,
            domains=[
                np.argwhere(color == color_small[i]).flatten()
                for i in range(len(color_small))
            ],
            induced=True,
        )
    elif algorithm == "vf2":
        return graph.get_subisomorphisms_vf2(
            other=graph_small,
            color1=color,
            color2=color_small,
        )
    else:
        raise RuntimeError("Impossible !!!")


def _filter_matches_by_itlee(
    pattern: BondGraph,
    target: BondGraph,
    matches: list[list[int]] | np.ndarray,
    tolerance: float,
    *,
    allow_reflection: bool,
) -> list[tuple[np.ndarray, float]]:
    """Keep graph embeddings whose geometry passes the ITLEE criterion."""
    if not hasattr(pattern, "positions") or not hasattr(target, "positions"):
        raise TypeError(
            "tolerance requires SysGraph-like objects with positions."
        )
    n_pattern = len(pattern.numbers)
    rows = np.asarray(matches, dtype=int)
    if rows.size == 0:
        return []
    rows = rows.reshape(-1, n_pattern)
    result: list[tuple[np.ndarray, float]] = []
    for row in rows:
        checked = itlee_check_permutation(
            pattern.positions,
            target.positions[row],
            np.arange(n_pattern, dtype=int),
            tolerance,
            ref_colors=pattern.numbers,
            cur_colors=target.numbers[row],
            allow_reflection=allow_reflection,
        )
        if checked is not None:
            result.append((row.copy(), checked.rmsd))
    return result


def _build_match_result(
    matches: list[np.ndarray],
    n_target: int,
    n_pattern: int,
    *,
    return_match_target: bool,
) -> np.ndarray:
    rows = np.asarray(matches, dtype=int)
    if rows.size == 0:
        return np.empty((0, n_target if return_match_target else n_pattern))
    rows = rows.reshape(-1, n_pattern)
    if not return_match_target:
        return rows
    result = -np.ones(shape=(len(rows), n_target), dtype=int)
    result[
        np.column_stack([np.arange(len(rows))] * n_pattern),
        rows,
    ] = np.arange(n_pattern, dtype=int)
    return result


def match(
    pattern: BondGraph,
    pattern4match: BondGraph,  # big graph
    algorithm: Literal["lad", "vf2"] = "lad",
    return_match_target: bool = True,
    only_number_color: bool = False,
    only_count: bool = False,
    *,
    tolerance: float | None = None,
    other_tolerance: float | None = None,
    verify_exact: bool = True,
    relax_coordination: bool = True,
    allow_reflection: bool = False,
    return_metrics: bool = False,
    max_branches: int = 200_000,
) -> int | None | np.ndarray | list[tuple[np.ndarray, float]]:
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
        tolerance (float | None, optional): ITLEE L2 tolerance in Angstrom.
            ``None`` or ``0`` keeps the exact behaviour; negative values
            raise ``ValueError``.
        other_tolerance (float | None, optional): second tolerance; the
            effective value is ``min(tolerance, other_tolerance)``.
        verify_exact (bool, optional): verify exact candidates with ITLEE
            before falling back to the element-only colour pass.
        relax_coordination (bool, optional): retry with element-only
            colours when the CN-coloured pass has no valid candidate.
        allow_reflection (bool, optional): allow improper rotations.
        return_metrics (bool, optional): return ``(mapping, rmsd)`` pairs;
            the mapping is a ``(n_pattern,)`` array of target indices.
        max_branches (int, optional): ITLEE search branch cap.  It is
            accepted for API symmetry; the current checker does not search.

    Returns:
        int | None | np.ndarray | list[tuple[np.ndarray, float]]:
            If only_count is True, the result is the integer.
            Otherwise, the result is None when no matching found.
                If matched, a (N_match, N_pattern)
                shape numpy.ndarray will be outputed.
            If ``return_metrics`` is True, a list of ``(mapping, rmsd)``
            tuples is returned instead.
    """
    cls: type = BondGraph
    msg = "The `{:s}` must be a `BondGraph` instance."
    assert isinstance(pattern, cls), msg.format("pattern")
    assert isinstance(pattern4match, cls), msg.format("pattern4match")
    assert algorithm in ["lad", "vf2"], "algorithm must be lad or vf2."

    graph: IGraph = getattr(pattern4match, f"_{cls.__name__}__IGRAPH")
    graph_small: IGraph = getattr(pattern, f"_{cls.__name__}__IGRAPH")
    n_pattern = len(pattern.numbers)
    n_target = len(pattern4match.numbers)

    def exact_colors() -> tuple[np.ndarray, np.ndarray]:
        if only_number_color:
            return pattern4match.numbers, pattern.numbers
        color = getattr(pattern4match, f"_{cls.__name__}__COLOR")
        color_small = getattr(pattern, f"_{cls.__name__}__COLOR")
        return (
            np.vectorize(hash)(color),
            np.vectorize(hash)(color_small),
        )

    def relaxed_colors() -> tuple[np.ndarray, np.ndarray]:
        return pattern4match.numbers, pattern.numbers

    assert n_target >= n_pattern, (
        f"Pattern(N={n_pattern}) is too big !!! Target(N={n_target})."
    )

    delta: float | None = None
    if tolerance is not None:
        if float(tolerance) < 0:
            raise ValueError("tolerance must be non-negative.")
        if float(tolerance) > 0:
            delta = (
                float(tolerance)
                if other_tolerance is None
                else min(float(tolerance), float(other_tolerance))
            )
            if delta <= 0:
                raise ValueError("tolerance must be positive.")

    if delta is None:
        color, color_small = exact_colors()
        out = _get_subisomorphisms(
            graph,
            graph_small,
            color,
            color_small,
            algorithm,
        )
        if len(out) == 0:
            return None
        result = _build_match_result(
            [np.asarray(row, dtype=int) for row in out],
            n_target,
            n_pattern,
            return_match_target=return_match_target,
        )
        return matchmode2nmatch(result) if only_count else result

    color_pairs: list[tuple[bool, np.ndarray, np.ndarray]] = [
        (True, *exact_colors())
    ]
    if (not only_number_color) and relax_coordination:
        color_pairs.append((False, *relaxed_colors()))

    metrics: list[tuple[np.ndarray, float]] = []
    for is_exact, color, color_small in color_pairs:
        out = _get_subisomorphisms(
            graph,
            graph_small,
            color,
            color_small,
            algorithm,
        )
        if len(out) == 0:
            continue
        if (not is_exact) or verify_exact or return_metrics:
            metrics = _filter_matches_by_itlee(
                pattern,
                pattern4match,
                out,
                delta,
                allow_reflection=allow_reflection,
            )
        else:
            metrics = [
                (np.asarray(row, dtype=int), float("nan")) for row in out
            ]
        if len(metrics) > 0:
            break

    if len(metrics) == 0:
        return None
    if return_metrics:
        return metrics
    matches = [mapping for mapping, _ in metrics]
    result = _build_match_result(
        matches,
        n_target,
        n_pattern,
        return_match_target=return_match_target,
    )
    return matchmode2nmatch(result) if only_count else result


def matchmode2nmatch(matchmode: np.ndarray) -> int:
    """Return the number of matching for matchmode.

    original code:
        u = np.sort(matchmode, axis=1)
        return max([len(np.unique(row)) for row in u.T])

    """
    assert matchmode.ndim == 2, "matchmode must be 2D."
    if matchmode.size == 0:
        return 0
    s = np.sort(matchmode, axis=1)
    s = np.sort(s, axis=0)
    eq = s[1:] == s[:-1]
    if np.issubdtype(s.dtype, np.floating):
        eq |= np.isnan(s[1:]) & np.isnan(s[:-1])
    return int((1 + (~eq).sum(axis=0)).max())
