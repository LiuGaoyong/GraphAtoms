import json
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Annotated, Any, Self, override

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from igraph import Graph as IGraph
from pandas import DataFrame
from pydantic import model_validator
from rdkit import Chem
from scipy import sparse as sp

from ...dataclasses import NDArray, OurBaseModel, numpy_validator
from ...geometry import bond_list
from ...utils import rdutils
from ..atoms import Box, Energetics, Matter, Structure
from ._bonds import _BOND_ATTRS, BondGraph, _subgraph_edges
from ._gasMixin import GasMixin


class AtomTag(OurBaseModel):
    move_fix_tag: Annotated[NDArray, numpy_validator("int8")] | None = None
    is_adsorbate: Annotated[NDArray, numpy_validator(bool)] | None = None
    is_outer: Annotated[NDArray, numpy_validator(bool)] | None = None

    @model_validator(mode="after")
    def __check_atoms(self) -> Self:
        for k in AtomTag.__pydantic_fields__.keys():
            v = getattr(self, k, None)
            if v is not None:
                assert len(v) == self.natoms, (
                    f"Invalid shape for `{k}`: Len({k})="
                    f"{len(v)} but natoms={self.natoms}."
                )
        if self.move_fix_tag is not None:
            assert self.isfix.sum() != 0, "`isfix` sum == 0"
            assert self.iscore.sum() != 0, "`iscore` sum == 0"
            assert self.isfix.sum != self.natoms, "`ismoved` sum == 0"
        return self

    @override
    def _string(self) -> str:
        lst: list[str] = []
        if self.move_fix_tag is not None:
            lst.extend(
                [
                    f"NCORE={self.ncore}",
                    f"NMOVED={self.nmoved}",
                    f"NFIX={self.nfix}",
                ]
            )
        if self.is_outer is not None:
            lst.append(f"NOUTER={np.sum(self.is_outer)}")
        if self.is_adsorbate is not None:
            lst.append(f"NADS={np.sum(self.is_adsorbate)}")
        else:
            lst.append("NADS=0")
        return ",".join(lst)

    @cached_property
    @abstractmethod
    def natoms(self) -> int: ...

    @property
    def nfix(self) -> int:
        return int(self.isfix.sum())

    @property
    def isfix(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag < 0  # type: ignore

    @property
    def ncore(self) -> int:
        return int(self.iscore.sum())

    @property
    def iscore(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == 0

    @property
    def nmoved(self) -> int:
        return self.natoms - self.nfix

    @property
    def isfirstmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == 1

    @property
    def islastmoved(self) -> np.ndarray:
        if self.move_fix_tag is None:
            raise KeyError("The `move_fix_tag` is None.")
        return self.move_fix_tag == np.max(self.move_fix_tag)


class SysGraph(BondGraph, Structure, AtomTag, GasMixin):
    """Frozen dataclass for chemical graph representation.

    Attributes:
        Z (numbers): Atomic numbers.
        R (positions): Atomic positions.
        CN: Coordination numbers.
        P (pair): Bond pair indices.
        S (shift): Bond shift indices.
        D (distance): Bond distances.
        B (box): Simulation box (periodic systems).
        E (energy): System energy.
        FQ (frequencies): Vibrational frequencies.
    """

    @override
    @classmethod
    def from_str(cls, data: str, **kw) -> Self:
        return cls.from_dict(json.loads(data), **kw)

    @override
    def _string(self) -> str:
        lst: list[str] = [
            Structure._string(self),
            BondGraph._string(self),
        ]
        if self.is_gas:
            lst.insert(0, GasMixin._string(self))
        lst.append(AtomTag._string(self))
        return ",".join(lst)

    @cached_property
    def smarts(self) -> str:
        """Get the SMILES string of this object."""
        fml = self.symbols.get_chemical_formula("metal", True)
        if self.is_adsorbate is None or np.sum(self.is_adsorbate) == 0:
            return f"{fml}-{self.hash}"
        else:
            idxs = np.concatenate(np.where(self.is_adsorbate))
            idxs = np.unique(np.append(idxs, self.get_neighbors(idxs)))
            idxs_lst: list[int] = idxs.tolist()
            igraph: IGraph = self.get_igraph()
            for idx in idxs_lst:
                this_idxs = igraph.get_shortest_paths(idx, idxs_lst)
                this_idxs = np.concatenate(this_idxs)
                idxs = np.append(idxs, this_idxs)
            subgraph: IGraph = _subgraph_edges(igraph, idxs)
            adj = sp.coo_matrix(subgraph.get_adjacency_sparse())
            source, target = adj.coords
            rdmol: rdutils.RDMol = rdutils.get_rdmol(
                # numbers=np.where(self.is_adsorbate, self.numbers, 0),
                numbers=self.numbers,
                source=source,
                target=target,
            )
            fragments = Chem.GetMolFrags(rdmol, asMols=True)  # type: ignore
            largest_frag = max(
                fragments,
                default=rdmol,
                key=lambda m: m.GetNumAtoms(),
            )
            return Chem.MolToSmarts(largest_frag)  # type: ignore

    ###################################################
    # from/to dict

    @override
    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        parse_bonds: Mapping[str, Any] | None = None,
        parse_bonds_distance: bool = False,
        parse_bonds_order: bool = False,
        **kwargs,
    ) -> Self:
        obj = super().from_dict(data, **kwargs)
        dct: dict[str, np.ndarray | float] = obj.to_dict()

        if parse_bonds is not None and len(parse_bonds) == 0:
            parse_bonds = None
        ndata: int = len(dct)

        # parse bonds pair index
        if obj.pair is None and parse_bonds is not None:
            atoms = Atoms(
                numbers=obj.numbers,
                positions=obj.positions,
                pbc=obj.is_periodic,
                cell=obj.ase_cell,
            )
            m = sp.coo_matrix(
                bond_list(
                    atoms=atoms,
                    infer_order=parse_bonds_order,
                    **parse_bonds,
                )
            )
            dct["pair"] = pair = np.column_stack(m.coords)
            if np.any(pair[:, 0] == pair[:, 1]):
                raise RuntimeError(
                    "The `pair` should not contain self-loop bonds. It "
                    "is typically caused by the structure is periodic "
                    "and contains too less atoms (bulk? surface? etc). "
                )
            order = np.asarray(m.data, dtype=int)
            if not np.all(order == 1):
                dct["order"] = order
        else:
            pair = obj.P
        assert isinstance(pair, np.ndarray)
        if pair.size == 0:
            pair = None

        if pair is not None:
            i, j = np.transpose(pair)
            # parse bonds distance
            if obj.distance is None and parse_bonds_distance:
                v, c = obj.positions[i] - obj.positions[j], obj.ase_cell
                _, dct["distance"] = find_mic(v, c, obj.is_periodic)

        if len(dct) == ndata:
            return obj
        else:
            return super().from_dict(dct, **kwargs)

    @override
    def to_dict(
        self,
        *,
        exclude_none: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_computed_fields: bool = True,
        numpy_ndarray_compatible: bool = True,
        numpy_convert_to_list: bool = False,
        exclude_bond_attibutes: bool = False,
        exclude_energetics: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", set())
        return super().to_dict(
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_computed_fields=exclude_computed_fields,
            exclude=(
                exclude
                | (
                    set()
                    if not exclude_energetics
                    else Energetics.__pydantic_fields__.keys()
                )
                | (set() if not exclude_bond_attibutes else set(_BOND_ATTRS))
            ),
            numpy_ndarray_compatible=numpy_ndarray_compatible,
            numpy_convert_to_list=numpy_convert_to_list,
            **kwargs,
        )

    # from/to ASE
    @classmethod
    def from_ase(
        cls,
        atoms: Atoms,
        parse_bonds: Mapping[str, Any] | None = {"method": "raw"},
        parse_bonds_distance: bool = False,
        parse_bonds_order: bool = False,
        **kwargs,
    ) -> Self:
        return cls.from_dict(
            Structure._ase2dct(atoms=atoms, **kwargs),
            parse_bonds_distance=parse_bonds_distance,
            parse_bonds_order=parse_bonds_order,
            parse_bonds=parse_bonds,
            **kwargs,
        )

    def to_ase(
        self,
        *,
        exclude_energetics: bool = False,
        exclude_bond_attibutes: bool = False,
        **kwargs,
    ) -> Atoms:
        return Atoms(
            numbers=self.numbers,
            positions=self.positions,
            pbc=self.is_periodic,
            cell=self.ase_cell,
            info=self.to_dict(
                exclude_none=True,
                exclude_computed_fields=True,
                exclude_bond_attibutes=exclude_bond_attibutes,
                exclude_energetics=exclude_energetics,
                exclude=(
                    {"positions"}
                    | Box.__pydantic_fields__.keys()
                    | Matter.__pydantic_fields__.keys()
                ),
                numpy_ndarray_compatible=True,
                numpy_convert_to_list=False,
            ),
        )

    ###################################################
    # from/to igraph
    @classmethod
    @override
    def from_igraph(cls, graph: IGraph, **kwargs) -> Self:
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

    def to_igraph(self, *args, **kwargs) -> IGraph:
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
        G = IGraph.DataFrame(df_bonds, False, df_atoms, True)
        for k in (
            Box.__pydantic_fields__.keys()
            | Energetics.__pydantic_fields__.keys()
        ):
            v = getattr(self, k)
            if v is not None:
                G[k] = v
        return G

    @override
    @classmethod
    def SUPPORTED_CONVERT_FORMATS(cls) -> Sequence[str]:
        # result = ("ase", "pygdata", "igraph", "networkx", "rustworkx")
        return tuple(super().SUPPORTED_CONVERT_FORMATS()) + ("ase", "igraph")


# class NetworkXConverter(IGraphConverter):
#     def to_networkx(self, *args, **kwargs) -> NXGraph:
#         return self.to_igraph(*args, **kwargs).to_networkx()

#     @classmethod
#     def from_networkx(cls, graph: NXGraph, **kwargs) -> Self:
#         return cls.from_igraph(graph=Graph.from_networkx(graph), **kwargs)


# class RustworkXConverter(IGraphConverter):
#     def to_rustworkx(self) -> RXGraph:
#         graph = self.to_igraph()
#         # Ref: https://github.com/igraph/python-igraph/blob/main/src/igraph/io/libraries.py#L1-L70
#         G = RXGraph(
#             multigraph=False,
#             attrs={x: graph[x] for x in graph.attributes()},
#         )
#         G.add_nodes_from([v.attributes() for v in graph.vs])
#         G.add_edges_from((e.source, e.target, e.attributes())
#                           for e in graph.es)
#         return G

#     @classmethod
#     def from_rustworkx(cls, graph: RXGraph, **kwargs) -> Self:
#         df_nodes = DataFrame(graph.nodes())
#         source, target = np.asarray(graph.edge_list(), dtype=int).T
#         df_edges0 = DataFrame({"i": source, "j": target})
#         df_edges1 = DataFrame(graph.edges())
#         df_edges = pd_concat(
#             [df_edges0, df_edges1],
#             ignore_index=True,
#             axis="columns",
#         )
#         df_edges.columns = list(df_edges0.columns) + list(df_edges1.columns)
#         igraph = Graph.DataFrame(df_edges, False, df_nodes, True)
#         for k, v in graph.attrs.items():  # attr is dict type
#             if k in cls.__pydantic_fields__ and v is not None:
#                 igraph[k] = v
#         return cls.from_igraph(graph=igraph, **kwargs)


# class PyTorchGeometricConverter(Base):
#     @classmethod
#     def from_pygdata(cls, data: DataPyG, **kwargs) -> Self:
#         assert data.pos is not None and "numbers" in data.keys()
#         dct: dict[str, Any] = {
#             "positions": data.pos.numpy(force=True),
#             "pair": (
#                 data.edge_index.numpy(force=True).T
#                 if data.edge_index is not None
#                 else None
#             ),
#         } | {
#             k: data[k].numpy(force=True)
#             if isinstance(data[k], torch.Tensor)
#             else data[k]
#             for k in set(cls.__pydantic_fields__.keys()) & set(data.keys())
#             if k not in ("positions", "pair")
#         }
#         return cls.from_dict(dct, **kwargs)

#     def to_pygdata(self) -> DataPyG:
#         #  UserWarning: The given NumPy array is not writable, and
#         #   PyTorch does not support non-writable tensors. This means
#         #   writing to this tensor will result in undefined behavior.
#         #   You may want to copy the array to protect its data or make it
#         #   writable before converting it to a tensor. This type of warning
#         #   will be suppressed for the rest of this program. (Triggered
#         #       internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=UserWarning)
#             conn = np.column_stack([self.source, self.target])
#             result = DataPyG(
#                 pos=torch.from_numpy(self.positions),
#                 edge_index=torch.from_numpy(conn.astype(int).T),
#             )
#             for k, v in self.to_dict(
#                 mode="python",
#                 exclude_none=True,
#                 exclude_computed_fields=True,
#                 exclude={"positions", "pair"},
#                 numpy_ndarray_compatible=True,
#                 numpy_convert_to_list=False,
#             ).items():
#                 if isinstance(v, np.ndarray):
#                     dtype_name: str = v.dtype.name
#                     if dtype_name.startswith("uint"):
#                         if dtype_name != "uint8":
#                             d = dtype_name[1:]
#                             v = v.astype(d)
#                     result[k] = torch.from_numpy(v)
#                 elif np.isscalar(v) or isinstance(v, list | tuple):
#                     result[k] = v
#                 else:
#                     raise TypeError(f"{k}(type={type(v)}: {v}")
#         result.validate(raise_on_error=True)
#         return result

#     def get_active_subgraph(self, k: ArrayLike) -> Self:
#         """Return the edge subgraph only active nodes included."""
#         data = self.to_pygdata()
#         na, nb = data.num_nodes, data.num_edges
#         if na is None or na == 0:
#             # raise KeyError(f"Got zero node PyG Data object: {data}")
#             return self
#         elif nb is None or nb == 0:
#             # raise KeyError(f"Got zero edge PyG Data object: {data}")
#             return self
#         elif data.edge_index is None:
#             return self
#         else:
#             i, j = data.edge_index.numpy(force=True)
#             subset = self.get_index(k=k, n=na)
#             mask = torch.from_numpy(np.isin(i, subset) & np.isin(j, subset))
#             subpygdata = data.edge_subgraph(mask)  # shallow copy
#             return self.from_pygdata(data=subpygdata)

#     def get_edge_subgraph(self, k: ArrayLike) -> Self:
#         data = self.to_pygdata()
#         n = data.num_edges
#         if n is None or n == 0:
#             # raise KeyError(f"Got zero edge PyG Data object: {data}")
#             return self
#         else:
#             subset = torch.from_numpy(self.get_mask_or_index(k=k, n=n))
#             subpygdata = data.edge_subgraph(subset)  # shallow copy
#             return self.from_pygdata(data=subpygdata)

#     def get_induced_subgraph(self, k: ArrayLike) -> Self:
#         data = self.to_pygdata()
#         n = data.num_nodes
#         if n is None or n == 0:
#             # raise KeyError(f"Got zero node PyG Data object: {data}")
#             return self
#         else:
#             subset = torch.from_numpy(self.get_mask_or_index(k=k, n=n))
#             subpygdata = data.subgraph(subset)  # shallow copy
#             return self.from_pygdata(data=subpygdata)

#     def get_k_hop_subgraph(self, k: ArrayLike, num_hops: int = 1) -> Self:
#         k_hop_neighbor = self.get_k_hop_neighbor(k, num_hops=num_hops)
#         if k_hop_neighbor.size == 0:
#             raise KeyError(
#                 "The number of nodes or edges is zero. Input: self="
#                 f"{repr(self)}, k={repr(k)}, num_hops={repr(num_hops)}."
#             )
#         else:
#             return self.get_k_hop_subgraph(k_hop_neighbor)

#     def get_k_hop_neighbor(self, k: ArrayLike,
#               num_hops: int = 1) -> np.ndarray:
#         data = self.to_pygdata()
#         n = data.num_nodes
#         if data.edge_index is None or n is None or n == 0:
#             # raise KeyError("Unsupported: PyG.Data without edge.")
#             # raise KeyError(f"Got zero node PyG Data object: {data}")
#             return np.asarray([], dtype=int)
#         else:
#             subset, _, _, _ = k_hop_subgraph(
#                 torch.from_numpy(self.get_index(k=k, n=n)),
#                 num_hops=num_hops,
#                 edge_index=data.edge_index,
#                 relabel_nodes=False,
#                 num_nodes=n,
#                 directed=False,
#             )
#             return subset.numpy(force=True).astype(int)


# if __name__ == "__main__":
def test_benchmark_convert() -> None:
    from collections import defaultdict
    from time import perf_counter

    from ase.cluster import Octahedron
    from pandas import DataFrame

    result = defaultdict(dict)
    for n in [8, 12, 16, 20, 25, 32]:
        # [344, 1156, 2736, 5340, 10425, 21856]
        obj = SysGraph.from_ase(Octahedron("Au", n))
        result[n]["natoms"] = len(obj.numbers)
        for mode in (
            "ASE",
            # "PyGData",
            "IGraph",
            # "RustworkX",
            # "NetworkX",
        ):
            t0 = perf_counter()
            for _ in range(10):
                _obj = obj.convert_to(mode.lower())  # type: ignore
                obj.convert_from(_obj, mode.lower())
            result[n][f"{mode}(ms)"] = (perf_counter() - t0) * 1000 / 10
    df = DataFrame(result).T
    df["natoms"] = df["natoms"].astype(int)
    print("\n", df)
    ###########################################################################
    #  "Only To"  ASE(ms)  PyGData(ms)  IGraph(ms)  RustworkX(ms)  NetworkX(ms)
    # 8      344  0.05555      0.21208     3.69529        5.84161      10.37565
    # 12    1156  0.08168      0.28180     5.46619       15.39528      30.21391
    # 16    2736  0.15297      0.57349    10.00264       29.72876      68.21798
    # 20    5340  0.27284      0.92302    15.50519       53.65532     126.72180
    # 25   10425  0.52049      1.56988    30.62596      127.98179     293.56670
    # 32   21856  1.07051      3.15097    61.55693      217.29759     534.46987
    #  ms/natoms  0.00007      0.00026     0.00463        0.01224       0.02626
    #  natoms/ms    13697         3900         215             81            40
    ###########################################################################
    #  "From/To"  ASE(ms)  PyGData(ms)  IGraph(ms)  RustworkX(ms)  NetworkX(ms)
    # 8      344  0.21315      0.39355     8.11748       13.46136      17.93142
    # 12    1156  0.19550      0.46889    19.78003       35.43563      55.59955
    # 16    2736  0.20679      0.63878    42.61782       75.81725     126.79931
    # 20    5340  0.21546      0.97900    80.73528      244.69746     313.01750
    # 25   10425  0.24309      1.57129   172.54592      287.59963     524.07426
    # 32   21856  0.32291      3.65045   339.27791      601.12843    1182.83410
    #  ms/natoms  0.00016      0.00038     0.01725        0.03307       0.05157
    #  natoms/ms     6364         2626          58             30            19
    ###########################################################################
