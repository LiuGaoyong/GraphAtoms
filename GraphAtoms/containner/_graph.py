import warnings
from collections.abc import Hashable
from functools import cached_property
from typing import Any, Literal, override

import numpy as np
import torch
from ase import Atoms
from igraph import Graph as IGraph
from networkit import Graph as NetworKitGraph
from networkx import Graph as NetworkXGraph
from pandas import DataFrame
from pandas import concat as pd_concat
from pydantic import model_validator, validate_call
from rustworkx import PyGraph as RustworkXGraph
from scipy import sparse as sp
from torch_geometric.data import Data as DataPyG
from typing_extensions import Self

from GraphAtoms.containner._atomic import ATOM_KEY, TOTAL_KEY, AtomicContainner
from GraphAtoms.containner._bonded import BOND_KEY, BondAttrMixin
from GraphAtoms.containner._mConnComp import GraphMixinConnectedComponents
from GraphAtoms.containner._mDataPyG import GraphMixinPyG
from GraphAtoms.containner._mFreeE import FreeEnergyMixin
from GraphAtoms.containner._mIGraph import GraphMixinIGraph
from GraphAtoms.containner._mNetworKit import GraphMixinNetworKit
from GraphAtoms.containner._mNetworkX import GraphMixinNetworkX
from GraphAtoms.containner._mRustworkX import GraphMixinRustworkX
from GraphAtoms.common.string import hash_string


class __KEY:
    ATOM = ATOM_KEY
    BOND = BOND_KEY
    GRAPH = TOTAL_KEY

    @property
    def _DICT(self) -> dict[str, str]:
        result: dict[str, str] = dict()
        result |= self.GRAPH._DICT
        result |= self.BOND._DICT
        result |= self.ATOM._DICT
        return result


GRAPH_KEY = __KEY()
_OBJ_TYPE_0 = Atoms | DataPyG | IGraph
_OBJ_TYPE_1 = NetworkXGraph | RustworkXGraph
_OBJ_TYPE = _OBJ_TYPE_0 | _OBJ_TYPE_1 | NetworKitGraph
_MODE_TYPE_1 = Literal["networkit", "networkx", "rustworkx"]
_MODE_TYPE_0 = Literal["ase", "pygdata", "igraph"]
_MODE_TYPE = _MODE_TYPE_0 | _MODE_TYPE_1


class GraphContainner(
    AtomicContainner,
    BondAttrMixin,
    GraphMixinConnectedComponents,
    GraphMixinIGraph,
    GraphMixinPyG,
    GraphMixinNetworkX,
    GraphMixinNetworKit,
    GraphMixinRustworkX,
    FreeEnergyMixin,
    Hashable,
):
    # In Base Class: `hash: str | None`
    hash: str = "None"  # type: ignore

    @override
    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif self.natoms != other.natoms or self.nbonds != other.nbonds:
            return False
        elif self.is_sub != other.is_sub:
            return False
        elif (
            not self.connected_components_number
            == other.connected_components_number
        ):
            return False
        elif not np.all(
            (self.connected_components_label)
            == (other.connected_components_label)
        ):
            return False
        elif not np.all(self.CN == other.CN):
            return False
        elif not AtomicContainner.__eq__(self, other):
            return False
        elif BondAttrMixin.__eq__(self, other):
            return True
        else:
            return self.__hashstring == other.__hashstring

    @override
    def __hash__(self) -> int:
        return hash(self.__hashstring)

    @cached_property
    def __hashstring(self) -> str:
        labels = sorted(self.get_weisfeiler_lehman_hashes())
        return hash_string(",".join(labels), digest_size=8)

    @model_validator(mode="after")
    def __reset_hashstring_and_check_keys(self) -> Self:
        object.__setattr__(self, TOTAL_KEY.HASH, self.__hashstring)
        _all_keys_set: set[str] = set(self.__pydantic_fields__.keys())
        assert _all_keys_set >= set(GRAPH_KEY._DICT.values())
        assert self.hash is not None
        return self

    @property
    def CN(self) -> np.ndarray:
        try:
            return super().CN
        except Exception:
            return super().CN_MATRIX

    @property
    @override
    def THERMO_ATOMS(self) -> Atoms:
        result = Atoms(self.Z, self.R)
        result.info[TOTAL_KEY.ENERGY] = self.energy
        result.info[TOTAL_KEY.FREQS] = self.frequencies
        result.info[TOTAL_KEY.PRESSURE] = self.pressure
        result.info["nsymmetry"] = self.nsymmetry
        return result

    @override
    def _string(self) -> str:
        result = AtomicContainner._string(self)
        result += f",{BondAttrMixin._string(self)}"
        result += f",{self.connected_components_number}FRAG"
        return f"{result},{self.__hashstring}"

    @validate_call
    def get_weisfeiler_lehman_hashes(
        self,
        hash_depth: int = 3,
        digest_size: int = 6,
        backend: Literal["igraph", "networkx"] = "igraph",
    ) -> list[str]:
        """Return hash value for each atom."""
        if backend == "igraph":
            return GraphMixinIGraph.get_weisfeiler_lehman_hashes(
                self, hash_depth=hash_depth, digest_size=digest_size
            )
        elif backend == "networkx":
            raise NotImplementedError
        else:
            raise KeyError(
                f"Invalid backend: {backend}. Only "
                '"igraph" & "networkx" are supported.'
            )

    @validate_call(config={"arbitrary_types_allowed": True})
    def update_geometry(
        self,
        geometry: np.ndarray,
        plus_factor: float = 0.5,
        multiply_factor: float = 1,
        infer_order: bool = False,
        return_dict: bool = False,
        charge: int = 0,
    ) -> Self | dict[str, Any]:
        assert geometry.shape == (self.natoms, 3)
        dct = self.model_dump(exclude_none=True)
        dct[GRAPH_KEY.ATOM.POSITION] = np.asarray(geometry)
        conn = dct.pop(GRAPH_KEY.BOND.CONNECTIVITY, None)
        order = dct.pop(GRAPH_KEY.BOND.ORDER, None)
        if conn is not None:
            dct = AtomicContainner._infer_bond(
                dct=dct,
                plus_factor=plus_factor,
                multiply_factor=multiply_factor,
                infer_order=(order is not None) and (infer_order),
                charge=0,
            )
            cn = dct.get(GRAPH_KEY.ATOM.COORDINATION, None)
            tag = dct.get(GRAPH_KEY.ATOM.MOVE_FIX_TAG, None)
            if all(i is not None for i in (cn, tag)):
                cn, tag = (np.asarray(i) for i in (cn, tag))
            else:
                raise ValueError("The `cn` or `tag` is none for Cluster.")
            conn = dct.get(GRAPH_KEY.BOND.CONNECTIVITY)
            _cn = np.asarray(IGraph(len(cn), conn).degree(), dtype=int)
            dct[GRAPH_KEY.ATOM.COORDINATION] = np.where(tag == 0, cn, _cn)
        return dct if return_dict else self.model_validate(dct)

    #########################################################################
    #                          Start of Interface                           #
    #########################################################################

    def convert_as(self, mode: _MODE_TYPE) -> _OBJ_TYPE:
        return getattr(self, f"as_{mode.lower()}")()

    @classmethod
    def convert_from(
        cls,
        obj: _OBJ_TYPE,
        mode: _MODE_TYPE,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
    ) -> Self:
        return getattr(cls, f"from_{mode.lower()}")(
            obj,
            infer_conn=infer_conn,
            infer_order=infer_order,
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            charge=charge,
        )

    #########################################################################
    #                          ASE Atoms Interface                          #
    #########################################################################

    @override
    def as_ase(self) -> Atoms:
        return super().as_ase()

    @classmethod
    @override
    def from_ase(
        cls,
        atoms: Atoms,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
    ) -> Self:
        _allowed_keys_set = set(cls.__pydantic_fields__.keys())
        _allowed_keys_set -= set(AtomicContainner.__pydantic_fields__.keys())
        dct = AtomicContainner.from_ase(atoms).model_dump(exclude_none=True) | {
            k: v for k, v in atoms.info.items() if k in _allowed_keys_set
        }
        if any([infer_conn, infer_order]):
            dct = cls._infer_bond(
                dct,
                plus_factor=plus_factor,
                multiply_factor=multiply_factor,
                infer_order=infer_order,
                charge=charge,
            )
        elif BOND_KEY.CONNECTIVITY not in dct:
            raise ValueError(f"Missing {BOND_KEY.CONNECTIVITY}.")
        return cls.model_validate(dct)

    #########################################################################
    #                          PyG Data Interface                           #
    #########################################################################
    @staticmethod
    def __pygdata_exclude_keys_set() -> set[str]:
        return {
            GRAPH_KEY.ATOM.NUMBER,
            GRAPH_KEY.ATOM.POSITION,
            GRAPH_KEY.BOND.CONNECTIVITY,
            GRAPH_KEY.BOND.ORDER,
        }

    @override
    def as_pygdata(self) -> DataPyG:
        m = sp.coo_matrix(self.MATRIX)
        #  UserWarning: The given NumPy array is not writable, and
        #   PyTorch does not support non-writable tensors. This means
        #   writing to this tensor will result in undefined behavior.
        #   You may want to copy the array to protect its data or make it
        #   writable before converting it to a tensor. This type of warning
        #   will be suppressed for the rest of this program. (Triggered
        #       internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
        conn, order = np.column_stack(m.coords), m.data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = DataPyG(
                pos=torch.from_numpy(self.positions),
                edge_index=torch.from_numpy(conn.astype(int).T),
            )
            if self.order is not None:
                result[GRAPH_KEY.BOND.ORDER] = torch.from_numpy(order)
            result[GRAPH_KEY.ATOM.NUMBER] = torch.from_numpy(self.Z)
            for k, v in self.model_dump(
                mode="python",
                exclude_none=True,
                exclude=self.__pygdata_exclude_keys_set(),
            ).items():
                if isinstance(v, np.ndarray):
                    dtype_name: str = v.dtype.name
                    if dtype_name.startswith("uint"):
                        if dtype_name != "uint8":
                            d = dtype_name[1:]
                            v = v.astype(d)
                    result[k] = torch.from_numpy(v)
                elif np.isscalar(v):
                    result[k] = v
                else:
                    raise TypeError(f"{k}(type={type(v)}: {v}")
            if self.box is not None:
                result[GRAPH_KEY.GRAPH.BOX] = torch.from_numpy(self.box)
        result.validate(raise_on_error=True)
        return result

    @classmethod
    @override
    def from_pygdata(
        cls,
        data: DataPyG,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
    ) -> Self:
        assert data.pos is not None
        assert data.edge_index is not None
        assert GRAPH_KEY.ATOM.NUMBER in data.keys()
        dct = {
            GRAPH_KEY.ATOM.POSITION: data.pos.numpy(force=True),
            GRAPH_KEY.BOND.CONNECTIVITY: data.edge_index.numpy(force=True).T,
        } | {
            k: data[k].numpy(force=True)
            if isinstance(data[k], torch.Tensor)
            else data[k]
            for k in set(cls.__pydantic_fields__.keys()) & set(data.keys())
            if k not in (GRAPH_KEY.ATOM.POSITION, GRAPH_KEY.BOND.CONNECTIVITY)
        }
        if any([infer_conn, infer_order]):
            dct = cls._infer_bond(
                dct,
                plus_factor=plus_factor,
                multiply_factor=multiply_factor,
                infer_order=infer_order,
                charge=charge,
            )
        elif BOND_KEY.CONNECTIVITY not in dct:
            raise ValueError(f"Missing {BOND_KEY.CONNECTIVITY}.")
        return cls.model_validate(dct)

    #########################################################################
    #                          NetworKit Interface                          #
    #########################################################################
    @staticmethod
    def __networkit_dtype_dict() -> dict[str, Any]:
        return {
            GRAPH_KEY.BOND.ORDER: float,
            GRAPH_KEY.ATOM.NUMBER: int,
            GRAPH_KEY.ATOM.IS_OUTER: int,
            GRAPH_KEY.ATOM.COORDINATION: int,
        } | {f"{GRAPH_KEY.ATOM.POSITION}_{i}": float for i in "xyz"}

    @override
    def as_networkit(self) -> NetworKitGraph:
        G = NetworKitGraph(
            n=self.natoms,
            weighted=False,
            directed=False,
            edgesIndexed=False,
        )
        i = self.DF_BONDS[self.DF_BONDS.columns[0]].to_numpy(int)
        j = self.DF_BONDS[self.DF_BONDS.columns[1]].to_numpy(int)
        G.addEdges((i, j))
        for k, ofType in self.__networkit_dtype_dict().items():
            if k in GRAPH_KEY.BOND._DICT.values() and k in self.DF_BONDS:
                setter = G.attachEdgeAttribute(k, ofType)
                v: np.ndarray = self.DF_BONDS[k].to_numpy(copy=False)
                G.forEdges(lambda x: setter.__setitem__(x[3], ofType(v[x[3]])))
            elif k not in GRAPH_KEY.BOND._DICT.values() and k in self.DF_ATOMS:
                setter = G.attachNodeAttribute(k, ofType)
                v: np.ndarray = self.DF_ATOMS[k].to_numpy(copy=False)
                G.forNodes(lambda i: setter.__setitem__(i, ofType(v[i])))

        raise NotImplementedError()
        for k in set(TOTAL_KEY._DICT.values()):
            value = getattr(self, k, None)
            if value is not None:
                G.__setattr__(f"_my_{k}", value)
        return G

    @classmethod
    @override
    def from_networkit(
        cls,
        graph: NetworKitGraph,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
    ) -> Self:
        na, nb = graph.numberOfNodes(), graph.numberOfEdges
        conn = np.asarray([i for i in graph.iterEdges()])
        df_edge, df_node = DataFrame(conn, columns=["i", "j"]), DataFrame()
        for k, ofType in cls.__networkit_dtype_dict().items():
            if k not in GRAPH_KEY.BOND._DICT.values():
                try:
                    getter = graph.getNodeAttribute(k, ofType)
                    v: np.ndarray = np.empty(na, dtype=ofType)
                    graph.forNodes(lambda i: v.__setitem__(i, getter[i]))
                    df_node[k] = v
                except RuntimeError:
                    pass
            else:
                try:
                    getter = graph.getEdgeAttribute(k, ofType)
                    v: np.ndarray = np.empty(nb, dtype=ofType)
                    graph.forEdges(lambda x: v.__setitem__(x[3], getter[x[3]]))
                    df_edge[k] = v
                except RuntimeError:
                    pass
        dct = BondAttrMixin.DF_BONDS_PARSER(df_edge)
        dct |= AtomicContainner.DF_ATOMS_PARSER(df_node) | {
            k: graph.__getattribute__(f"__my_{k}")
            for k in set(TOTAL_KEY._DICT.values())
            if hasattr(graph, f"__my_{k}")
        }
        if any([infer_conn, infer_order]):
            dct = cls._infer_bond(
                dct,
                plus_factor=plus_factor,
                multiply_factor=multiply_factor,
                infer_order=infer_order,
                charge=charge,
            )
        elif BOND_KEY.CONNECTIVITY not in dct:
            raise ValueError(f"Missing {BOND_KEY.CONNECTIVITY}.")
        return cls.model_validate(dct)

    #########################################################################
    #                           NetworkX Interface                          #
    #########################################################################

    @override
    def as_networkx(self) -> NetworkXGraph:
        G = self.as_igraph().to_networkx()
        # Timing:                       incl.     excl.
        # ----------------------------------------------------
        # Natoms=21856,==>ASE:          0.004     0.004   0.1% |
        # Natoms=21856,==>IGraph:       0.301     0.301   5.2% |-|
        # Natoms=21856,==>NetworkX:     2.341     2.341  40.0% |---------------|
        # Natoms=21856,==>PyGData:      0.050     0.050   0.9% |
        # Other:                        0.508     0.508   8.7% |--|
        # ----------------------------------------------------
        # Total:                                  5.846 100.0%

        # G = NetworkXGraph(
        #     **self.model_dump(
        #         mode="python",
        #         include=set(set(TOTAL_KEY._DICT.values())),
        #         exclude_none=True,
        #     )
        # )
        # G.add_nodes_from(
        #     list(range(self.natoms)),
        #     **{k: self.DF_ATOMS[k] for k in self.DF_ATOMS.columns},
        # )
        # G.add_edges_from(
        #     self.DF_BONDS[self.DF_BONDS.columns[:2]].to_numpy(),
        #     **{k: self.DF_BONDS[k] for k in self.DF_BONDS.columns[2:]},
        # )
        #         Timing:                       incl.     excl.
        # ----------------------------------------------------
        # Natoms=21856,==>ASE:          0.003     0.003   0.0% |
        # Natoms=21856,==>IGraph:       0.302     0.302   2.3% ||
        # Natoms=21856,==>NetworkX:     6.093     6.093  45.4% |--------...|
        # Natoms=21856,==>PyGData:      0.044     0.044   0.3% |
        # Other:                        0.532     0.532   4.0% |-|
        # ----------------------------------------------------
        # Total:                                 13.430 100.0%

        return G

    @classmethod
    @override
    def from_networkx(
        cls,
        graph: NetworkXGraph,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
    ) -> Self:
        return cls.from_igraph(
            graph=IGraph.from_networkx(graph),
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            infer_order=infer_order,
            infer_conn=infer_conn,
            charge=charge,
        )

    #########################################################################
    #                          RustworkX Interface                          #
    #########################################################################

    @override
    def as_rustworkx(self) -> RustworkXGraph:
        graph = self.as_igraph()
        # Ref: https://github.com/igraph/python-igraph/blob/main/src/igraph/io/libraries.py#L1-L70
        G = RustworkXGraph(
            multigraph=False,
            attrs={x: graph[x] for x in graph.attributes()},
        )
        G.add_nodes_from([v.attributes() for v in graph.vs])
        G.add_edges_from((e.source, e.target, e.attributes()) for e in graph.es)
        #         Timing:                        incl.     excl.
        # -----------------------------------------------------
        # Natoms=21856,==>ASE:           0.007     0.007   0.1% |
        # Natoms=21856,==>IGraph:        0.310     0.310   4.1% |-|
        # Natoms=21856,==>NetworkX:      2.184     2.184  28.7% |----------|
        # Natoms=21856,==>PyGData:       0.041     0.041   0.5% |
        # Natoms=21856,==>RustworkX:     1.126     1.126  14.8% |-----|
        # Other:                         0.465     0.465   6.1% |-|
        # -----------------------------------------------------

        #         Timing:                        incl.     excl.
        # -----------------------------------------------------
        # Natoms=21856,==>ASE:           0.004     0.004   0.0% |
        # Natoms=21856,==>IGraph:        0.305     0.305   2.4% ||
        # Natoms=21856,==>NetworkX:      1.996     1.996  15.7% |-----|
        # Natoms=21856,==>PyGData:       0.039     0.039   0.3% |
        # Natoms=21856,==>RustworkX:     3.457     3.457  27.2% |----------|
        # Other:                         0.548     0.548   4.3% |-|
        # -----------------------------------------------------
        # G = RustworkXGraph(
        #     multigraph=False,
        #     attrs=self.model_dump(
        #         mode="python",
        #         include=set(set(TOTAL_KEY._DICT.values())),
        #         exclude_none=True,
        #     ),
        # )
        # G.add_nodes_from(self.DF_ATOMS.iterrows())
        # if self.order is not None:
        #     df = self.DF_BONDS[self.DF_BONDS.columns[:3]]
        #     G.extend_from_weighted_edge_list(df.itertuples(index=False))
        # else:
        #     df = self.DF_BONDS[self.DF_BONDS.columns[:2]]
        #     G.extend_from_edge_list(df.itertuples(index=False))
        # Timing:    incl.     excl.
        # -----------------------------
        # E:     0.037     0.037  11.1% |---|
        # G:     0.000     0.000   0.0% |
        # V:     0.295     0.295  88.9% |-----------------------------------|
        # Other:    0.000     0.000   0.0% |
        # -----------------------------

        return G

    @classmethod
    @override
    def from_rustworkx(
        cls,
        graph: RustworkXGraph,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
    ) -> Self:
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
        igraph = IGraph.DataFrame(df_edges, False, df_nodes, True)
        for k, v in graph.attrs.items():  # attr is dict type
            if k in cls.__pydantic_fields__ and v is not None:
                igraph[k] = v
        return cls.from_igraph(
            graph=igraph,
            infer_conn=infer_conn,
            infer_order=infer_order,
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            charge=charge,
        )

    #########################################################################
    #                            IGraph Interface                           #
    #########################################################################

    @override
    def as_igraph(self) -> IGraph:
        G = IGraph.DataFrame(self.DF_BONDS, False, self.DF_ATOMS, True)
        for k in self.__pydantic_fields__:
            v = getattr(self, k, None)
            if v is not None:
                G[k] = v
        return G

    @classmethod
    @override
    def from_igraph(
        cls,
        graph: IGraph,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
    ) -> Self:
        dct = {
            k: graph[k]
            for k in cls.__pydantic_fields__
            if k in graph.attributes()
        }
        dct |= AtomicContainner.DF_ATOMS_PARSER(graph.get_vertex_dataframe())
        dct |= BondAttrMixin.DF_BONDS_PARSER(graph.get_edge_dataframe())
        if any([infer_conn, infer_order]):
            dct = cls._infer_bond(
                dct,
                plus_factor=plus_factor,
                multiply_factor=multiply_factor,
                infer_order=infer_order,
                charge=charge,
            )
        elif BOND_KEY.CONNECTIVITY not in dct:
            raise ValueError(f"Missing {BOND_KEY.CONNECTIVITY}.")
        return cls.model_validate(dct)

    #########################################################################
    #                           End of Interface                            #
    #########################################################################


def benchmark_convert() -> None:
    from ase.cluster import Octahedron
    from ase.utils.timing import Timer

    timer, N = Timer(), 10
    for n in [8, 12, 16, 20, 25, 32]:
        # [344, 1156, 2736, 5340, 10425, 21856]
        obj = GraphContainner.from_ase(Octahedron("Au", n))
        for mode in (
            "ASE",
            "PyGData",
            "IGraph",
            "RustworkX",
            "NetworkX",
            "NetworKit",
        ):
            with timer(f"Natoms={obj.natoms:05d},==>{mode}"):
                for _ in range(N):
                    _obj = obj.convert_as(mode.lower())  # type: ignore
    # Timing:                        incl.     excl.
    # -----------------------------------------------------
    # Natoms=00344,==>ASE:           0.000     0.000   0.0% |
    # Natoms=00344,==>IGraph:        0.012     0.012   0.1% |
    # Natoms=00344,==>NetworKit:     0.026     0.026   0.3% |
    # Natoms=00344,==>NetworkX:      0.035     0.035   0.4% |
    # Natoms=00344,==>PyGData:       0.001     0.001   0.0% |
    # Natoms=00344,==>RustworkX:     0.021     0.021   0.3% |
    # Natoms=01156,==>ASE:           0.000     0.000   0.0% |
    # Natoms=01156,==>IGraph:        0.021     0.021   0.3% |
    # Natoms=01156,==>NetworKit:     0.036     0.036   0.4% |
    # Natoms=01156,==>NetworkX:      0.104     0.104   1.3% ||
    # Natoms=01156,==>PyGData:       0.002     0.002   0.0% |
    # Natoms=01156,==>RustworkX:     0.055     0.055   0.7% |
    # Natoms=02736,==>ASE:           0.000     0.000   0.0% |
    # Natoms=02736,==>IGraph:        0.037     0.037   0.5% |
    # Natoms=02736,==>NetworKit:     0.056     0.056   0.7% |
    # Natoms=02736,==>NetworkX:      0.335     0.335   4.1% |-|
    # Natoms=02736,==>PyGData:       0.003     0.003   0.0% |
    # Natoms=02736,==>RustworkX:     0.112     0.112   1.4% ||
    # Natoms=05340,==>ASE:           0.000     0.000   0.0% |
    # Natoms=05340,==>IGraph:        0.067     0.067   0.8% |
    # Natoms=05340,==>NetworKit:     0.092     0.092   1.1% |
    # Natoms=05340,==>NetworkX:      0.479     0.479   5.8% |-|
    # Natoms=05340,==>PyGData:       0.007     0.007   0.1% |
    # Natoms=05340,==>RustworkX:     0.233     0.233   2.8% ||
    # Natoms=10425,==>ASE:           0.001     0.001   0.0% |
    # Natoms=10425,==>IGraph:        0.126     0.126   1.5% ||
    # Natoms=10425,==>NetworKit:     0.161     0.161   1.9% ||
    # Natoms=10425,==>NetworkX:      1.024     1.024  12.4% |----|
    # Natoms=10425,==>PyGData:       0.011     0.011   0.1% |
    # Natoms=10425,==>RustworkX:     0.432     0.432   5.2% |-|
    # Natoms=21856,==>ASE:           0.002     0.002   0.0% |
    # Natoms=21856,==>IGraph:        0.263     0.263   3.2% ||
    # Natoms=21856,==>NetworKit:     0.322     0.322   3.9% |-|
    # Natoms=21856,==>NetworkX:      2.732     2.732  33.1% |------------|
    # Natoms=21856,==>PyGData:       0.022     0.022   0.3% |
    # Natoms=21856,==>RustworkX:     1.069     1.069  12.9% |----|
    # Other:                         0.366     0.366   4.4% |-|
    # -----------------------------------------------------
    # Total:                                   8.265 100.0%
    # Conclusion:
    #   if      ==> ASE             as  1                   1e4 atoms/ms
    #           ==> PyGData:            11      slower      1e3 atoms/ms
    #           ==> IGraph:             132     slower      1e2 atoms/ms
    #           ==> RustworkX:          534     slower       50 atoms/ms
    #           ==> NetworKit:          ---     slower        ? atoms/ms
    #           ==> NetworkX:           1366    slower        7 atoms/ms
    timer.write()
