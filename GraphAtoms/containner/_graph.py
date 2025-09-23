import warnings
from functools import cached_property
from typing import Any, Literal, override

import numpy as np
import torch
from ase import Atoms
from igraph import Graph as IGraph
from networkx import Graph as NetworkXGraph
from pandas import DataFrame
from pandas import concat as pd_concat
from pydantic import validate_call
from rustworkx import PyGraph as RustworkXGraph
from scipy import sparse as sp
from torch_geometric.data import Data as DataPyG
from typing_extensions import Self

from GraphAtoms.containner._aMixin import ATOM_KEY
from GraphAtoms.containner._atomic import AtomsWithBoxEng
from GraphAtoms.containner._g2DataPyG import GraphMixinPyG
from GraphAtoms.containner._g2IGraph import GraphMixinIGraph
from GraphAtoms.containner._g2NetworkX import GraphMixinNetworkX
from GraphAtoms.containner._g2RustworkX import GraphMixinRustworkX
from GraphAtoms.containner._gBonded import BOND_KEY, BondsWithComp
from GraphAtoms.utils import rdtool as rdutils
from GraphAtoms.utils.string import hash as hash_string


class Graph(
    AtomsWithBoxEng,
    BondsWithComp,
    GraphMixinIGraph,
    GraphMixinPyG,
    GraphMixinNetworkX,
    GraphMixinRustworkX,
):
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
        elif not BondsWithComp.__eq__(self, other):
            return False
        elif not AtomsWithBoxEng.__eq__(self, other):
            return False
        else:
            return self.hash == other.hash

    @override
    def __hash__(self) -> int:
        return super().__hash__()

    @cached_property
    def hash(self) -> str:
        labels = sorted(self.get_weisfeiler_lehman_hashes())
        return hash_string(",".join(labels), digest_size=8)

    @property
    def CN(self) -> np.ndarray:
        try:
            return super().CN
        except Exception:
            return super().CN_MATRIX

    @cached_property
    @override
    def RDMol(self) -> rdutils.RDMol:
        return rdutils._get_rdmol_with_bonds(
            numbers=self.Z,
            geometry=self.R,
            source=self.source,
            target=self.target,
            order=self.order if self.order is not None else None,
            infer_order=False,
            charge=0,
        )

    @cached_property
    def __SASA(self) -> np.ndarray:
        return np.asarray(rdutils.get_atomic_sasa(self.RDMol))

    def get_atomic_sasa(self) -> np.ndarray:
        return self.__SASA

    @override
    def _string(self) -> str:
        result = AtomsWithBoxEng._string(self)
        result += f",{BondsWithComp._string(self)}"
        result += f",{self.connected_components_number}FRAG"
        return f"{result},{self.hash}"

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

    # @validate_call(config={"arbitrary_types_allowed": True})
    # def update_geometry(
    #     self,
    #     geometry: np.ndarray,
    #     plus_factor: float = 0.5,
    #     multiply_factor: float = 1,
    #     infer_order: bool = False,
    #     return_dict: bool = False,
    #     charge: int = 0,
    # ) -> Self | dict[str, Any]:
    #     assert geometry.shape == (self.natoms, 3)
    #     dct = self.model_dump(exclude_none=True)
    #     dct[GRAPH_KEY.ATOM.POSITION] = np.asarray(geometry)
    #     conn = dct.pop(GRAPH_KEY.BOND.CONNECTIVITY, None)
    #     order = dct.pop(GRAPH_KEY.BOND.ORDER, None)
    #     if conn is not None:
    #         dct = AtomicContainner._infer_bond(
    #             dct=dct,
    #             plus_factor=plus_factor,
    #             multiply_factor=multiply_factor,
    #             infer_order=(order is not None) and (infer_order),
    #             charge=0,
    #         )
    #         cn = dct.get(GRAPH_KEY.ATOM.COORDINATION, None)
    #         tag = dct.get(GRAPH_KEY.ATOM.MOVE_FIX_TAG, None)
    #         if all(i is not None for i in (cn, tag)):
    #             cn, tag = (np.asarray(i) for i in (cn, tag))
    #         else:
    #             raise ValueError("The `cn` or `tag` is none for Cluster.")
    #         conn = dct.get(GRAPH_KEY.BOND.CONNECTIVITY)
    #         _cn = np.asarray(IGraph(len(cn), conn).degree(), dtype=int)
    #         dct[GRAPH_KEY.ATOM.COORDINATION] = np.where(tag == 0, cn, _cn)
    #     return dct if return_dict else self.model_validate(dct)

    #########################################################################
    #                          Start of Interface                           #
    #########################################################################

    @override
    def convert_to(self, format="bytes", **kw) -> Any:
        return super().convert_to(format=format, **kw)

    @classmethod
    @override
    def convert_from(
        cls,
        data: Any,
        format="bytes",
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
        **kw,
    ) -> Self:
        return super().convert_from(
            data=data,
            format=format,
            infer_conn=infer_conn,
            infer_order=infer_order,
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            charge=charge,
            **kw,
        )

    #########################################################################
    #                          ASE Atoms Interface                          #
    #########################################################################
    @override
    def to_ase(self) -> Atoms:
        atoms = super().to_ase()
        for k in BOND_KEY._DICT.values():
            v = getattr(self, k, None)
            if v is not None:
                atoms.info[k] = v
        return atoms

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
        **kw,
    ) -> Self:
        obj = AtomsWithBoxEng.from_ase(atoms)
        dct = obj.model_dump(exclude_none=True)
        for k in BOND_KEY._DICT.values():
            if k in atoms.info:
                dct[k] = atoms.info[k]
        if any([infer_conn, infer_order]):
            dct.update(
                BondsWithComp.infer_bond_as_dict(
                    obj,
                    plus_factor=plus_factor,
                    multiply_factor=multiply_factor,
                    infer_order=infer_order,
                    charge=charge,
                )
            )
        for k in (BOND_KEY.SOURCE, BOND_KEY.TARGET):
            if k not in dct:
                raise ValueError(f"Missing `{k}`.")
        return cls.model_validate(dct)

    #########################################################################
    #                          PyG Data Interface                           #
    #########################################################################
    @override
    def to_pygdata(self) -> DataPyG:
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
                result[BOND_KEY.ORDER] = torch.from_numpy(order)
            result[ATOM_KEY.NUMBER] = torch.from_numpy(self.Z)
            for k, v in self.model_dump(
                mode="python",
                exclude_none=True,
                exclude=(
                    {ATOM_KEY.NUMBER, ATOM_KEY.POSITION}
                    | set(BOND_KEY._DICT.values())
                ),
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
        result.validate(raise_on_error=True)
        return result

    @classmethod
    @override
    def from_pygdata(
        cls,
        data: DataPyG,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
        **kw,
    ) -> Self:
        assert data.pos is not None
        assert data.edge_index is not None
        assert ATOM_KEY.NUMBER in data.keys()
        dct = {
            ATOM_KEY.POSITION: data.pos.numpy(force=True),
            BOND_KEY.SOURCE: data.edge_index[0].numpy(force=True),
            BOND_KEY.TARGET: data.edge_index[1].numpy(force=True),
        } | {
            k: data[k].numpy(force=True)
            if isinstance(data[k], torch.Tensor)
            else data[k]
            for k in set(cls.__pydantic_fields__.keys()) & set(data.keys())
            if k not in (ATOM_KEY.POSITION, BOND_KEY.SOURCE, BOND_KEY.TARGET)
        }
        obj = cls.model_validate(dct)
        if infer_order:
            order = cls.infer_bond_as_dict(
                obj,
                plus_factor=plus_factor,
                multiply_factor=multiply_factor,
                infer_order=infer_order,
                charge=charge,
            )[BOND_KEY.ORDER]
            object.__setattr__(obj, BOND_KEY.ORDER, order)
        return obj

    #########################################################################
    #                           NetworkX Interface                          #
    #########################################################################
    @override
    def to_networkx(self) -> NetworkXGraph:
        G = self.to_igraph().to_networkx()
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
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
        **kw,
    ) -> Self:
        return cls.from_igraph(
            graph=IGraph.from_networkx(graph),
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            infer_order=infer_order,
            charge=charge,
        )

    #########################################################################
    #                          RustworkX Interface                          #
    #########################################################################

    @override
    def to_rustworkx(self) -> RustworkXGraph:
        graph = self.to_igraph()
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
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
        **kw,
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
            infer_order=infer_order,
            multiply_factor=multiply_factor,
            plus_factor=plus_factor,
            charge=charge,
        )

    #########################################################################
    #                            IGraph Interface                           #
    #########################################################################

    @override
    def to_igraph(self) -> IGraph:
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
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
        **kw,
    ) -> Self:
        dct = {
            k: graph[k]
            for k in cls.__pydantic_fields__
            if k in graph.attributes()
        }
        dct |= cls.DF_ATOMS_PARSER(graph.get_vertex_dataframe())
        dct |= cls.DF_BONDS_PARSER(graph.get_edge_dataframe())
        obj = cls.model_validate(dct)
        if infer_order:
            order = cls.infer_bond_as_dict(
                obj,
                plus_factor=plus_factor,
                multiply_factor=multiply_factor,
                infer_order=infer_order,
                charge=charge,
            )[BOND_KEY.ORDER]
            object.__setattr__(obj, BOND_KEY.ORDER, order)
        return obj

    #########################################################################
    #                           End of Interface                            #
    #########################################################################


# def benchmark_convert() -> None:
if __name__ == "__main__":
    from ase.cluster import Octahedron
    from ase.utils.timing import Timer

    timer, N = Timer(), 10
    for n in [8, 12, 16, 20, 25, 32]:
        # [344, 1156, 2736, 5340, 10425, 21856]
        obj = Graph.from_ase(Octahedron("Au", n))
        for mode in (
            "ASE",
            "PyGData",
            "IGraph",
            "RustworkX",
            "NetworkX",
        ):
            with timer(f"Natoms={obj.natoms:05d},==>{mode}"):
                for _ in range(N):
                    _obj = obj.convert_to(mode.lower())  # type: ignore
    # Timing:                        incl.     excl.
    # -----------------------------------------------------
    # Natoms=00344,==>ASE:           0.001     0.001   0.0% |
    # Natoms=00344,==>IGraph:        0.126     0.126   0.5% |
    # Natoms=00344,==>NetworkX:      0.119     0.119   0.5% |
    # Natoms=00344,==>PyGData:       0.004     0.004   0.0% |
    # Natoms=00344,==>RustworkX:     0.091     0.091   0.4% |
    # Natoms=01156,==>ASE:           0.002     0.002   0.0% |
    # Natoms=01156,==>IGraph:        0.154     0.154   0.6% |
    # Natoms=01156,==>NetworkX:      0.355     0.355   1.4% ||
    # Natoms=01156,==>PyGData:       0.006     0.006   0.0% |
    # Natoms=01156,==>RustworkX:     0.227     0.227   0.9% |
    # Natoms=02736,==>ASE:           0.002     0.002   0.0% |
    # Natoms=02736,==>IGraph:        0.308     0.308   1.2% |
    # Natoms=02736,==>NetworkX:      0.998     0.998   3.9% |-|
    # Natoms=02736,==>PyGData:       0.007     0.007   0.0% |
    # Natoms=02736,==>RustworkX:     0.467     0.467   1.8% ||
    # Natoms=05340,==>ASE:           0.003     0.003   0.0% |
    # Natoms=05340,==>IGraph:        0.584     0.584   2.3% ||
    # Natoms=05340,==>NetworkX:      1.670     1.670   6.5% |--|
    # Natoms=05340,==>PyGData:       0.011     0.011   0.0% |
    # Natoms=05340,==>RustworkX:     1.002     1.002   3.9% |-|
    # Natoms=10425,==>ASE:           0.005     0.005   0.0% |
    # Natoms=10425,==>IGraph:        1.289     1.289   5.0% |-|
    # Natoms=10425,==>NetworkX:      3.061     3.061  12.0% |----|
    # Natoms=10425,==>PyGData:       0.024     0.024   0.1% |
    # Natoms=10425,==>RustworkX:     1.737     1.737   6.8% |--|
    # Natoms=21856,==>ASE:           0.012     0.012   0.0% |
    # Natoms=21856,==>IGraph:        2.384     2.384   9.3% |---|
    # Natoms=21856,==>NetworkX:      6.324     6.324  24.8% |---------|
    # Natoms=21856,==>PyGData:       0.044     0.044   0.2% |
    # Natoms=21856,==>RustworkX:     3.742     3.742  14.7% |-----|
    # Other:                         0.771     0.771   3.0% ||
    # -----------------------------------------------------
    # Total:                                  25.528 100.0%
    # Conclusion:
    #   if      ==> ASE             as  1                   1e4 atoms/ms
    #           ==> PyGData:            11      slower      1e3 atoms/ms
    #           ==> IGraph:             132     slower      1e2 atoms/ms
    #           ==> RustworkX:          534     slower       50 atoms/ms
    #           ==> NetworkX:           1366    slower        7 atoms/ms
    timer.write()
