import numpy as np
from igraph import Graph
from igraph.utils import numpy_to_contiguous_memoryview

from graphatoms.system import SysGraph


def func(obj: SysGraph) -> Graph:
    natoms, nbonds = obj.natoms, obj.nbonds
    if nbonds == 0:
        g = Graph(n=natoms)
    else:
        g = Graph(n=natoms, edges=numpy_to_contiguous_memoryview(obj.pair))
    for k, v in obj.to_dict().items():
        if k == "pair":
            continue
        else:
            if isinstance(v, np.ndarray):
                v0 = numpy_to_contiguous_memoryview(v)
            else:
                v0 = v
            if v.shape[0] == nbonds:
                g.es[k] = v0
            elif v.shape[0] == natoms:
                print(k)
                if k == "positions":
                    continue
                else:
                    g.vs[k] = v0
            else:
                g[k] = v
    print(g.summary())
    return g


def test_igraph():
    from collections import defaultdict
    from time import perf_counter

    from ase.cluster import Octahedron
    from pandas import DataFrame

    result = defaultdict(dict)
    for n in [8, 12, 16, 20, 25, 32]:
        # [344, 1156, 2736, 5340, 10425, 21856]
        obj = SysGraph.from_ase(Octahedron("Au", n))
        result[n]["natoms"] = len(obj.numbers)

        dct = obj.to_dict()
        print(dct)

        t0 = perf_counter()
        for _ in range(10):
            func(obj)
        result[n]["IGraph2"] = (perf_counter() - t0) * 1000 / 10

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
                # obj.convert_from(_obj, mode.lower())
            result[n][f"{mode}(ms)"] = (perf_counter() - t0) * 1000 / 10
    df = DataFrame(result).T
    df["natoms"] = df["natoms"].astype(int)
    print("\n", df)
