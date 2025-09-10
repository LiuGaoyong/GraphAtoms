import itertools
import pathlib
import pickle
from time import perf_counter_ns

import pandas as pd

from GraphAtoms.common.string import compress, compress_string
from GraphAtoms.containner import Cluster

f = pathlib.Path(__file__).parent / "./OctCu8_Cluster.csv"
df = pd.read_csv(f, index_col=0)

for i in df["cluster_json"]:
    for fmt, level in sorted(
        itertools.product(
            ["z", "gz", "bz2"],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    ) + [("snappy", 0)]:
        t0 = perf_counter_ns()
        for _ in range(10):
            b2 = compress_string(
                i,
                format=fmt,  # type: ignore
                compresslevel=level,
            )
        t = int((perf_counter_ns() - t0) / 10)
        clv = len(b2) / len(i)  # type: ignore
        print(
            "json->bytes",
            len(i),
            len(b2),  # type: ignore
            f"Time={t:09d}",
            f"Comperss={clv:.2f}",
            fmt,
            level,
            sep="\t",
        )

        c = Cluster.model_validate_json(i)
        obj = c.model_dump(exclude_none=True, exclude={"hash"})
        b = pickle.dumps(obj)
        t0 = perf_counter_ns()
        for _ in range(10):
            b2 = compress(
                b,
                format=str(fmt),  # type: ignore
                compresslevel=level,
            )
        t = int((perf_counter_ns() - t0) / 10)
        clv = len(b2) / len(b)  # type: ignore
        print(
            "dict->bytes",
            len(b),
            len(b2),  # type: ignore
            f"Time={t:09d}",
            f"Comperss={clv:.2f}",
            fmt,
            level,
            sep="\t",
        )

        b = pickle.dumps(c)
        t0 = perf_counter_ns()
        for _ in range(10):
            b2 = compress(
                b,
                format=str(fmt),  # type: ignore
                compresslevel=level,
            )
        t = int((perf_counter_ns() - t0) / 10)
        clv = len(b2) / len(b)  # type: ignore
        print(
            "obj->bytes",
            len(b),
            len(b2),  # type: ignore
            f"Time={t:09d}",
            f"Comperss={clv:.2f}",
            fmt,
            level,
            sep="\t",
        )
        print("-" * 80)
