from pydantic import model_validator
from typing_extensions import Self

from .._convert import Convert
from .._graph import GraphMixin


class SysGraph(Convert, GraphMixin):
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


class System(SysGraph):
    """Full periodic or non-periodic system."""

    @model_validator(mode="after")
    def __some_keys_should_be_none(self) -> Self:
        msg = "The `{:s}` should be None for System."
        assert self.coordination is None, msg.format("coordination")
        assert not self.is_gas, "The `is_gas` should be False for System."
        return self


# def benchmark_convert() -> None:
if __name__ == "__main__":
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
            "PyGData",
            "IGraph",
            "RustworkX",
            "NetworkX",
        ):
            t0 = perf_counter()
            for _ in range(10):
                _obj = obj.convert_to(mode.lower())  # type: ignore
                obj.convert_from(_obj, mode.lower())
            result[n][f"{mode}(ms)"] = (perf_counter() - t0) * 1000 / 10
    df = DataFrame(result).T
    df["natoms"] = df["natoms"].astype(int)
    print(df)
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
