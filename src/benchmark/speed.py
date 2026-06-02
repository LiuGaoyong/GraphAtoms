from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pytest
from ase.cluster import Octahedron

from graphatoms.system import System


@pytest.mark.parametrize("n", range(3, 20))
def test_speed(n: int) -> None:  # noqa: D103
    print()
    start = perf_counter()
    a = Octahedron("Cu", n)
    print(f"Created Atoms with {len(a)} atoms")
    sys = System.from_ase(a, parse_bonds={"method": "raw"})
    print(f"Created System with {sys.natoms} atoms and {sys.nbonds} bonds")
    end = perf_counter()
    print(f"Creating Atoms with {len(a)} atoms took {end - start:.6f} seconds")

    with TemporaryDirectory() as tmpdir:
        for fmt in ("npz", "pickle"):
            fname = Path(tmpdir) / f"test.{fmt}"
            sys.to_dict
            sys.write(fname, exclude_bond_attibutes=True)
            print(
                f"Write {fmt} file {fname} with size "  #
                f"{fname.stat().st_size} bytes"
            )
    print("-" * 32)
