from time import perf_counter

import pytest
from ase.cluster import Octahedron

from GraphAtoms.system import System


@pytest.mark.parametrize("n", range(3, 15))
def test_speed(n: int) -> None:
    start = perf_counter()
    a = Octahedron("Cu", n)
    print(f"Created Atoms with {len(a)} atoms")
    sys = System.from_ase(a, parse_bonds={"method": "raw"})
    print(f"Created System with {sys.natoms} atoms and {sys.nbonds} bonds")
    end = perf_counter()
    print(f"Creating Atoms with {len(a)} atoms took {end - start:.6f} seconds")
