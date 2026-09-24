from pathlib import Path

import numpy as np
import pytest
from ase.cluster import Octahedron

from graphatoms.system import SysGraph, System
from graphatoms.system.bonds import matchmode2nmatch

this_dir = Path(__file__).parent
data_dir = this_dir.parent / "tests-datasets" / "for-match"


@pytest.fixture(scope="module")
def sys() -> System:
    return System.from_ase(Octahedron("Pd", 8))


@pytest.mark.parametrize("p", list(data_dir.rglob("*.npz")))
def test_match(p: Path, sys: System) -> None:
    print(p)
    sub = SysGraph.read_npz(p)
    print(sys.CN)
    print(sub.CN)
    m = sys.get_match_mode(sub)
    print(m)
    if isinstance(m, np.ndarray):
        print(matchmode2nmatch(m))
    print(sys.get_match_mode(sub, only_count=True))
    print("-" * 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
