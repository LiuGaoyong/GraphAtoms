from pathlib import Path

import pytest
from ase.cluster import Octahedron

from graphatoms.system import SysGraph, System

this_dir = Path(__file__).parent
data_dir = this_dir.parent / "tests-data-for-match"


@pytest.fixture(scope="module")
def sys() -> System:
    return System.from_ase(Octahedron("Pd", 8))


@pytest.mark.parametrize("p", list(data_dir.rglob("*.npz")))
def test_match(p: Path, sys: System) -> None:
    print(p)
    sub = SysGraph.read_npz(p)
    print(sys.CN)
    print(sub.CN)
    print(sys.get_match_mode(sub))
    print("-" * 32)
