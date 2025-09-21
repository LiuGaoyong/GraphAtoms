# ruff: noqa: D100, D101, D102, D103, D104

import pytest
from ase.cluster import Octahedron

from GraphAtoms.containner import System


@pytest.fixture(scope="session")
def system() -> System:
    return System.from_ase(Octahedron("Cu", 8))
