# ruff: noqa: D100, D101, D102, D103, D104

from pathlib import Path
from typing import Any

import pytest
from ase.cluster import Octahedron
from pandas import read_csv

from GraphAtoms.containner import System


@pytest.fixture(scope="session")
def system() -> System:
    return System.from_ase(Octahedron("Cu", 8))


@pytest.fixture(scope="session")
def OctCu8_Cluster_ListofDict() -> list[dict[str, Any]]:
    f = Path(__file__).parent / "OctCu8_Cluster.csv"
    rows = read_csv(f, index_col=0).iterrows()
    return [row.to_dict() for _, row in rows]
