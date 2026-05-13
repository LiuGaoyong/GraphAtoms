"""The module for the definition of System, Cluster and Gas.

They are the main classes for the `GraphAtoms` package.
    Gas         : for the gas molecule.
    System      : for the whole system.
    Cluster     : for the part of the system.
They are all inherited from the `SysGraph` class, which is
the base class based on Graph Theory.
"""

# ruff: noqa: F401
from ._base._engMixin import EnergeticsMixin
from ._graph._igraph import (
    DEFAULT_WH_HASH_DEPTH,
    DEFAULT_WH_HASH_SIZE,
)
from ._sys._sysAllThing import SysGraph, System
from ._sys._sysCluster import Cluster
from ._sys._sysGas import Gas

__all__ = [
    "SysGraph",
    "System",
    "Cluster",
    "Gas",
]
