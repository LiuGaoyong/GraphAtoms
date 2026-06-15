"""The graph representation of the system."""
# ruff: noqa: F401

from ._bonds import (
    DEFAULT_WH_HASH_DEPTH,
    DEFAULT_WH_HASH_SIZE,
    BondGraph,
)
from ._sysGraph import SysGraph

__all__ = ["BondGraph", "SysGraph"]
