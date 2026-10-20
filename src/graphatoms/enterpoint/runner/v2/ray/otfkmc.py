from typing import override

from ase import Atoms

from graphatoms.enterpoint.runner.general.otfkmc import (
    OTFKMC as _OTFKMC,
)
from graphatoms.enterpoint.runner.general.otfkmc import OTFKMCInfo
from graphatoms.system.system import System

from ._expl import ExplorationRayBase

__all__ = ["OTFKMCInfo", "RayOTFKMC"]


class RayOTFKMC(ExplorationRayBase, _OTFKMC):
    """On-the-fly KMC with an actor-owned reaction network."""

    @override
    def explore(
        self,
        system: System | Atoms | None = None,
    ) -> tuple[bool, float]:
        return super().explore(system)
