from typing import override

from ase import Atoms

from graphatoms.enterpoint.runner.general.rxngen import (
    ReactionNetworkGenerator as _Rxngen,
)
from graphatoms.system.system import System

from ._expl import ExplorationRayBase

__all__ = ["RayReactionNetworkGenerator"]


class RayReactionNetworkGenerator(ExplorationRayBase, _Rxngen):
    """Reaction-network generation with an actor-owned network."""

    @override
    def explore(
        self,
        system: System | Atoms | None = None,
    ) -> tuple[bool, float]:
        return super().explore(system)
