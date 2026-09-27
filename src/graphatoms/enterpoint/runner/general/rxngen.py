"""The steps for the on-the-fly KMC simulation."""

from ._expl import ExplorationBase

__all__ = ["ReactionNetworkGenerator"]


class ReactionNetworkGenerator(ExplorationBase):
    """The class for the on-the-fly KMC simulation."""
