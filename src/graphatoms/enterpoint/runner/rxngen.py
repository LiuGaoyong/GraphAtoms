"""The steps for the on-the-fly KMC simulation."""

from ._base import ExplorationABC

__all__ = ["ReactionNetworkGenerator"]


class ReactionNetworkGenerator(ExplorationABC):
    """The class for the on-the-fly KMC simulation."""
