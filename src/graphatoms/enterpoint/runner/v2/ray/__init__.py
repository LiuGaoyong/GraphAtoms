"""The Ray-native runners.

Importing this package does not require :mod:`ray`: the Ray modules are only
imported when Ray is installed, and a descriptive
:class:`~._optional.RayNotInstalledError` is raised when a Ray runner is
requested without it.  The public runner package keeps these imports lazy so a
Ray-free installation can still import ``graphatoms.enterpoint.runner``.
"""

from __future__ import annotations

from typing import Any

from ._optional import (
    RAY_INSTALL_HINT,
    RayNotInstalledError,
    require_ray,
)

__all__ = [
    "RAY_INSTALL_HINT",
    "RayNetworkActor",
    "RayNotInstalledError",
    "RayOTFKMC",
    "RayReactionNetworkGenerator",
    "create_network_actor",
]

_RAY_EXPORTS = (
    "RayNetworkActor",
    "RayOTFKMC",
    "RayReactionNetworkGenerator",
    "create_network_actor",
)


def __getattr__(name: str) -> Any:
    """Import the Ray runner only when Ray is installed and requested."""
    if name not in _RAY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    require_ray()
    if name in ("RayNetworkActor", "create_network_actor"):
        from ._actor import RayNetworkActor, create_network_actor

        return {
            "RayNetworkActor": RayNetworkActor,
            "create_network_actor": create_network_actor,
        }[name]
    if name == "RayOTFKMC":
        from .otfkmc import RayOTFKMC

        return RayOTFKMC
    from .rxngen import RayReactionNetworkGenerator

    return RayReactionNetworkGenerator
