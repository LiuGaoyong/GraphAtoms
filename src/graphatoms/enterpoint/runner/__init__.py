import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

from typing import Any

import hydra

from graphatoms.enterpoint.config import CONFIG_DIR, Config

from .v2.general.otfkmc import OTFKMC as OnTheFlyKMC
from .v2.general.rxngen import ReactionNetworkGenerator

__all__ = [
    "OnTheFlyKMC",
    "ReactionNetworkGenerator",
    "RayOnTheFlyKMC",
    "RayReactionNetworkGenerator",
    "run",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the Ray runners without importing Ray by default."""
    if name in ("RayOnTheFlyKMC", "RayOTFKMC"):
        from .v2.ray.otfkmc import RayOTFKMC

        return RayOTFKMC
    elif name == "RayReactionNetworkGenerator":
        from .v2.ray.rxngen import RayReactionNetworkGenerator

        return RayReactionNetworkGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@hydra.main(
    config_path=CONFIG_DIR.as_posix(),
    config_name="run",
    version_base=None,
)
def run(cfg: Config) -> None:
    run_type = str(cfg.run_type).lower()
    if str(cfg.parallel).lower() == "ray":
        # The ray backend uses a dedicated actor-based implementation of the
        # v2 runners; importing it lazily keeps ray an optional dependency.
        if run_type == "otfkmc":
            from .v2.ray.otfkmc import RayOTFKMC

            RayOTFKMC(config=cfg).run()
        elif run_type == "rxngen":
            from .v2.ray.rxngen import RayReactionNetworkGenerator

            RayReactionNetworkGenerator(config=cfg).run()
        else:
            raise ValueError(f" run_type `{cfg.run_type}` is not supported.")
    elif run_type == "otfkmc":
        OnTheFlyKMC(config=cfg).run()
    elif run_type == "rxngen":
        ReactionNetworkGenerator(config=cfg).run()
    else:
        raise ValueError(f" run_type `{cfg.run_type}` is not supported.")
