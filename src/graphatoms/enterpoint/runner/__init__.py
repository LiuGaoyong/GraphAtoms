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
    "run",
    "run_ray",
]


def _import_ray_runners():
    """Import the Ray runners, raising a descriptive error without Ray."""
    from .v2.ray._optional import (
        RAY_INSTALL_HINT,
        RayNotInstalledError,
    )

    try:
        from .v2.ray import RayOTFKMC, RayReactionNetworkGenerator
    except RayNotInstalledError as e:  # pragma: no cover - install dependent
        raise ImportError(RAY_INSTALL_HINT) from e
    return RayOTFKMC, RayReactionNetworkGenerator


def __getattr__(name: str) -> Any:
    """Lazily expose the Ray runners without importing Ray by default."""
    if name in ("RayOnTheFlyKMC", "RayOTFKMC"):
        RayOTFKMC, _ = _import_ray_runners()
        return RayOTFKMC
    elif name == "RayReactionNetworkGenerator":
        _, RayReactionNetworkGenerator = _import_ray_runners()
        return RayReactionNetworkGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_ray(cfg: Config) -> None:
    """Run the Ray-native runner for the configured ``run_type``."""
    run_type = str(cfg.run_type).lower()
    RayOTFKMC, RayReactionNetworkGenerator = _import_ray_runners()
    if run_type == "otfkmc":
        runner = RayOTFKMC(config=cfg)
    elif run_type == "rxngen":
        runner = RayReactionNetworkGenerator(config=cfg)
    else:
        raise ValueError(f" run_type `{cfg.run_type}` is not supported.")
    try:
        runner.run()
    finally:
        runner.close()


@hydra.main(
    config_path=CONFIG_DIR.as_posix(),
    config_name="run",
    version_base=None,
)
def run(cfg: Config) -> None:
    run_type = str(cfg.run_type).lower()
    if str(cfg.parallel).lower() == "ray":
        # The ray backend uses a dedicated Ray-native runner.  ``run_ray``
        # imports Ray lazily and raises a descriptive error when the optional
        # dependency is missing.
        run_ray(cfg)
    elif run_type == "otfkmc":
        OnTheFlyKMC(config=cfg).run()
    elif run_type == "rxngen":
        ReactionNetworkGenerator(config=cfg).run()
    else:
        raise ValueError(f" run_type `{cfg.run_type}` is not supported.")
