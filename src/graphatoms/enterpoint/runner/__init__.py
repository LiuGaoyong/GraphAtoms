import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

import hydra

from graphatoms.enterpoint.config import CONFIG_DIR, Config

from .v2.general.otfkmc import OTFKMC as OnTheFlyKMC
from .v2.general.rxngen import ReactionNetworkGenerator

# NOTE: the experimental ray actors under `.v2.ray` are intentionally *not*
# imported here.  They import `ray` unconditionally at module import time,
# while ray is only a dev dependency, so importing them would make the
# `graphatoms-run` console script unusable on any installation without ray.
# The `parallel: ray` backend itself lives in
# `graphatoms.enterpoint.parallel` and is selected through the config; these
# actor-based runners are still work in progress.
# from .v2.ray.otfkmc import RayOTFKMC as RayOnTheFlyKMC
# from .v2.ray.rxngen import RayReactionNetworkGenerator


@hydra.main(
    config_path=CONFIG_DIR.as_posix(),
    config_name="run",
    version_base=None,
)
def run(cfg: Config) -> None:
    if str(cfg.run_type).lower() == "otfkmc":
        OnTheFlyKMC(config=cfg).run()
    elif str(cfg.run_type).lower() == "rxngen":
        ReactionNetworkGenerator(config=cfg).run()
    else:
        raise ValueError(f" run_type `{cfg.run_type}` is not supported.")
