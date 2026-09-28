import os

os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra

from graphatoms.enterpoint.config import CONFIG_DIR, Config

from .general.otfkmc import OTFKMC as OnTheFlyKMC
from .general.rxngen import ReactionNetworkGenerator
from .ray.otfkmc import RayOTFKMC as RayOnTheFlyKMC
from .ray.rxngen import RayReactionNetworkGenerator


@hydra.main(
    config_path=CONFIG_DIR.as_posix(),
    config_name="run",
    version_base=None,
)
def run(cfg: Config) -> None:
    if str(cfg.run_type).lower() == "otfkmc":
        OnTheFlyKMC(config=cfg).run()
        return
        if str(cfg.parallel).lower() == "ray":
            RayOnTheFlyKMC(config=cfg).run()
        else:
            OnTheFlyKMC(config=cfg).run()
    elif str(cfg.run_type).lower() == "rxngen":
        raise NotImplementedError("rxngen is not implemented.")
        if str(cfg.parallel).lower() == "ray":
            RayReactionNetworkGenerator(config=cfg).run()
        else:
            ReactionNetworkGenerator(config=cfg).run()
    else:
        raise ValueError(f" run_type `{cfg.run_type}` is not supported.")
